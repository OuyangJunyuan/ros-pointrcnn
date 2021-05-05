#!/home/ou/software/anaconda3/envs/prcnn/bin/python
# setting some path
from prcnn_import import *

# utils
import numpy as np
import os
import random

# ros
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# pytorch
import torch
import torch.nn.functional as F

# pointrcnn
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg, cfg_from_file
import lib.utils.calibration as calibration
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils


# Load checkpoint
def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    if os.path.isfile(filename):
        print("loading from checkpoint", filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("Done")
    else:
        raise FileNotFoundError

    return it, epoch


def cropped_roi(pts_lidar, roi):
    x_range, y_range, z_range = roi
    pts_x, pts_y, pts_z = pts_lidar[:, 0], pts_lidar[:, 1], pts_lidar[:, 2]
    range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                 & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                 & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
    pts_lidar = pts_lidar[range_flag, :]
    return pts_lidar


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """

    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom


def transform_pointcloud(pts_lidar, tf):
    """

    :param pts_lidar: (N,2/3)
    :param tf: [4,4]
    :return: (N,2/3)
    """

    pts_lidar_hom = cart_to_hom(pts_lidar)
    pts_rect = np.dot(pts_lidar_hom, tf.T)
    return pts_rect


def transform_point(pt, tf):
    pts_rect = np.dot(tf[:3, :3], pt) + tf[:3, 3]
    return pts_rect[0:3]


def numpy2pc2(pts_input, tf_frame):
    pc2 = np.zeros(pts_input.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
    pc2['x'] = pts_input[:, 0]
    pc2['y'] = pts_input[:, 1]
    pc2['z'] = pts_input[:, 2]
    pc2 = ros_numpy.msgify(PointCloud2, pc2)
    pc2.header.frame_id = tf_frame
    return pc2


def threshold_score(bbox3d, scores, value):
    thr_score = []
    scores = np.squeeze(scores)  # make sure they are 1D
    idx_kept = []
    for i, score in enumerate(scores):
        if score > value:
            idx_kept.append(i)
            thr_score.append(score)
    thr_bbox3d = np.zeros((len(idx_kept), 7))
    for idx, value in enumerate(idx_kept):
        thr_bbox3d[idx, :] = bbox3d[value, :]
    thr_score = np.expand_dims(thr_score, axis=1)  # return it back to original shape
    return thr_bbox3d, thr_score


class ROSPointRCNN(object):
    def __init__(self):
        # dnn
        cfg_from_file(cfg_file)
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        cfg.RPN.LOC_XZ_FINE = False
        self.pc_roi = [[-25, 25], [-3, 2], [-25, 25]]
        self.down_sample = {'axis': int(0), 'depth': self.pc_roi[0][1] / 2}  # [axis,depth]
        self.mode = 'TEST'
        with torch.no_grad():
            self.model = PointRCNN(num_classes=2, use_xyz=True, mode=self.mode)
            self.model.cuda()
            self.model.eval()
            load_checkpoint(model=self.model, optimizer=None, filename=pointrcnn_weight)

        # ros
        self.pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_cb, queue_size=1, buff_size=2 ** 24)

        self.pc_pub = rospy.Publisher(pack_name + "/networks_input", PointCloud2, queue_size=1) if is_viz else None
        self.mk_pub = rospy.Publisher(pack_name + "/networks_output", MarkerArray, queue_size=1) if is_viz else None
        self.Tr_velo_kitti_cam = np.array([0.0, - 1.0, 0.0, 0.0,
                                           0.0, 0.0, -1.0, 1.5,
                                           1.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0]).reshape(4, 4) if is_tf else np.identity(4)

    def pc_cb(self, data):
        pts_input = self.extract_networks_input_from_pc2rosmsg(data)
        if self.pc_pub is not None:
            self.pc_pub.publish(numpy2pc2(pts_input, data.header.frame_id))

        np.random.seed(666)
        with torch.no_grad():
            # 准备输入数据
            MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            inputs = torch.unsqueeze(inputs, 0)

            # 模型推理
            input_data = {'pts_input': inputs}
            ret_dict = self.model(input_data)

            # 分析结果
            batch_size = 1
            roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M) 提案置信度预测
            roi_boxes3d = ret_dict['rois']  # (B, M, 7) 提案框
            seg_result = ret_dict['seg_result'].long()  # (B, N) 前景点分割

            rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])  # (B, M, n) bin分类结果
            rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C) res回归结果

            # 解算3D BBOx
            anchor_size = MEAN_SIZE
            pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                              anchor_size=anchor_size,
                                              loc_scope=cfg.RCNN.LOC_SCOPE,
                                              loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                              num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                              get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                              loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                              get_ry_fine=True).view(batch_size, -1, 7)

            # cfg.SCORE_THRESH 置信度阈值
            if rcnn_cls.shape[2] == 1:
                batch_raw_scores = rcnn_cls  # (B, M, 1)
                batch_norm_scores = torch.sigmoid(batch_raw_scores)  # (B,M,1)
                batch_pred_classes = (batch_norm_scores > cfg.RCNN.SCORE_THRESH).long()  # (B,M,1)
            else:
                batch_pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
                batch_raw_scores = rcnn_cls[:, batch_pred_classes]
                batch_norm_scores = F.softmax(rcnn_cls, dim=1)[:, batch_pred_classes]

            # scores threshold
            inds = batch_norm_scores > cfg.RCNN.SCORE_THRESH
            for batch in range(batch_size):
                inds_in_each_batch = inds[batch].view(-1)
                if inds_in_each_batch.sum() == 0:  # batch 内没有超过阈值的3dbbox
                    continue

                pred_boxes3d_in_each_batch = pred_boxes3d[batch, inds_in_each_batch]
                raw_scores_in_each_batch = batch_raw_scores[batch, inds_in_each_batch]
                norm_scores_in_each_batch = batch_norm_scores[batch, inds_in_each_batch]

                # 非极大值抑制
                boxes_bev_in_each_batch = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_in_each_batch)
                keep_idx = iou3d_utils.nms_gpu(boxes_bev_in_each_batch,
                                               raw_scores_in_each_batch,
                                               cfg.RCNN.NMS_THRESH).view(-1)
                pred_boxes3d_in_each_batch = pred_boxes3d_in_each_batch[keep_idx]
                raw_scores_in_each_batch = raw_scores_in_each_batch[keep_idx]

                output = {'boxes3d': pred_boxes3d_in_each_batch.cpu().numpy(),
                          'scores': raw_scores_in_each_batch.cpu().numpy()}
                self.visualize(output, data.header.frame_id)

    def visualize(self, result, frame_id):
        boxes = result['boxes3d']
        scores = result['scores']

        self.visualize_lidar_plane(boxes, frame_id)
        print("Number of detections pr msg: ", boxes.shape[0])

    def visualize_lidar_plane(self, bbox3d, frame_id):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.header.stamp = rospy.Time.now()

        # marker scale (scale y and z not used due to being linelist)
        marker.scale.x = 0.08
        # marker color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.points = []
        corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6]
        corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)  # (N,8,3)
        for box_nr in range(corners3d.shape[0]):
            box3d_pts_3d_velo = corners3d[box_nr]  # (8,3)
            for corner in corner_for_box_list:
                p = np.array(box3d_pts_3d_velo[corner, 0:4])
                transformed_p = transform_point(p, np.linalg.inv(self.Tr_velo_kitti_cam))
                p = Point()
                p.x = transformed_p[0]
                p.y = transformed_p[1]
                p.z = transformed_p[2]
                marker.points.append(p)
        marker_array.markers.append(marker)

        id = 0
        for m in marker_array.markers:
            m.id = id
            id += 1
        self.mk_pub.publish(marker_array)
        marker_array.markers = []
        pass

    def extract_networks_input_from_pc2rosmsg(self, data):
        random_select = True

        pts_lidar = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        pts_lidar = transform_pointcloud(pts_lidar, self.Tr_velo_kitti_cam)
        pts_lidar = cropped_roi(pts_lidar, self.pc_roi)
        pts_lidar = pts_lidar[:, 0:3]  # only xyz

        if cfg.RPN.NUM_POINTS < len(pts_lidar):
            pts_depth = pts_lidar[:, self.down_sample['axis']]
            pts_near_flag = pts_depth < self.down_sample['depth']
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, cfg.RPN.NUM_POINTS - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)

        pts_input = pts_lidar[choice, :]
        return pts_input


def main():
    rospy.init_node('BoundingBoxNode', anonymous=True)
    ROSPointRCNN()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")


if __name__ == "__main__":
    main()
    pass
