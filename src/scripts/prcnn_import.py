import argparse
import os
import sys
import rospkg.rospack as rospack

pack_name = 'ros_pointrcnn'
# argparser
argparse = argparse.ArgumentParser(description='ros_pointnet')
argparse.add_argument('-s', '--src', dest='pointrcnn_dir', required=True, help='where pointrcnn src code dir')
argparse.add_argument('-m', '--model', dest='model', required=True, help='model path')
argparse.add_argument('-c', '--cfg', help='modle cfg file path', default=None)
argparse.add_argument('-t', '--topic', help='topic to receive lidar points', default='/points')
argparse.add_argument('-v', '--viz', help='topic to receive lidar points', type=bool, default=True)
argparse.add_argument('-tf', '--tf', help='tf to kitti camera frame to use pretrain model', type=bool, default=True)
args = argparse.parse_args()
print('args: \n\t', args)

# copy to var
pointrcnn_dir = args.pointrcnn_dir
pointrcnn_weight = args.model
cfg_file = args.cfg if args.cfg is not None else os.path.join(pointrcnn_dir, 'tools/cfgs/default.yaml')
pc_topic = args.topic
is_viz = args.viz
is_tf = args.tf

# assert validation
assert os.path.exists(pointrcnn_dir), "pointrcnn path not exists"
assert os.path.exists(pointrcnn_weight), "pointrcnn_model path not exists"
assert os.path.exists(cfg_file), "cfg file not exists"


def import_pointrcnn():
    while 1:
        try:
            package_dir = rospack.RosPack().get_path(pack_name)
        except rospack.ResourceNotFound:
            package_dir = None
        print('pkg dir: \n\t', package_dir)
        assert package_dir is not None, 'no package named %s is found' % pack_name
        break
    while 1:
        __local_pointrcnn_dir = os.path.join(package_dir, 'src', os.path.basename(pointrcnn_dir))
        if os.path.exists(__local_pointrcnn_dir):
            cmd = __local_pointrcnn_dir
        else:
            cmd = 'ln -s {} {}'.format(pointrcnn_dir, __local_pointrcnn_dir)
            os.system(cmd)
        print('local pointrcnn: \n\t%s' % cmd)
        break
    while 1:
        sys.path.append(__local_pointrcnn_dir)
        sys.path.append(os.path.join(__local_pointrcnn_dir, 'lib/net'))
        sys.path.append(os.path.join(__local_pointrcnn_dir, 'lib/net/datasets'))
        break


import_pointrcnn()
