import argparse
import os
import sys
import rospkg.rospack as rospack

argparse = argparse.ArgumentParser(description='ros_pointnet')
argparse.add_argument('-s', '--src', dest='pointrcnn_dir', required=True, help='where pointrcnn src code dir')
argparse.add_argument('-m', '--model', dest='model', required=True, help='model path')
argparse.add_argument('-c', '--cfg', help='modle cfg file path', default=None)
args = argparse.parse_args()

pointrcnn_dir = args.pointrcnn_dir
pointrcnn_model = args.model
cfg_file = args.cfg if args.cfg is not None else os.path.join(pointrcnn_dir, 'tools/cfgs/default.yaml')

assert os.path.exists(pointrcnn_dir), "pointrcnn path not exists"
assert os.path.exists(pointrcnn_model), "pointrcnn_model path not exists"
assert os.path.exists(cfg_file), "cfg file not exists"

package_dir = rospack.RosPack().get_path('ros_pointrcnn')

print('find pointrcnn_dir: %s' % pointrcnn_dir)
print('find pointrcnn_model: %s' % pointrcnn_model)
print('find package_dir: %s' % package_dir)
print('find cfg_file: %s' % cfg_file)

__local_pointrcnn_dir = os.path.join(package_dir, 'src', os.path.basename(pointrcnn_dir))
if os.path.exists(__local_pointrcnn_dir):
    print('find local_pointrcnn_dir: %s' % __local_pointrcnn_dir)
else:
    cmd = 'ln -s {} {}'.format(pointrcnn_dir, __local_pointrcnn_dir)
    print(cmd)
    os.system(cmd)

sys.path.append(__local_pointrcnn_dir)
sys.path.append(os.path.join(__local_pointrcnn_dir, 'lib/net'))
sys.path.append(os.path.join(__local_pointrcnn_dir, 'lib/net/datasets'))
