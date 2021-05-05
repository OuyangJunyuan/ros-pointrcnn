# 使用

下载PointRCNN和这个包

使用`rosrun ros_pointrcnn ros_pointrcnn.py  -h` 查看参数意义

`rosrun ros_pointrcnn ros_pointrcnn.py  
-s /home/ou/workspace/code/PointRCNN
-m /home/ou/workspace/code/PointRCNN/tools/PointRCNN.pth
-c /home/ou/workspace/ros_ws/pointrcnn_ws/src/ros-pointrcnn/src/config/pointrcnn.yaml
-t /excavator/lidar_perception/viz_cloud_0`

检测结果如下

![image-20210505193101204](https://i.loli.net/2021/05/05/hyYOvCeJAWx7bBF.png)