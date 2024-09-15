# 202302暑期学习
> yolo部分学习效果较差，最终数字识别比赛成品见***number_detect***

> 其余yolo部分记载训练程序及本人学习过程

> ros2图像处理及话题转化部分详见ros2

本地部署
git clone -b master git@github.com:Whaltze/Robot.git

## number_detect本地部署

cd 202302/number_detect/

pip install -r requirements.txt 

python number_detect.py

## ros2本地部署

cd 202302/ros2

分别打开三个终端，依次输入

> ros2 launch usb_cam camera.launch.py

> ros2 run camera camera_node #这里如果找不到包可以先source install/setup.bash即可

> ros2 run rqt_image_view rqt_image_view

选择rec_img右上角便可以看看白色矩形框
