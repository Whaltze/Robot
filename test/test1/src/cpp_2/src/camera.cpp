#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;

int camera_width = 640;
int camera_height = 480;

int main(int argc, char const *argv[])
{
    // 初始化变量和对象
    cv::VideoCapture cap(1);
    cap.set(CAP_PROP_FRAME_WIDTH, camera_width);
    cap.set(CAP_PROP_FRAME_HEIGHT, camera_height);
    // 循环处理每一帧图像
    while (true) {
        cv::Mat color_image;
        cap.read(color_image);
        if (color_image.empty()) {
            cerr << "Failed to capture image" << endl;
            break;
        }
     imshow("Color Image", color_image);
     char key = waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    // 释放资源
    cap.release();
    destroyAllWindows();
    return 0;
}