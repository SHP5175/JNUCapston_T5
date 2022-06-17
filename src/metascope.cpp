#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

#define MAX_FRAME 258
#define MIN_NUM_FEAT 2000
#define FOCAL_X 2746
#define FOCAL_Y 2745
#define PP_X 1541
#define PP_Y 2050


class CameraInfo {


public:
    Point2d focal;
    Point2d ppoint;
    Mat cameraMatrix;
    Mat distortCoeff;

    CameraInfo (double focal_x, double focal_y, double pp_x, double pp_y) {
        focal.x = focal_x;
        focal.y = focal_y;
        ppoint.x = pp_x;
        ppoint.y = pp_y;
        cameraMatrix = (Mat_<double>(3,3) << focal.x, 0, ppoint.x, 0, focal.y, ppoint.y, 0, 0, 1);
    }
    CameraInfo (double focal, double pp) {
        CameraInfo (focal, focal, pp, pp);
    }
    CameraInfo () {
        CameraInfo (1.0, 1.0, 0, 0);
    }
};

//const static CameraInfo galaxy_s22_p(2300, 2300, 2040, 1530);
//const static CameraInfo camera_default(1, 1, 0, 0);


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status) {
    vector<float> err;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);    
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    int indexCorrection = 0;

    for(int i=0; i<status.size(); i++) { 
        Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)) {
     		if((pt.x<0)||(pt.y<0)) {
     		  	status.at(i) = 0;
     		}
     		points1.erase (points1.begin() + (i - indexCorrection));
     		points2.erase (points2.begin() + (i - indexCorrection));
     		indexCorrection++;
     	}
     }
}

void featureDetection(Mat img_1, vector<Point2f>& points1) {
    vector<KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}



int main() {
    CameraInfo camera1(FOCAL_X, FOCAL_Y, PP_X, PP_Y);
    CameraInfo camera2(952.94, 1270.59, 554.76, 983.96);
    // CameraInfo camera3(476.47, 635.29, 277.38, 491.98);
    CameraInfo camera3(839.8670995, 830.521428, 260.988979, 488.094087);
    CameraInfo camera4(FOCAL_X / 5, FOCAL_Y / 5, PP_X / 5, PP_Y / 5);
    //camera22, camera33 : camera2, camera3의 cameraMatrix 전치행렬
    CameraInfo camera22(952.94, 1270.59, 554.76, 983.96);
    CameraInfo camera33(476.47, 635.29, 277.38, 491.98);
    // CameraInfo camera33(476.47, 635.29, 277.38, 491.98);

    /*
    fx 1679.734199
    fy 1661.042856
    cx 521.977958
    cy 976.188174
    */
    camera22.cameraMatrix = (Mat_<double>(3,3) << camera22.focal.x, 0, 0, 0, camera22.focal.y, 0, camera22.ppoint.x, camera22.ppoint.y, 1);
    camera33.cameraMatrix = (Mat_<double>(3,3) << camera33.focal.x, 0, 0, 0, camera33.focal.y, 0, camera33.ppoint.x, camera33.ppoint.y, 1);

    camera1.distortCoeff = (Mat_<double>(5, 1) << 0.087182604, -0.117068306, 0.04453369, 0, 0);
    camera2.distortCoeff = (Mat_<double>(5, 1) << 0.087182604, -0.117068306, 0.04453369, 0, 0);
    camera3.distortCoeff = (Mat_<double>(5, 1) << 0.087182604, -0.117068306, 0.04453369, 0, 0);
    char filename[100];
    char text[100];
    int count = 1;
    Mat prevImage_c, currImage_c, inputImage, currImage_p, undistortedImage, inputImage2;
    Mat prevImage_r, currImage_r;
    Mat prevImage, currImage;
    Mat R_f, t_f;
    Mat F, E, R, t, mask;
    vector<Point2f> prevFeatures, currFeatures;
    vector<uchar> status;
    int fontFace = FONT_HERSHEY_PLAIN;
    float fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);
    namedWindow("Road facing camera", WINDOW_NORMAL );
    resizeWindow("Road facing camera", 600, 800);
    namedWindow("Trajectory", WINDOW_NORMAL );
    resizeWindow("Trajectory", 1000, 1000);

    Mat traj = Mat::zeros(1000, 1000, CV_8UC3);

    //실험
    int num_Frame=0;
    int last_feature_Detect_Frame=0;
    int recal_frame_size=5;
    
    VideoCapture cap;
    cap.open("/Users/reofard/Desktop/project/capstone/data_src/testvideo.mp4");
    if (!cap.isOpened()) {
        printf("video not opened -- error made");
        return -1;
    }

    while(1) {
        if(count == 2209)
        {
            printf("동영상 끝\n");
            continue;
        }
        if(count > 2209)
        {
            continue;
        }
        printf("Frame = %d \n", count++);
        cap >> inputImage;
        resize(inputImage, inputImage2, Size(540, 960), 0, 0, INTER_AREA);

        //undistort 주석처리, 해제
        //undistort(inputImage2, undistortedImage, camera3.cameraMatrix, camera3.distortCoeff);
        undistortedImage = inputImage2;
        //undistortedImage = inputImage;

        //resize(undistortedImage, currImage_c, Size(540, 960), 0, 0, INTER_AREA);
        //resize(undistortedImage, currImage_c, Size(600, 800), 0, 0, INTER_AREA);
        //cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        cvtColor(undistortedImage, currImage, COLOR_BGR2GRAY);
        cout << "size : " << currImage.size() << endl;
        //cout << numFrame << "   currImage.isContinuous() = " << currImage.isContinuous() << endl;
        //올바르지 않은 데이터면 프래임이면 스킵
        if(!currImage.isContinuous()) {
            cout << "error : currImage is not correct image" << endl;
            return -1;
        }
        //cout << numFrame << "  prevFeatures size : " << prevFeatures.size() << endl;
        
        //첫번째 프레임이면 스킵
        if(prevFeatures.size()==0)
        {
            last_feature_Detect_Frame = count;
            featureDetection(currImage, currFeatures);

            prevImage = currImage.clone();
            prevFeatures = currFeatures;
            continue;
        }

        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        //E : essentialMatrix || F : fundamentalMatrix

        E = findEssentialMat(currFeatures, prevFeatures, camera3.cameraMatrix, RANSAC, 0.999, 1.0, mask);
        // F = findFundamentalMat(currFeatures, prevFeatures, FM_RANSAC);
        // E = camera33.cameraMatrix * F * camera3.cameraMatrix;

        //recoverPose 두개 차이 : fundamentalMat 사용 시 mask 매개변수 제외

        //if(E.isContinuous()) recoverPose(E, currFeatures, prevFeatures, camera3.cameraMatrix, R, t, mask);
        if(E.isContinuous()) recoverPose(E, currFeatures, prevFeatures, camera3.cameraMatrix, R, t);
        
        if(t_f.cols==0){
            t_f = t.clone(); 
            R_f = R.clone();
        }
        else {
            t_f = t_f + (R_f*t);
            R_f = R*R_f;
        }

        // 특징점이 너무 적을때 || 특징점 탐색수 일정 프레임이 경과할 경우
        if(prevFeatures.size() < MIN_NUM_FEAT || count - last_feature_Detect_Frame > recal_frame_size)
        {
            last_feature_Detect_Frame = count;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
        }

        //이전 이미지 복사
        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        cout << endl;

        
        int x = int(t_f.at<double>(0)) + 400;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(x, y) , 1, CV_RGB(255,0,0), 1);
        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), cv::FILLED);
        // sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        sprintf(text, "feature count: x = %d", currFeatures.size());
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
        //imshow("Road facing camera", currImage_c);
        //imshow("Road facing camera", inputImage);
        imshow("Road facing camera", undistortedImage);
        imshow("Trajectory", traj);
        waitKey(10);
        //sleep(100);
    }
}
