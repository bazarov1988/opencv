#include <iostream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <cmath>
#include <vector>
#include <typeinfo>

using namespace std;
using namespace cv;

double get_speed(cv::Mat& frame, vector<Point>& contour,Point2f center,float radius,vector<Point2f> prevContourCenters,double fps)
{
    int meters = 30;
    int maxSpeed = 100;
    if(center.x>5&&center.x<250){
        for(size_t i = 0; i < prevContourCenters.size(); i++ )
        {
            if(prevContourCenters[i].x<center.x){
                double diff = center.x-prevContourCenters[i].x;
                cv::Point x,y;
                int fontFace = 3;
                double fontScale = 0.5;
                int thickness = 1;
                Point textOrg(center.x-20,center.y-radius-5);
                String str2 = " km/h";
                double speed = diff*fps/300*meters*3.6;
                std::stringstream strSpeed;
                strSpeed << speed;
                if(speed>maxSpeed){
                    putText(frame, strSpeed.str()+str2, textOrg, fontFace, fontScale, Scalar(10,10,10), thickness,5);
                    imwrite("/home/igrosso/www/inna_project_new/untitled/violators/image.jpg",frame);
                } else {
                    putText(frame, strSpeed.str()+str2, textOrg, fontFace, fontScale, Scalar(255,255,255), thickness,5);
                }
                return speed;
            }
        }

    }
};

int main()
{
    int count;
    double area, ar;
    Mat frame, fore, img, prevImg, temp, gray, vehicle_ROI, img_temp;
    VideoCapture cap("/home/igrosso/www/inna_project_new/untitled/data/video2.avi");
    BackgroundSubtractorMOG2 bg(500, 25, false);
    vector<vector<Point> > contours;
    vector<Point2f> prevContoursCenters;
    vector<Point2f> tempContoursCenters;
    vector<Rect> cars;
    namedWindow("Frame");

    cap >> img_temp;
    cvtColor(img_temp, gray, CV_BGR2GRAY);
    gray.convertTo(temp, CV_8U);
    bilateralFilter(temp, prevImg, 5, 20, 20);

    // start and end times
    time_t start, end;
    // fps calculated using number of frames / seconds
    double fps;
    // frame counter
    int counter = 0;
    // floating point seconds elapsed since start
    double sec;
    // start the clock
    time(&start);


    while(true)
    {
        count=0;
        cap >> frame;
        cvtColor(frame, gray, CV_BGR2GRAY);
        gray.convertTo(temp, CV_8U);
        bilateralFilter(temp, img, 5, 20, 20);
        bg.operator()(img,fore);
        erode(fore,fore,Mat());
        dilate(fore,fore,Mat());
        findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());
        vector<Point2f> center( contours.size() );
        vector<float> radius( contours.size());
        tempContoursCenters.clear();
        time(&end);
        // calculate current FPS
        ++counter;
        sec = difftime (end, start);
        fps = counter / sec;
        for(size_t i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            vehicle_ROI = img(boundRect[i]);
            area = contourArea(contours[i], false);
            ar = vehicle_ROI.cols/vehicle_ROI.rows;
            if(area > 450.0 && ar > 0.8)
            {
                minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
                rectangle( frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 1, 8, 0 );
                tempContoursCenters.push_back(center[i]);
                get_speed(frame,contours[i],center[i],radius[i],prevContoursCenters,fps);
                count=count+1;
            }
        }
        stringstream ss;
        ss << count;
        string s = ss.str();
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 1;
        int thickness = 3;
        cv::Point textOrg(10, 110);
        cv::putText(frame, s, textOrg, fontFace, fontScale, Scalar(255,255,255), thickness,5);
        int win_size = 10;
        int maxCorners = 200;
        double qualityLevel = 0.5;
        double minDistance = 1;
        int blockSize = 3;
        double k = 0.04;
        vector<Point2f> img_corners;
        img_corners.reserve(maxCorners);
        vector<Point2f> prevImg_corners;
        prevImg_corners.reserve(maxCorners);
        goodFeaturesToTrack(img, img_corners, maxCorners,qualityLevel,minDistance,Mat(),blockSize,true);
        cornerSubPix( img, img_corners, Size( win_size, win_size ), Size( -1, -1 ),
                     TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
        vector<uchar> features_found;
        features_found.reserve(maxCorners);
        vector<float> feature_errors;
        feature_errors.reserve(maxCorners);
        calcOpticalFlowPyrLK( img, prevImg, img_corners, prevImg_corners, features_found, feature_errors ,
                             Size( win_size, win_size ), 3,
                             cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0, k);
//        for( int i=0; i < features_found.size(); i++ ){
//            Point2f p0( ceil( img_corners[i].x ), ceil( img_corners[i].y ) );
//            Point2f p1( ceil( prevImg_corners[i].x ), ceil( prevImg_corners[i].y ) );
//            line( frame, p0, p1, CV_RGB(255,0,0), 5 );
//        }
        prevImg = img;
        prevContoursCenters.clear();
        prevContoursCenters = tempContoursCenters;
        imshow("Frame",frame);
        if(waitKey(5) >= 0)
            break;
    }

}


