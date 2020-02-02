
#include "opencv2/opencv.hpp"
#include<iostream>
#include<vector>
#include<fstream>
#include<string>

int thresh;
int lower_canny;
cv::Mat frame;
cv::Mat edgesFrame;
cv::Mat myFrame;
std::vector<cv::Point> HandContour;
std::vector<std::vector<cv::Point>> contours;

int maxRGB(int _R, int _G, int _B)
{
    return(std::max(_R,std::max(_G,_B)));
}
int minRGB(int _R, int _G, int _B)
{
    return(std::min(_R,std::min(_G,_B)));
}

void onTrack(int val, void*)
{
    thresh = val;
}
void onTrack2(int val, void*)
{
    lower_canny = val;
}

int main(int argc, char** argv)
{
    thresh = 25;
    std::ofstream myoutfile;
   // myoutfile.open (filename);
 // myfile << "Writing this to a file.\n";
 // myfile.close();
    std::string filename = argv[1];
    if(argc > 2)
    {
        thresh = atoi(argv[2]);
    }
    cv::VideoCapture cap(filename);
    size_t lastindex = filename.find_last_of("."); 
    std::string rawname = filename.substr(0, lastindex);
    myoutfile.open (rawname+".txt");
    bool bSuccess = cap.read(frame);
    myFrame = cv::Mat(cv::Size(frame.cols,frame.rows),CV_8UC1,cv::Scalar(0));
    if( !cap.isOpened())
    {
        std::cout << "Can not open webcam" << std::endl;
        return -1;
    }
    cv::namedWindow("video",CV_WINDOW_AUTOSIZE);
    cv::namedWindow("color",CV_WINDOW_AUTOSIZE);
    cv::namedWindow("canny",CV_WINDOW_AUTOSIZE);
    cv::createTrackbar("my_bar","color",&thresh,255,onTrack);
    cv::createTrackbar("my_bar_2","canny",&lower_canny,255,onTrack2);
    while(1)
    {
        
        bool bSuccess = cap.read(frame);
        //cv::cvtColor(frame,HSVFrame,cv::COLOR_BGR2HSV);
        if(!bSuccess)
        {
            std::cout << "Can not read frame from webcam" << std::endl;
            break;
        }
        int B,G,R;
        
        for(int i = 0; i < frame.rows ; i++)
        {
            for (int j = 0 ; j < frame.cols ; j++)
            {
                B = (int)frame.at<cv::Vec3b>(i,j)[0];
                G = (int)frame.at<cv::Vec3b>(i,j)[1];
                R = (int)frame.at<cv::Vec3b>(i,j)[2];
                if(R == maxRGB(R,B,G) && (R - minRGB(R,G,B) > thresh))
                {
                    myFrame.at<uchar>(i,j) = 255;
                }
                else
                {
                    myFrame.at<uchar>(i,j) = 0;
                }
            }
        }
        cv::medianBlur(myFrame,myFrame,7);
        cv::medianBlur(myFrame,myFrame,7);
        cv::imshow("color",myFrame);
        cv::Canny(myFrame,edgesFrame,lower_canny,lower_canny*3);
        cv::findContours(edgesFrame,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        size_t max_length = 0;
        for(auto c : contours)
        {
            if(max_length <= c.size())
            {
                HandContour = c;
                max_length = c.size();
            }
        }
        for(auto p : HandContour)
        {
            frame.at<cv::Vec3b>(p.y,p.x)[0] = 255;
            frame.at<cv::Vec3b>(p.y,p.x)[1] = 255;
            frame.at<cv::Vec3b>(p.y,p.x)[2] = 0;
            myoutfile << p.x << " " << p.y << std::endl;

        }
        myoutfile << -1 << " " << -1 << std::endl;
        cv::imshow("canny",edgesFrame);
        cv::imshow("video",frame);
        if(cv::waitKey(30) == 27)
        {
            std::cout << "Exiting" << std::endl;
            myoutfile.close();
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}