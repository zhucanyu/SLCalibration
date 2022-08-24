#include<opencv2/opencv.hpp>
#include <fstream> 
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

//Find the centers of the calibration plate circle

int main()
{

	Mat img, img8 ,img8U3;
	Mat s1;
	Size board_size;
	vector<Point2f> centers;
	bool found;
	board_size.width =5;
	board_size.height = 3;
	
	for (int i = 0; i <5 ; i++)  //Number of calibrated pictures
	{
	
	img = imread("E:/opencv_test/" + to_string(i + 1) + ".png", IMREAD_ANYDEPTH + IMREAD_GRAYSCALE);  //Change it to "Your File Directory/"...

	namedWindow(to_string(i + 1), 0);
	resizeWindow(to_string(i + 1), img.cols / 4, img.rows / 4);

	resize(img, img8, Size(img.cols / 1, img.rows / 1), 0, 0, INTER_LINEAR);

	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 200;
	params.blobColor = 255;
	params.maxArea = 5000;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	found =findCirclesGrid(img8, board_size,centers,CALIB_CB_ASYMMETRIC_GRID, detector);//找到中心点 findCirclesGrid of Asymmetric dot calibration plate
	//found = findCirclesGrid(img8, board_size, centers, 1, detector);//找到中心点					findCirclesGrid of Symmetric dot calibration plate

	cout<<i+1 << "号图片搜索结果："<<found << endl;
	cout << "找到中心点数："<<centers.size() << endl;

	cvtColor(img8, img8U3, COLOR_GRAY2RGB, 3);

	drawChessboardCorners(img8U3, board_size, Mat(centers), found);
	
	imwrite("E:/opencv_test/centers_found/" + to_string(i + 1) + "centers.png", img8U3);  //Change it to "Your File Directory/centers_found/"...
	imshow(to_string(i + 1), img8U3);
	//保存中心点数据
	ofstream of;
	of.open("E:/opencv_test/centers_found/centers"+to_string(i+1)+".csv", ios::out); //Change it to "Your File Directory/centers_found/centers"...
	for (int j = 0; j < centers.size(); j++)
	{
		of<< centers[j].x*1<<","<<centers[j].y*1 <<endl;
	}
	of.close();
	}

	waitKey(0);
	destroyAllWindows();
	return 0;

}