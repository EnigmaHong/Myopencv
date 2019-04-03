#pragma once
#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <stack>

using namespace std;
using namespace cv;

typedef struct _Feather
{
	int label;              // 连通域的label值
	int area;               // 连通域的面积
	cv::Rect boundingbox;       // 连通域的外接矩形框
} Feather;

void icvprCcaBySeedFill(const cv::Mat& _binImg, cv::Mat& _lableImg);

void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
cv::Scalar icvprGetRandomColor();
void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);

int BwLabel(cv::Mat &src, cv:: Mat &dst, std::vector<Feather> & featherList);
