#pragma once
//此头文件主要是尝试重写opencv各个函数
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <windows.h>
#include <iostream>
#include <vector>
#include<opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <ConectDomain.h>
#include <iomanip>

using namespace std;
using namespace cv;

int _image_reduction_(Mat src) {//通过保留奇数行列的方式缩小图像的一半
	if (!src.data) {
		cout << "could not load image..." << endl;;
		return -1;
	}
	int M = src.rows;
	int N = src.cols;
	Mat dst(M / 2, N / 2, CV_8UC3, Scalar(0, 0, 255));
	for (int i = 1, k = 0; i < M; i += 2, k++) {
		uchar *data_src = src.ptr<uchar>(i);
		uchar *data_dst = dst.ptr<uchar>(k);
		for (int j = 3, v = 0; j < N * 3; j += 6, v += 3) {
			data_dst[v] = data_src[j];
			data_dst[v + 1] = data_src[j + 1];
			data_dst[v + 2] = data_src[j + 2];
		}
	}
	namedWindow("原图");
	imshow("原图", src);
	//namedWindow("输出");
	imshow("输出", dst);
	waitKey(0);
	return 0;
}

int _image_magnification_(Mat src) {//通过复制奇数行列的方式放大图像一半
	if (src.empty()) {
		cout << "could not load image..." << endl;;
		return -1;
	}
	int M = src.rows;
	int N = src.cols;
	Mat dst(M+M / 2, N+N / 2, CV_8UC3, Scalar(0, 0, 255));
	for (int i = 0, k = 0; i < M; i++ , k++) {
		uchar *data_src = src.ptr<uchar>(i);
		uchar *data_dst = dst.ptr<uchar>(k);
		for (int j = 0, v = 0; j < N * 3; j += 3, v += 3) {
			data_dst[v] = data_src[j];
			data_dst[v + 1] = data_src[j + 1];
			data_dst[v + 2] = data_src[j + 2];
			if (j % 2 == 1) {
				data_dst[v+3] = data_src[j];
				data_dst[v + 4] = data_src[j + 1];
				data_dst[v + 5] = data_src[j + 2];
				v = v + 3;
			}
		}
		if (i % 2 == 1) {
			uchar *data_dst = dst.ptr<uchar>(k + 1);
			k++;
			for (int j = 0, v = 0; j < N * 3; j += 3, v += 3) {
				data_dst[v] = data_src[j];
				data_dst[v + 1] = data_src[j + 1];
				data_dst[v + 2] = data_src[j + 2];
				if (j % 2 == 1) {
					data_dst[v + 3] = data_src[j];
					data_dst[v + 4] = data_src[j + 1];
					data_dst[v + 5] = data_src[j + 2];
					v = v + 3;
				}
			}
		}
	}
	namedWindow("原图");
	imshow("原图", src);
	//namedWindow("输出");
	imshow("输出", dst);
	waitKey(0);
	return 0;
}

int _image_inversion_(Mat src) {//图像反转变换s=L-1-r
	if (src.empty()) {
		cout << "could not load image..." << endl;;
		return -1;
	}
	cvtColor(src,src,CV_RGB2GRAY);
	Mat dst(src.size(),src.type());
	for (int i = 0; i < src.rows; i++) {
		uchar *data_src = src.ptr<uchar>(i);
		uchar *data_dst = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) {
			data_dst[j] = 255 - 1 - data_src[j];
		}
	}
	namedWindow("原图");
	imshow("原图", src);
	namedWindow("输出");
	imshow("输出", dst);
	waitKey(0);
	return 0;
}

int _image_log_enhance_(Mat src,int const &c) {//图像对数变换
	if (src.empty()) {
		cout << "Can not load image ..." << endl;
		return -1;
	}
	Mat dst(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3f>(i, j)[0] = c*log(1+src.at<Vec3b>(i,j)[0]);
			dst.at<Vec3f>(i, j)[1] = c*log(1 + src.at<Vec3b>(i, j)[1]);
			dst.at<Vec3f>(i, j)[2] = c*log(1 + src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化0-25
	normalize(dst, dst, 0, 255, CV_MINMAX);
	//转换成8bit图像显示
	convertScaleAbs(dst, dst);
	namedWindow("原图");
	imshow("原图", src);
	namedWindow("输出");
	imshow("输出", dst);
	waitKey(0);
	return 0;
}

int _image_Gamma_enhance_(Mat src) {//伽马变换增强图像对比度 伽马函数为三次方
	if (src.empty()) {
		cout << "Can not load image ..." << endl;
		return -1;
	}
	Mat dst(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3f>(i, j)[0] = src.at<Vec3b>(i, j)[0] * src.at<Vec3b>(i, j)[0] * src.at<Vec3b>(i, j)[0];
			dst.at<Vec3f>(i, j)[1] = src.at<Vec3b>(i, j)[1] * src.at<Vec3b>(i, j)[1] * src.at<Vec3b>(i, j)[1];
			dst.at<Vec3f>(i, j)[2] = src.at<Vec3b>(i, j)[2] * src.at<Vec3b>(i, j)[2] * src.at<Vec3b>(i, j)[2];
		}
	}
	normalize(dst, dst, 0, 255, CV_MINMAX);
	convertScaleAbs(dst, dst);
	namedWindow("原图");
	imshow("原图", src);
	namedWindow("输出");
	imshow("输出", dst);
	waitKey(0);
	return 0;
}

int _image_enhance_(Mat src) {//图像增强实例，数字图像处理中的人体骨骼
	Mat gray, laplace, final;
	Mat laplaceAddsrc = Mat(src.size(), src.type());
	//GaussianBlur(src,src,Size(5,5),0,0);
	//cvtColor(src,gray,CV_RGB2GRAY);
	Mat kernel_laplace = (Mat_<char>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
	filter2D(src, laplace, CV_8U, kernel_laplace, Point(-1, -1), 0);
	imshow("final", laplace);

	add(src, laplace, laplaceAddsrc);
	imshow("laplaceAddsrc", laplaceAddsrc);

	Mat sobel, gx, gy;
	Sobel(src, gx, CV_8U, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
	Sobel(src, gy, CV_8U, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
	convertScaleAbs(gx, gx);//求绝对值
	convertScaleAbs(gy, gy);
	add(gx, gy, sobel);
	convertScaleAbs(sobel, sobel);
	imshow("sobel", sobel);//没有去模糊直接求梯度图
	//5*5的均值滤波，模糊原图
	Mat blured, blur_sobel, multy;
	blur(sobel, blured, Size(5, 5), Point(-1, -1));
	//blur(src, blured, Size(5, 5), Point(-1, -1));
	//Sobel(blured, gx, CV_8U, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT);
	//Sobel(blured, gy, CV_8U, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT);
	//convertScaleAbs(gx, gx);
	//convertScaleAbs(gy, gy);
	//add(gx, gy, blur_sobel);
	//imshow("e_sobelAfterBlur", blur_sobel);//先对图像去模糊再求梯度

	//与操作
	Mat f, g;
	bitwise_and(laplace, sobel, f);
	imshow("f", f);

	add(src, f, g);
	imshow("g", g);

	//效果图，自加一次，增强。冈萨雷斯里面是做幂律变换，我没做，只是做了个简单的增强，看着玩。
	//add(g, g, g);
	Mat dst(g.size(), CV_32FC3);
	for (int i = 0; i < g.rows; i++) {
		for (int j = 0; j < g.cols; j++) {
			dst.at<Vec3f>(i, j)[0] = g.at<Vec3b>(i, j)[0] * g.at<Vec3b>(i, j)[0] * g.at<Vec3b>(i, j)[0];
			dst.at<Vec3f>(i, j)[1] = g.at<Vec3b>(i, j)[1] * g.at<Vec3b>(i, j)[1] * g.at<Vec3b>(i, j)[1];
			dst.at<Vec3f>(i, j)[2] = g.at<Vec3b>(i, j)[2] * g.at<Vec3b>(i, j)[2] * g.at<Vec3b>(i, j)[2];
		}
	}
	normalize(dst, dst, 0, 255, CV_MINMAX);
	convertScaleAbs(dst, dst);
	add(src, dst, dst);
	imshow("h", dst);

	return 0;
}
