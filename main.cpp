#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <windows.h>
#include <iostream>

using namespace std;
using namespace cv;

string xmlPath = "haarcascade_frontalface_alt.xml";
//xmlpath 字符串记录那个.xml文件的路径
void detectAndDisplay(Mat image)
{
	CascadeClassifier ccf;      //创建脸部对象
	ccf.load(xmlPath);           //导入opencv自带检测的文件
	vector<Rect> faces;
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	ccf.detectMultiScale(gray, faces, 1.1, 3, 0, Size(50, 50), Size(500, 500));
	for (vector<Rect>::const_iterator iter = faces.begin(); iter != faces.end(); iter++)
	{
		rectangle(image, *iter, Scalar(0, 0, 255), 2, 8); //画出脸部矩形
	}
	Mat image1;

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//image1 = image(Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
	}

	imshow("1", image);
	//imshow("2", image1);
	cvWaitKey(0);

}
int _DetectFace_image_() {
	string path = "test.jpg";//以检测图片1.jpg为例
	Mat image = imread(path, -1);

	CascadeClassifier a;     //创建脸部对象
	if (!a.load(xmlPath))     //如果读取文件不出错，则检测人脸
	{
		cout << "无法加载xml文件" << endl;
		return 0;
	}
	detectAndDisplay(image);// 检测人脸
	return 0;
}

int _DetectFace_vido_() {
	CascadeClassifier faceCascade;
	double t = 0;
	int nRet = 0;
	/* 加载分类器 */
#ifdef VERSION_2_4	
	nRet = faceCascade.load("/haarcascade_frontalface_alt.xml");
#else
	nRet = faceCascade.load("haarcascade_frontalface_alt.xml");
#endif
	
	VideoCapture capture;
	capture.open(0);
	//  capture.open("video.avi");
	if (!capture.isOpened())
	{
		cout << "open camera failed. " << endl;
		return -1;
	}
	if (!nRet)
	{
		printf("load xml failed.\n");
		return -1;
	}

	Mat img, imgGray;
	vector<Rect> faces;
	while (1)
	{
		capture >> img;
		if (img.empty())
		{
			continue;
		}

		cvtColor(img, imgGray, CV_RGB2GRAY);

		/* 检测人脸 */
		t = (double)getTickCount();
		faceCascade.detectMultiScale(imgGray, faces,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		t = (double)getTickCount() - t;
		
		cout << "detection time =" << t * 1000 / getTickFrequency() << "ms" <<endl;

		/* 画矩形框出人脸 */
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
				Scalar(0, 255, 0), 1, 8);
		}

		imshow("CameraFace", img);

		if (waitKey(1) > 0)		// delay ms 等待按键退出
		{
			break;
		}
	}

	return 0;
}

void _Splitter_RGB_() {
	const char *strimagename = "007.jpg";
	IplImage *img = cvLoadImage(strimagename,CV_LOAD_IMAGE_UNCHANGED);	
	CvScalar avgChannels = cvAvg(img);
	//独立计算数组每个通道的平均值
	double avgB = avgChannels.val[0];
	double avgG = avgChannels.val[1];
	double avgR = avgChannels.val[2];
	cout << avgB << " " << avgR << " " << avgR << endl;

	//分割RGB颜色通道
	IplImage* rImg = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	IplImage* gImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	IplImage* bImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

	cvSetImageCOI(img,1);
	cvCopy(img, bImg);
	cvSetImageCOI(img,2);
	cvCopy(img,gImg);
	cvSetImageCOI(img,3);
	cvCopy(img,rImg);

	cvShowImage("原图",img);
	cvShowImage("rImg", rImg);
	cvShowImage("gImg", gImg);
	cvShowImage("bImg", bImg);

	//将单通道的图片转换成多通道图片
	IplImage* rrImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage* rgImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage* rbImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	cvMerge(0,0,rImg,0,rrImg);//rgby
	cvMerge(0,gImg,0,0,rgImg);
	cvMerge(bImg,0,0,0,rbImg);
	cvShowImage("rrImg", rrImg);
	cvShowImage("rgImg", rgImg);
	cvShowImage("rbImg", rbImg);
	

	waitKey();
	
}

int _Hist_equalization_() {//彩色图像直方图均衡化实现
	Mat image = imread("007.jpg");
	if (image.empty()) {
		cout << "打开图片失败，请检查!" << endl;
		return -1;
	}
	imshow("原图",image);
	Mat imageRGB[3];
	split(image, imageRGB);
	/*imshow("1",imageRGB[0]);
	imshow("2",imageRGB[1]);
	imshow("3",imageRGB[2]);*/
	for (int i= 0; i < 3; i++) {//计算直方图均衡化
		equalizeHist(imageRGB[i],imageRGB[i]); 
	}
	/*imshow("11",imageRGB[0]);
	imshow("22",imageRGB[1]);
	imshow("33",imageRGB[2]);*/
	merge(imageRGB, 3, image);//将单通道的图片转换成多通道图片
	imshow("直方图均衡化图像增强效果",image);
	waitKey();
	return 0;
	
}

int _Laplace_ope_() {//拉普拉斯算子增强局部图像的对比度
	Mat image = imread("007.jpg");
	if (image.empty()) {
		cout << "打开图像失败，请重新检查图片！" << endl;
		return -1;
	}
	imshow("原图",image);
	Mat imageEnhance;
	Mat kernel = (Mat_<float>(3,3)<<0,-1,0,0,10,0,0,-1,0);
	filter2D(image,imageEnhance,CV_8UC3,kernel);
	imshow("拉普拉斯图像增强",imageEnhance);

	waitKey();
	return 0;
}


int _Log_Enhance_() {//对数Log变换的图像增强
	Mat image = imread("007.jpg");
	if (image.empty()) {
		cout << "打开图像失败，请重新检查！" << endl;
		return -1;
	}
	imshow("原图",image);
	Mat imageLog(image.size(),CV_32FC3);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			imageLog.at<Vec3f>(i,j)[0] = log(1+image.at<Vec3b>(i,j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1+image.at<Vec3b>(i,j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1+image.at<Vec3b>(i,j)[2]);
		}
	}
	//归一化0-25
	normalize(imageLog,imageLog,0,255,CV_MINMAX);
	//转换成8bit图像显示
	convertScaleAbs(imageLog,imageLog);
	imshow("Log增强后图像",imageLog);
	waitKey();
	return 0;

}


int _Gamma_Enhance_() {//伽马变换增强图像对比度
	Mat image = imread("007.jpg");
	if (image.empty()) {
		cout << "打开图像失败，请重新检查！" << endl;
		return -1;
	}
	imshow("原图",image);
	Mat imageGama(image.size(),CV_32FC3);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			imageGama.at<Vec3f>(i, j)[0] = image.at<Vec3b>(i, j)[0] * image.at<Vec3b>(i, j)[0] * image.at<Vec3b>(i, j)[0];
			imageGama.at<Vec3f>(i, j)[1] = image.at<Vec3b>(i, j)[1] * image.at<Vec3b>(i, j)[1] * image.at<Vec3b>(i, j)[1];
			imageGama.at<Vec3f>(i, j)[2] = image.at<Vec3b>(i, j)[2] * image.at<Vec3b>(i, j)[2] * image.at<Vec3b>(i, j)[2];
		}
	}
	normalize(imageGama,imageGama,0,255,CV_MINMAX);
	convertScaleAbs(imageGama,imageGama);
	imshow("伽马变换图像增强效果",imageGama);
	waitKey();
	return 0;
}


int main()
{
	_Log_Enhance_();
	
	//cin.get();



	
}
