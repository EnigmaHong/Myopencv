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

#define m_PI 3.14159265358979323846;

using namespace std;
using namespace cv;
Mat gaussian_kernal(int kernelRowSize, int kernelColSize, double sigma0);
Mat HSV2RGB(Mat src);

string xmlPath = "haarcascade_frontalface_alt.xml";
//xmlpath 字符串记录那个.xml文件的路径
Mat src, gray_src, drawImg;
int threshold_v = 170;
int threshold_max = 255;
const char* output_win = "rectangle-demo";
RNG rng(12345);
template <typename T>
void print_Mat(Mat img) {//输出图像每个像素值
	for (int row = 0; row < img.rows; row++) {
		T * pdata = img.ptr<T>(row);
		for (int col = 0; col < img.cols; col++) {
			T data = pdata[col];
			cout << data << " ";
		}
		cout << endl;
	}
}
template<typename T>
T SumElem(Mat &img) {//计算图像像素值之和
	T sum = 0;
	T *pdata = NULL;
	for (int row = 0; row < img.rows; row++) {
		pdata = img.ptr<T>(row);
		for (int col = 0; col < img.cols; col++) {
			sum += pdata[col];
		}
	}
	return sum;
}
template<typename T>
T meanElem(Mat img) {
	return T(SumElem<T>(img) / img.total());
}

int Sum_mat(Mat &img) {//计算图像像素值之和
	int sum = 0;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			sum += (int)img.at<uchar>(row, col);
		}
	}
	return sum;
}
int Count_Mat(Mat img) {
	int count = 0;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			if ((int)img.at<uchar>(row, col) == 0) {
				count++;
			}
		}
	}
	return count;
}
int _caohe_() {
	src = imread("exmaple.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3), Point(-1, -1));

	Mat binary_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	threshold(gray_src, binary_output, threshold_v, threshold_max, THRESH_BINARY);
	int count = Count_Mat(binary_output);
	findContours(binary_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	int area = contourArea(contours[1]);
	cout << "The object area is:" << area << endl;

	Mat imageContours = src;
	RotatedRect rect;

	for (size_t i = 0; i < contours.size(); i++) {
		rect = minAreaRect(contours[i]);
		Point2f P[4];
		rect.points(P);
		for (int j = 0; j <= 3; j++) {
			line(imageContours, P[j], P[(j + 1) % 4], Scalar(255), 2);
		}
		// cv::imwrite("imageContours.jpg", imageContours);
	}

	Point2f center = rect.center;//外接矩形中心点坐标
	Mat rot_mat = getRotationMatrix2D(center, rect.angle, 1.0);//求旋转矩阵
	Mat rot_image;
	Size dst_sz(imageContours.size());
	warpAffine(gray_src, rot_image, rot_mat, dst_sz);//原图像旋转
	Mat ROI = rot_image(Rect(center.x - (rect.size.width / 2), center.y - (rect.size.height / 2), rect.size.width, rect.size.height));//提取ROI
	//imwrite("ROI.jpg", ROI);

	Mat rotate;
	flip(ROI, rotate, 1);
	transpose(rotate, rotate);
	//imwrite("rotate.jpg", rotate);

	Mat show = Mat::zeros(rotate.size(), CV_8UC1);

	threshold(rotate, show, 128, 255, THRESH_BINARY);
	for (int row = 0; row < show.rows; row++) {
		for (int col = 0; col < show.cols; col++) {
			int gray = rotate.at<uchar>(row, col);
			show.at<uchar>(row, col) = 255 - gray;
			if ((int)show.at<uchar>(row, col) > 128) {
				show.at<uchar>(row, col) = 1;
			}
			else {
				show.at<uchar>(row, col) = 0;
			}
		}
	}

	int c = rotate.rows; int d = rotate.cols;
	float HWpercent[] = { 0.5,0.6,0.7,0.8,0.9,1.0 };
	vector<vector<float>> mmmm;
	vector<float> max_Iou_index;
	for (int i = 0; i < sizeof(HWpercent) / sizeof(float); i++) {
		float percent = HWpercent[i];
		int square_height = floor(sqrt(area / percent));
		int square_width = floor(square_height*percent);

		for (int m = 0; m < c - square_width + 1; m++) {
			for (int n = 0; n < d - square_height + 1; n++) {
				Mat square = Mat::zeros(Size(d, c), CV_8UC1);
				Mat out = square.clone();
				square.rowRange(m, m + square_width) = 1;
				square.colRange(n, n + square_height) = 1;
				out = square.mul(show);
				int count2 = Sum_mat(out);
				float Iou = (float)count2 / count;
				vector<float>tmp;
				tmp.push_back(Iou);
				tmp.push_back(n);
				tmp.push_back(m);
				mmmm.push_back(tmp);
				float max_Iou = 0.0;
				if (max_Iou < Iou) {
					max_Iou = Iou;
					vector<float>tmp2;
					tmp2.push_back(square_height);
					tmp2.push_back(square_width);
					tmp2.push_back(percent);
					tmp2.push_back(m);
					tmp2.push_back(n);
					tmp2.push_back(Iou);
					max_Iou_index = tmp2;
				}
			}
		}
	}
	Point point1 = Size(max_Iou_index[3], max_Iou_index[4]);
	Point point2 = Size(max_Iou_index[0], max_Iou_index[1]);
	rectangle(rotate, point1, point2, Scalar(0, 255, 0), 2);
	imshow("src", rotate);
}
void unevenLightCompensate(Mat &image, int blockSize)
{//一种不均匀光照的补偿方法
	if (image.channels() == 3) cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i * blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j * blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
	imwrite("dst.jpg", image);
}
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
	Mat image = imread("0002.jpg");
	if (image.empty()) {
		cout << "打开图片失败，请检查!" << endl;
		return -1;
	}
	//imshow("原图",image);
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

void _unevenLightCompensate_(Mat &image, int blockSize)
{//一种不均匀光照的补偿方法blockSize一般为32
	if (image.channels() == 3) cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i * blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j * blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
	imwrite("dst.jpg", image);
}

int _Find_cricl_(Mat & img) {
	Mat img3, img2, img4;
	
	
	cvtColor(img, img2, CV_BGR2GRAY);   //把彩色图转换为黑白图像
	GaussianBlur(img2, img2, Size(15, 15), 2, 2);
	
	//threshold(img2, img3, 130, 255, THRESH_BINARY);  //图像二值化，，注意阈值变化
	


	Canny(img2, img3, 50, 200);//边缘检测
	//namedWindow("detect circles_3", CV_NORMAL);
	//imshow("detect circles_3", img3);
	cv::adaptiveThreshold(img3, img3, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 25, 10);
	//namedWindow("detecte circles_2", CV_NORMAL);
	//imshow("detecte circles_2", img3);

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(img3, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE,Point());//查找出所有的圆边界
	
	//HoughCircles(img3,hierarchy, CV_HOUGH_GRADIENT, 2, img3.rows/10, 10, 100, 0, 100);
	Mat imageContours = img; 
	Mat cimage = Mat::zeros(img3.size(), CV_8UC3);
	RotatedRect rect;

	for (size_t i = 0; i < contours.size(); i++) {
		size_t count = contours[i].size();
		if (count < 6) {
			continue;
		}
		Scalar color(rand() & 255, rand() & 255, rand() & 255);

		Mat pointsf;
		Mat(contours[i]).convertTo(pointsf,CV_32F);
		RotatedRect box = fitEllipse(contours[i]);
		if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30) {
			continue;
		}
		drawContours(cimage,contours,i, color,1,8, hierarchy);
		//ellipse(cimage,box,color,1,LINE_AA);
		//ellipse(cimage,box.center,box.size*0.5f,box.angle,0,360,color,1,LINE_AA);

		rect = minAreaRect(contours[i]);
		Point2f P[4];
		box.points(P);
		/*for (int j = 0; j <= 3; j++) {
			line(cimage, P[j], P[(j + 1) % 4], color, 2);
		}*/
		
		circle(imageContours, box.center,5,Scalar(0,0,255),-1, LINE_AA);
	}
	imwrite("imageContours.jpg", cimage);


	//int index = 0;
	//for (; index >= 0; index = hierarchy[index][0])
	////for(size_t index = 0;index<hierarchy.size();index++)
	//{
	//	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	//	//Scalar color(0, 255, 255);
	//	drawContours(img, contours, index, color, 2, 8, hierarchy);
	//}

	//namedWindow("detected circles_4", CV_NORMAL);
	imwrite("rest.jpg", imageContours);
	//imshow("detected circles_4", img);
	//标准圆在图片上一般是椭圆，所以采用OpenCV中拟合椭圆的方法求解中心
	//Mat pointsf;
	//Mat(contours[0]).convertTo(pointsf, CV_32F);
	//RotatedRect box = fitEllipse(pointsf);
	//cout << box.center;
	waitKey();

	return 0;
}

int highlight_remove_Chi(IplImage* src, IplImage* dst, double Re)
{
	
	//cvSaveImage("src.jpg", src);
	int height = src->height;
	int width = src->width;
	int step = src->widthStep;
	int i = 0, j = 0;
	unsigned char R, G, B, MaxC;
	double alpha, beta, alpha_r, alpha_g, alpha_b, beta_r, beta_g, beta_b, temp = 0, realbeta = 0, minalpha = 0;
	double gama, gama_r, gama_g, gama_b;
	unsigned char* srcData;
	unsigned char* dstData;
	for (i = 0; i < height; i++)
	{
		srcData = (unsigned char*)src->imageData + i * step;
		dstData = (unsigned char*)dst->imageData + i * step;
		for (j = 0; j < width; j++)
		{
			R = srcData[j * 3];
			G = srcData[j * 3 + 1];
			B = srcData[j * 3 + 2];

			alpha_r = (double)R / (double)(R + G + B);
			alpha_g = (double)G / (double)(R + G + B);
			alpha_b = (double)B / (double)(R + G + B);
			alpha = max(max(alpha_r, alpha_g), alpha_b);
			MaxC = max(max(R, G), B);// compute the maximum of the rgb channels
			minalpha = min(min(alpha_r, alpha_g), alpha_b);                 beta_r = 1 - (alpha - alpha_r) / (3 * alpha - 1);
			beta_g = 1 - (alpha - alpha_g) / (3 * alpha - 1);
			beta_b = 1 - (alpha - alpha_b) / (3 * alpha - 1);
			beta = max(max(beta_r, beta_g), beta_b);//将beta当做漫反射系数，则有                 // gama is used to approximiate the beta
			gama_r = (alpha_r - minalpha) / (1 - 3 * minalpha);
			gama_g = (alpha_g - minalpha) / (1 - 3 * minalpha);
			gama_b = (alpha_b - minalpha) / (1 - 3 * minalpha);
			gama = max(max(gama_r, gama_g), gama_b);

			temp = (gama*(R + G + B) - MaxC) / (3 * gama - 1);
			//beta=(alpha-minalpha)/(1-3*minalpha)+0.08;
			//temp=(gama*(R+G+B)-MaxC)/(3*gama-1);
			dstData[j * 3] = R - (unsigned char)(temp + 0.5);
			dstData[j * 3 + 1] = G - (unsigned char)(temp + 0.5);
			dstData[j * 3 + 2] = B - (unsigned char)(temp + 0.5);
		}
	}	
	return 1;
}

Mat gaussian_kernal(int kernelRowSize, int kernelColSize, double sigma0) {
	float halfRowSize = (kernelRowSize - 1) / 2.0;
	float halfColSize = (kernelColSize - 1) / 2.0;
	Mat K(kernelRowSize, kernelColSize, CV_32FC1);
	//生成二维高斯核 
	double s2 = 2.0 * sigma0 * sigma0;
	for (float i = (-halfRowSize); i <= halfRowSize; i += 1) {
		int m = i + halfRowSize;
		for (float j = (-halfColSize); j <= halfColSize; j += 1) {
			int n = j + halfColSize;
			float v = exp(-(1.0*i*i + 1.0*j*j) / s2);
			K.ptr(m)[n] = v;
		}
	}
	Scalar all = sum(K);
	Mat gaussK;
	K.convertTo(gaussK, CV_32FC1, (1 / all[0]));
	return gaussK;
}
template<typename T>
void GetGuassianKernel11(Mat gaus, const int size, const double sigma)
{
	const double PI = 4.0*atan(1.0); //圆周率π赋值
	int center = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			/*gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];*/
			gaus.at<uchar>(i, j) = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus.at<uchar>(i, j) /= sum;
		}
	}
	return;
}

template <typename T>
void GetGuassianKernel(T **gaus, const int size, const double sigma)
{
	const double PI = 4.0*atan(1.0); //圆周率π赋值
	int center = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] /= sum;
			
		}
		
	}
	return;
}
Mat gaussian_kernal2(int kernelSize, double sigma0)
{
	int halfSize = (kernelSize - 1) / 2;
	Mat K(kernelSize, kernelSize, CV_32FC1);

	//生成二维高斯核  
	double s2 = 2.0 * sigma0 * sigma0;
	for (int i = (-halfSize); i <= halfSize; i++)
	{
		int m = i + halfSize;
		for (int j = (-halfSize); j <= halfSize; j++)
		{
			int n = j + halfSize;
			float v = exp(-(1.0*i*i + 1.0*j*j) / s2);
			K.ptr<float>(m)[n] = v;
		}
	}
	Scalar all = sum(K);
	Mat gaussK;
	K.convertTo(gaussK, CV_32FC1, (1 / all[0]));


	return gaussK;
}
Mat mMeanImage(Mat &srcImage, int n)
{//多福图像叠加求平均值
	Mat aftImgnum(srcImage.rows, srcImage.cols, CV_32FC1);
	Mat mergy;  //存放平均值
	mergy.convertTo(aftImgnum, CV_32FC1, 1, 0);
	for (int i = 0; i < n; i++)
	{
		accumulate(srcImage, aftImgnum);  //将所有图像叠加
	}
	aftImgnum = aftImgnum / n; //求出平均图像
	aftImgnum.convertTo(mergy, CV_8UC1, 1, 0); //将平均图像存储在mergy中
	return mergy;
}

Mat Vec2Mat(double **array, int row, int col)
{

	Mat img(row, col, CV_64F);
	double *ptmp = NULL;
	for (int i = 0; i < row; ++i)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = array[i][j];
		}
	}
	return img;
}

void _Hightlightremove_matlab(Mat &src) {
	Mat dst(src.size(), CV_8U, Scalar(0));
	Mat hsv(src.size(), CV_8U,Scalar(0));
	Mat img_h(src.size(), CV_8U, Scalar(0));
	Mat img_s(src.size(), CV_8U, Scalar(0));
	Mat img_v(src.size(), CV_8U, Scalar(0));
	vector<Mat>hsv_vec;
	vector<Mat>hsv2rgb_vec;
	//分割HSV三通道
	cvtColor(src,hsv,CV_BGR2HSV);
	split(hsv, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	/*namedWindow("hsv", CV_NORMAL);
	imshow("hsv", img_v);*/

	//高斯滤波
	int HSIZE = min(src.rows,src.cols);
	Size hsize(3, 3);
	double q = sqrt(2);
	int SIGMA1 = 15, SIGMA2 = 80, SIGMA3 = 250;
	/*Mat F1 = gaussian_kernal2(HSIZE,SIGMA1/q);
	Mat F2 = gaussian_kernal2(HSIZE,  SIGMA2 / q);
	Mat F3 = gaussian_kernal2(HSIZE,  SIGMA3 / q);*/
	Mat F1(HSIZE, HSIZE, CV_8U);
	Mat F2(HSIZE, HSIZE, CV_8U);
	Mat F3(HSIZE, HSIZE, CV_8U);

	double **f1 = new double *[HSIZE];
	double **f2 = new double *[HSIZE];
	double **f3 = new double *[HSIZE];
	for (int i = 0; i < HSIZE; i++) {
		f1[i] = new double[HSIZE];
		f2[i] = new double[HSIZE];
		f3[i] = new double[HSIZE];
	}
	GetGuassianKernel<double>(f1, HSIZE, SIGMA1 / q);
	GetGuassianKernel<double>(f2, HSIZE, SIGMA2 / q);
	GetGuassianKernel<double>(f3, HSIZE, SIGMA3 / q);

	/*Mat F11, F22, F33;
	GetGuassianKernel11<uchar>(F11, HSIZE, SIGMA1 / q);
	GetGuassianKernel11<uchar>(F22, HSIZE, SIGMA2 / q);
	GetGuassianKernel11<uchar>(F33, HSIZE, SIGMA3 / q);*/

	F1 = Vec2Mat(f1, HSIZE, HSIZE);
	F2 = Vec2Mat(f2, HSIZE, HSIZE);
	F3 = Vec2Mat(f3, HSIZE, HSIZE);
	
	Mat gaus1, gaus2, gaus3, gaus4,gaus;
	Mat gaus11, gaus22, gaus33;
	Point anchor_center(-1, -1);
	/*filter2D(img_v,gaus11,-1,F11);
	filter2D(img_v, gaus22, -1, F22);
	filter2D(img_v, gaus33, -1, F33);
	imwrite("gaus11.jpg", gaus11);
	imwrite("gaus22.jpg", gaus22);
	imwrite("gaus33.jpg", gaus33);*/

	filter2D(img_v, gaus1, -1, F1);
	filter2D(img_v, gaus2, -1, F2);
	filter2D(img_v, gaus3, -1, F3);
	
	addWeighted(gaus1,0.5,gaus2,0.5,0,gaus4);
	addWeighted(gaus4, 0.5, gaus3, 0.5, 0, gaus);

	/*accumulate(gaus1,gaus4);
	accumulate(gaus2, gaus4);
	accumulate(gaus3, gaus4);
	gaus = gaus4 / 3;*/
	imwrite("F.jpg",gaus);
	
	//二维伽马卷积
	Scalar tempVal = mean(gaus);
	uchar m = tempVal[0];//未归一化均值和
	int height = img_v.rows;
	int weight = img_v.cols*img_v.channels();
	Mat out = Mat::zeros(img_v.size(), CV_8U);
	Mat gama = Mat::zeros(img_v.size(), CV_8U);
	Mat result = Mat::zeros(img_v.size(), CV_8U);

	MatIterator_<double> it, end, it_gama = gama.begin<double>(),end_gama= gama.end<double>();

	MatIterator_<double> it_out=out.begin<double>(), end_out=out.end<double>(), it_img_v = img_v.begin<double>(), end_img_v = img_v.end<double>();

	for (it = gaus.begin<double>(), end = gaus.end<double>(); it != end, it_gama!=end_gama,it_out!=end_out,it_img_v!=end_img_v; it++) {
		*it_gama = pow(0.5, ((m - *it) / m));
		*it_out = pow(*it_img_v, *it_gama);
		it_gama++;
		it_out++;
		it_img_v++;
	}
	
	
	//for (int i = 0; i < height; i++) {
	//	double *pdata = gaus.ptr<double>(i);
	//	double *pdata1 = gama.ptr<double>(i);
	//	double *pdata2 = img_v.ptr<double>(i);
	//	double *pdata3 = out.ptr<double>(i);
	//	for (int j = 0; j < weight; j++) {
	//		/*gama.at<double>(i, j) = pow(0.5,((m-gaus.at<double>(i,j))/m));
	//		out.at<double>(i, j) = pow(img_v.at<double>(i,j),gama.at<double>(i,j));*/
	//		pdata1[j]=pow(0.5, ((m-pdata[j]) / m));
	//		pdata3[j] = pow(pdata2[j], pdata1[j]);
	//	}
	//}
	hsv2rgb_vec.push_back(img_h);
	hsv2rgb_vec.push_back(img_s);
	hsv2rgb_vec.push_back(out);
	imwrite("out.jpg", out);
	merge(hsv2rgb_vec,result);
	imwrite("HSV.jpg",result);
	cvtColor(result, result,CV_HSV2BGR);
	//normalize(result, result, 1, 0);
	imwrite("RGB.jpg", result);
	waitKey(0);

}

Mat RGB2HSV(Mat &src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(r, min(g, b));
			float maxx = max(r, max(g, b));
			dst.at<Vec3f>(i, j)[2] = maxx; //V
			float delta = maxx - minn;
			float h, s;
			if (maxx != 0) {
				s = delta / maxx;
			}
			else {
				s = 0;
			}
			if (r == maxx) {
				h = (g - b) / delta;
			}
			else if (g == maxx) {
				h = 2 + (b - r) / delta;
			}
			else {
				h = 4 + (r - g) / delta;
			}
			h *= 60;
			if (h < 0)
				h += 360;
			dst.at<Vec3f>(i, j)[0] = h;
			dst.at<Vec3f>(i, j)[1] = s;
		}
	}
	return dst;
}

Mat HSV2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	float r, g, b, h, s, v;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			h = src.at<Vec3f>(i, j)[0];
			s = src.at<Vec3f>(i, j)[1];
			v = src.at<Vec3f>(i, j)[2];
			if (s == 0) {
				r = g = b = v;
			}
			else {
				h /= 60;
				int offset = floor(h);
				float f = h - offset;
				float p = v * (1 - s);
				float q = v * (1 - s * f);
				float t = v * (1 - s * (1 - f));
				switch (offset)
				{
				case 0: r = v; g = t; b = p; break;
				case 1: r = q; g = v; b = p; break;
				case 2: r = p; g = v; b = t; break;
				case 3: r = p; g = q; b = v; break;
				case 4: r = t; g = p; b = v; break;
				case 5: r = v; g = p; b = q; break;
				default:
					break;
				}
			}
			dst.at<Vec3b>(i, j)[0] = int(b * 255);
			dst.at<Vec3b>(i, j)[1] = int(g * 255);
			dst.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	return dst;
}

Mat _Hightlightremove_matlab2(Mat src) {
	//去高光算法，二维伽马函数
	int row = src.rows;
	int col = src.cols;
	Mat now = RGB2HSV(src);
	Mat H(row, col, CV_32FC1);
	Mat S(row, col, CV_32FC1);
	Mat V(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			H.at<float>(i, j) = now.at<Vec3f>(i, j)[0];
			S.at<float>(i, j) = now.at<Vec3f>(i, j)[1];
			V.at<float>(i, j) = now.at<Vec3f>(i, j)[2];
		}
	}
	int kernel_size = min(row, col);
	if (kernel_size % 2 == 0) {
		kernel_size -= 1;
	}
	float SIGMA1 = 15;
	float SIGMA2 = 80;
	float SIGMA3 = 250;
	float q = sqrt(2.0);
	Mat F(row, col, CV_32FC1);
	Mat F1, F2, F3;
	GaussianBlur(V, F1, Size(kernel_size, kernel_size), SIGMA1 / q);
	GaussianBlur(V, F2, Size(kernel_size, kernel_size), SIGMA2 / q);
	GaussianBlur(V, F3, Size(kernel_size, kernel_size), SIGMA3 / q);
	
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			F.at <float>(i, j) = (F1.at<float>(i, j) + F2.at<float>(i, j) + F3.at<float>(i, j)) / 3.0;
		}
	}
	imwrite("F3.jpg", F);
	float average = mean(F)[0];
	Mat out(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float gamma = powf(0.5, (average - F.at<float>(i, j)) / average);
			out.at<float>(i, j) = powf(V.at<float>(i, j), gamma);
		}
	}
	vector <Mat> v;
	v.push_back(H);
	v.push_back(S);
	v.push_back(out);
	Mat merge_;
	merge(v, merge_);
	Mat dst = HSV2RGB(merge_);
	imwrite("dst000.jpg",dst);
	return dst;
}

void _MaxLiantongyu_(Mat img) {
	if (img.empty())
		exit(-1);
	Mat src(img.size(),img.type());
	threshold(img, src, 127, 255, THRESH_BINARY);   // 二值化图像
	imwrite("src.jpg",src);
	vector<Feather> featherList;                    // 存放连通域特征
	Mat dst;
	cout << "连通域数量： " << BwLabel(src, dst, featherList) << endl;

	// 为了方便观察，可以将label“放大”
	for (int i = 0; i < dst.rows; i++)
	{
		uchar *p = dst.ptr<uchar>(i);
		for (int j = 0; j < dst.cols; j++)
		{
			p[j] = 30 * p[j];
		}
	}

	cout << "标号" << "\t" << "面积" << endl;
	for (vector<Feather>::iterator it = featherList.begin(); it < featherList.end(); it++)
	{
		cout << it->label << "\t" << it->area << endl;
		rectangle(dst, it->boundingbox, 255);
	}

	//imshow("src", src);
	imwrite("liantongyu.jpg", dst);

	waitKey();
	destroyAllWindows();

	system("pause");

	
}

int _Image_process_() {
	Mat img_test = imread("view33.jpg");
	if (img_test.empty()) {
		cout << "打开图片失败，请重新检查!" << endl;
		return -1;
	}
	
	//_Hightlightremove_matlab(img_test);
	//_Hightlightremove_matlab2(img_test);

	return -1;
}

bool check(string str) {
	for (int i = 0; i < str.length(); i++) {
		if ((str[i] > '9' || str[i] < '0') && (str[i] != '.')) {
			return false;
		}
	}
	return true;
}
int main()
{

	_Image_process_();
	
	//cin.get();


	return 0;

	
}

