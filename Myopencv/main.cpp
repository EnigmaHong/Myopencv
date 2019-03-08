#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <windows.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

string xmlPath = "haarcascade_frontalface_alt.xml";
//xmlpath 字符串记录那个.xml文件的路径
Mat src, gray_src, drawImg;
int threshold_v = 170;
int threshold_max = 255;
const char* output_win = "rectangle-demo";
RNG rng(12345);
void print_Mat(Mat img) {
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			cout << (int)img.at<uchar>(row, col) << " ";
		}
		cout << endl;
	}
}
int Sum_mat(Mat img) {
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

int _Image_process_() {
	Mat image = imread("007.jpg");
//	Mat result_img(image.resize(),CV_32FC3);
	if (image.empty) {
		cout << "打开图片失败，请重新检查!" << endl;
		return -1;
	}
	
	IplImage * img =cvLoadImage("007.jpg");
	IplImage *pimg = cvCreateImage(cvGetSize(img), 8, 1);
	cvSobel(img, pimg,3,3,3);
	cvSaveImage("soble.jpg",pimg);
	


}


int main()
{
	_Image_process_();
	
	//cin.get();



	
}
