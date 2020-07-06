// Wheel.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#include"Defines.h"
//#include"Algo.h"
//#include"FileList.h"
#include"Wheel.h"
//#include"Struct2d.h"

//#include"../ransac/ransac_line2d.h"
//#include"../ransac/ransac_circle2d.h"
//#include"../ransac/ransac_ellipse2d.h"

using namespace std;
using namespace cv;

int g_j = 0;

void testConnectAnalyze(cv::Mat& src);


int main()
{
	Mat mImage, mImage2;

#ifdef COMPANY
	String sPath("E:\\Pictures\\First");
	String sName("E:\\Pictures\\First\\car_9.jpg");
	string sTempPath("E:\\Pictures\\First\\Temp");
#endif // COMPANY

#ifdef USEROWN
	String sPath("F:\\Pictures\\First");
	String sName("F:\\Pictures\\First\\car_9.jpg");
	string sTempPath("F:\\Pictures\\First\\Temp");
#endif // USEROWN
	
	std::vector<String> vsFileNames, vsFileOwnNames;
	getFiles2(sPath, vsFileNames, vsFileOwnNames, emJpg);

	cv::Mat Temp = cv::imread(sPath + "\\Temp\\car_1_temp.jpg");

	std::vector<cv::Mat> Temps;
	getTemps(sTempPath, Temps);
	for (size_t i = 0; i < vsFileNames.size(); i++)
	{
		/*cvMatch(Temp, vsFileNames[i],vsFileOwnNames[i]);
		cv::Size sz = Temp.size();
		cout << sz << endl;*/

		//cvMatchs(Temps, vsFileNames[i]);
		cvSameSzMatchs(Temps, vsFileNames[i]);
	}




	return 0;
}


void getTemps(string TempPath, std::vector<cv::Mat>& Temps)
{
	std::vector<String> vsFileNames, vsFileOwnNames;
	getFiles2(TempPath, vsFileNames, vsFileOwnNames, emJpg);

	for (size_t i = 0; i < vsFileNames.size(); i++)
	{
		Temps.push_back(cv::imread(vsFileNames[i]));

	}
}


bool cvSameSzMatchs(std::vector<cv::Mat>& Temps, string Filename)
{
	if (Temps.size() == 0)return false;
	if (Filename.empty())return false;
	bool bFilter(false);
	bool bMoveLoc(true);

	cv::Mat img1 = imread(Filename);
	cv::Mat img2 = img1.clone();
	cv::Mat dstImg;
	dstImg.create(img1.dims, img1.size, img1.type());
	
	int matchMethod = 5;

	std::vector<std::pair<double, double>> vpMax;
	std::vector<std::pair<Point, Point>> vpPoint;
	std::vector<cv::Size> vDstSize;//图像尺寸记录，分为width和height//有时候感觉还是定义结构体好一些，更清晰
	vpMax.clear();
	vpPoint.clear();
	vDstSize.clear();
	int j = 0;

	std::vector<Mat> vec_dst;
	std::vector<cv::Size> vec_dstSize;
	vec_dst.clear();
	vec_dstSize.clear();

	//把所有模板依次匹配，取分高的组成一组，能够更好看，但是其实没卵用
	//std::vector<cv::Size> vecTempSize;
	cv::Size szMean(0, 0);
	for (size_t i = 0; i < Temps.size(); i++)
	{
		std::pair<double, double> pad;
		std::pair<Point, Point> paPoint;
		cv::Mat& Temp_src = Temps[i];
		cv::Mat Temp;

		cv::Size sz = Temp_src.size();
		szMean += sz;
		//cv::matchTemplate(img1, Temp, dstImg, 0);
		cv::Mat img3;
		if (bMoveLoc)
		{
			int h = sz.height ;
			int w = sz.width ;
			copyMakeBorder(img1, img3, h/2, h/2, w/2, w/2, BORDER_WRAP);
		}
		else 
		{
			copyMakeBorder(img1, img3, 0, sz.height - 1, 0, sz.width - 1, BORDER_WRAP);
		}
		
		if (bFilter)
		{
			Image_fileter(img3, img3);
			Image_fileter(Temp_src, Temp);
		}
		else
		{
			Temp = Temp_src;
		}

		
		cv::matchTemplate(img3, Temp, dstImg, matchMethod);
		

		vec_dst.push_back(dstImg);
		vec_dstSize.push_back(dstImg.size());
#ifdef _DEBUG
		if (1)
		{
			Mat dstShow;
			double dMin,dMax;
			Point p1, p2;
			minMaxLoc(dstImg, &dMin, &dMax, &p1, &p2);
			dstShow = (dstImg - dMin) / (dMax - dMin) * 255;
			if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
			{
				dstShow = 255 - dstShow;
			}

			//cv::normalize(dstImg, dstShow, 0, 1, 32);//需要修正一下
			//dstShow = 1. - dstShow;
			string spath = "E:\\Pictures\\First\\Temp\\hot_show\\" + to_string(matchMethod);
			MkDir(spath);
			cv::imwrite(spath + "\\" + to_string(i) + "-" + to_string(g_j++) + ".jpg", dstShow/* * 255*/);
		}
#endif // _DEBUG

		
		
		vDstSize.push_back(Temps[i].size());

		cv::Mat Mask = cv::Mat::ones(dstImg.size(), CV_8UC1) * 255;
		roiToMask(Mask,
			//maxPoint,
			Point(0, 0),
			Point(img1.cols, img1.rows / 2));

		cv::Point minPoint;
		cv::Point maxPoint;
		double minVal = 0;
		double maxVal = 0;

		//cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);
		if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
		{
			findDoubleMin(dstImg, maxVal, maxPoint, Mask);//matchLoc = minLoc;
		}
		else
		{
			findDoubleMax(dstImg, maxVal, maxPoint, Mask);//matchLoc = maxLoc;
		}
		//findDoubleMin(dstImg, maxVal, maxPoint, Mask);
		pad.first = maxVal / (Temp.cols * Temp.rows);
		paPoint.first = maxPoint;


		roiToMask(Mask,
			//maxPoint,
			Point(maxPoint.x - Temp.cols / 2, maxPoint.y - Temp.rows / 2),
			Point(maxPoint.x + Temp.cols / 2, maxPoint.y + Temp.rows / 2));
		//Point(maxPoint.x + Temp.cols , maxPoint.y + Temp.rows ));
	//cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);

		maxVal -= maxVal; maxPoint -= maxPoint;
		if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
		{
			findDoubleMin(dstImg, maxVal, maxPoint, Mask);//matchLoc = minLoc;
		}
		else
		{
			findDoubleMax(dstImg, maxVal, maxPoint, Mask);//matchLoc = maxLoc;
		}
		//findDoubleMin(dstImg, maxVal, maxPoint, Mask);
		pad.second = maxVal / (Temp.cols * Temp.rows);
		paPoint.second = maxPoint;

		vpMax.push_back(pad);
		vpPoint.push_back(paPoint);


		Mat mDstMean;
		slidSumMean(dstImg, mDstMean, Temp.size());
		maxVal -= maxVal; maxPoint -= maxPoint;
		if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
		{
			findDoubleMin(mDstMean, maxVal, maxPoint, Mask);//matchLoc = minLoc;
		}
		else
		{
			findDoubleMax(mDstMean, maxVal, maxPoint, Mask);//matchLoc = maxLoc;
		}
		std::sort(mDstMean.ptr<float>(0), mDstMean.ptr<float>(mDstMean.rows+ mDstMean.cols - 1)/*, IsBiger*/);
#ifdef _DEBUG
		if (1)
		{
			cv::Point minPoint;
			cv::Point maxPoint;
			double minVal = 0;
			double maxVal = 0;

			cv::minMaxLoc(mDstMean, &minVal, &maxVal, &minPoint, &maxPoint);

		}
#endif // _DEBUG

		


	}
	szMean.height /= Temps.size();
	szMean.width /= Temps.size();

	
	cv::Point p_first_best(0, 0), p_second_best(0, 0);
	cv::Size sz_first_best(cv::Size(0, 0)), sz_second_best(cv::Size(0, 0));//按道理讲，这两个的值应该是一样的才对
	findBestPair(p_first_best, p_second_best, sz_first_best, sz_second_best,  vpMax, vpPoint, vDstSize, matchMethod);
	
	cv::rectangle(img1, p_first_best, cv::Point(p_first_best.x + sz_first_best.width, p_first_best.y + sz_first_best.height), cv::Scalar(0, 255, 0), 2, 8);
	cv::rectangle(img1, p_second_best, cv::Point(p_second_best.x + sz_second_best.width, p_second_best.y + sz_second_best.height), cv::Scalar(0, 255, 0), 2, 8);

#ifdef _DEBUG
	if (1)
	{
		string spath = "E:\\Pictures\\First\\Temp\\hot_after\\" +to_string(matchMethod);
		MkDir(spath);
		cv::imwrite(spath + "\\" + to_string(g_j++) + ".jpg", img1);
	}
#endif // _DEBUG

	


	

	cout << "==================" << endl;

	//vector<Point> vptDst; vptDst.clear();
	addDstImgAndFindBox(img2, vec_dst, vec_dstSize, szMean, matchMethod);



}

bool IsBiger(float *a, float *b)
{
	return a > b;
}

//滑动求和
bool slidSumMean(cv::Mat &Src, Mat &Dst,cv::Size sz)
{
	Mat Img;
	bool bMoveLoc(true);
	int h = sz.height;
	int w = sz.width;
	sz.width = sz.height = min(h, w);
	/*if (bMoveLoc)
	{
		
		copyMakeBorder(*Src, Img, h / 2, h / 2, w / 2, w / 2, BORDER_WRAP);
	}*/

	
	boxFilter(Src, Dst, -1/*CV_32FC1*/, sz/*, Point(-1, -1), false, BORDER_WRAP*/);

	return false;
}





//找出认为匹配最好的一对点
void findBestPair(cv::Point &p_first_best, cv::Point &p_second_best, cv::Size &sz_first_best, cv::Size &sz_second_best, 
	std::vector<std::pair<double, double>> vpMax, 
	std::vector<std::pair<Point, Point>> vpPoint, 
	std::vector<cv::Size> vDstSize,
	int matchMethod)
{
	

	if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
	{//找最小值
		double d_first_best = numeric_limits<double>::max(), d_second_best = numeric_limits<double>::max();
		for (size_t i = 0; i < vpMax.size(); i++)
		{
			if (vpPoint[i].first.x == 0 && vpPoint[i].first.y == 0) continue;
			if (d_first_best > vpMax[i].first)
			{
				d_first_best = vpMax[i].first;
				p_first_best = vpPoint[i].first;
				sz_first_best = vDstSize[i];
			}
			if (d_second_best > vpMax[i].second)
			{
				d_second_best = vpMax[i].second;
				p_second_best = vpPoint[i].second;
				sz_second_best = vDstSize[i];
			}
		}
	}
	else//找最大值
	{
		double d_first_best = numeric_limits<double>::min(), d_second_best = numeric_limits<double>::min();
		for (size_t i = 0; i < vpMax.size(); i++)
		{
			if (vpPoint[i].first.x == 0 && vpPoint[i].first.y == 0) continue;
			if (d_first_best < vpMax[i].first)
			{
				d_first_best = vpMax[i].first;
				p_first_best = vpPoint[i].first;
				sz_first_best = vDstSize[i];
			}
			if (d_second_best < vpMax[i].second)
			{
				d_second_best = vpMax[i].second;
				p_second_best = vpPoint[i].second;
				sz_second_best = vDstSize[i];
			}
		}
	}
	
}

bool cvMatchs(std::vector<cv::Mat>& Temps, string Filename)
{
	if (Temps.size() == 0)return false;
	if (Filename.empty())return false;

	cv::Mat img1 = imread(Filename);
	cv::Mat img2 = img1.clone();
	cv::Mat dstImg;
	dstImg.create(img1.dims, img1.size, img1.type());

	std::vector<std::pair<double, double>> vpMax;
	std::vector<std::pair<Point, Point>> vpPoint;
	std::vector<cv::Size> vDstSize;//图像尺寸记录，分为width和height//有时候感觉还是定义结构体好一些，更清晰
	vpMax.clear();
	vpPoint.clear();
	vDstSize.clear();
	int j = 0;

	std::vector<Mat> vec_dst;
	std::vector<cv::Size> vec_dstSize;
	vec_dst.clear();
	vec_dstSize.clear();

	//把所有模板依次匹配，取分高的组成一组，能够更好看，但是其实没卵用
	//std::vector<cv::Size> vecTempSize;
	cv::Size szMean(0, 0);
	for (size_t i = 0; i < Temps.size(); i++)
	{
		std::pair<double, double> pad;
		std::pair<Point, Point> paPoint;
		cv::Mat& Temp = Temps[i];
		
		cv::Size sz = Temp.size();
		szMean += sz;
		cv::matchTemplate(img1, Temp, dstImg, 0);

		vec_dst.push_back(dstImg);
		vec_dstSize.push_back(dstImg.size());
		//cv::normalize(dstImg, dstImg, 0, 1, 32);
		//dstImg = 1. - dstImg;
		string spath = "E:\\Pictures\\First\\Temp\\hot";
		MkDir(spath);
		cv::imwrite(spath + "\\" + to_string(i) + "-" + to_string(g_j++) + ".jpg", dstImg * 255);
		//dstImg = 255 - dstImg;
		vDstSize.push_back(Temps[i].size());

		cv::Mat Mask = cv::Mat::ones(dstImg.size(), CV_8UC1) * 255;
		roiToMask(Mask,
			//maxPoint,
			Point(0,0),
			Point(img1.cols, img1.rows / 2));

		cv::Point minPoint;
		cv::Point maxPoint;
		double minVal = 0;
		double maxVal = 0;

		//cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);
		findDoubleMin(dstImg, maxVal, maxPoint, Mask);
		pad.first = maxVal/(Temp.cols * Temp.rows);
		paPoint.first = maxPoint;
		

		roiToMask(Mask,
			//maxPoint,
			Point(maxPoint.x - Temp.cols / 2, maxPoint.y - Temp.rows / 2),
			Point(maxPoint.x + Temp.cols / 2, maxPoint.y + Temp.rows / 2));
			//Point(maxPoint.x + Temp.cols , maxPoint.y + Temp.rows ));
		//cv::minMaxLoc(dstImg, &minVal, &maxVal, &minPoint, &maxPoint);

		maxVal -= maxVal; maxPoint -= maxPoint;
		findDoubleMin(dstImg, maxVal, maxPoint, Mask);
		pad.second = maxVal / (Temp.cols * Temp.rows);
		paPoint.second = maxPoint;

		vpMax.push_back(pad);
		vpPoint.push_back(paPoint);
	}
	szMean.height /= Temps.size();
	szMean.width /= Temps.size();

	double d_first_best = numeric_limits<double>::max(), d_second_best = numeric_limits<double>::max();
	cv::Point p_first_best(0,0), p_second_best(0,0);
	cv::Size sz_first_best(cv::Size(0,0)), sz_second_best(cv::Size(0, 0));//按道理讲，这两个的值应该是一样的才对
	for (size_t i = 0; i < Temps.size(); i++)
	{
		if (vpPoint[i].first.x == 0 && vpPoint[i].first.y == 0) continue;
		if (d_first_best > vpMax[i].first) 
		{ 
			d_first_best = vpMax[i].first;
			p_first_best = vpPoint[i].first;
			sz_first_best = vDstSize[i];
		}
		if (d_second_best > vpMax[i].second)
		{
			d_second_best = vpMax[i].second;
			p_second_best = vpPoint[i].second;
			sz_second_best = vDstSize[i];
		}	
	}
	cv::rectangle(img1, p_first_best, cv::Point(p_first_best.x + sz_first_best.width, p_first_best.y + sz_first_best.height), cv::Scalar(0, 255, 0), 2, 8);
	cv::rectangle(img1, p_second_best, cv::Point(p_second_best.x + sz_second_best.width, p_second_best.y + sz_second_best.height), cv::Scalar(0, 255, 0), 2, 8);


	string spath = "E:\\Pictures\\First\\Temp\\hot_after";
	MkDir(spath);
	cv::imwrite(spath + "\\" + to_string(g_j++) + ".jpg", img1);

	cout << "==================" << endl;

	//vector<Point> vptDst; vptDst.clear();
	addDstImgAndFindBox(img2,vec_dst, vec_dstSize,szMean,0);


	
}

bool addDstImgAndFindBox(Mat &Src,std::vector<cv::Mat> &vec_dstImg,std::vector<cv::Size> &vec_dstSize,cv::Size szMean,int matchMethod)
{
	//检查
	if (vec_dstImg.size() == 0)return false;
	if (vec_dstSize.size() == 0)return false;

	//查找最大尺寸
	cv::Size szMax(0, 0);// szMean(0, 0);
	for (size_t i = 0; i < vec_dstSize.size(); i++)
	{
		if (szMax.height < vec_dstSize[i].height)szMax.height = vec_dstSize[i].height;
		if (szMax.width < vec_dstSize[i].width)szMax.width = vec_dstSize[i].width;
	}
	

	
	//所有图等尺寸,并且归一化相加
	std::vector<cv::Mat> vec_NewMat;
	vec_NewMat.clear();
	cv::Mat MatSum = cv::Mat::zeros(szMax, CV_32FC1);
	Mat MatSumShow = cv::Mat::zeros(szMax, CV_32FC1);
	for (size_t i = 0; i < vec_dstImg.size(); i++)
	{
		cv::Mat newMatNorm = cv::Mat::ones(szMax, CV_32FC1) /** numeric_limits<float>::max()*/;
		cv::Mat matNorm;

		//cv::normalize(vec_dstImg[i], matNorm, 0, 1, 32);
		double dMin, dMax;
		Point p1, p2;
		minMaxLoc(vec_dstImg[i], &dMin, &dMax, &p1, &p2);
		matNorm = (vec_dstImg[i] - dMin) / (dMax - dMin) * 255;
		if (matchMethod == TM_SQDIFF || matchMethod == TM_SQDIFF_NORMED)
		{
			matNorm = 255 - matNorm;
		}
		MatExternRB(matNorm, newMatNorm,/*cv::Size szAimSize,*//*numeric_limits<float>::max()*//*1.0*/255);
		MatSumShow = MatSumShow + newMatNorm / vec_dstImg.size();
		
		cv::Mat newMat = cv::Mat::ones(szMax, CV_32FC1) * numeric_limits<float>::max();
		MatExternRB(vec_dstImg[i], newMat,/*cv::Size szAimSize,*/numeric_limits<float>::max());
		MatSum = MatSum + newMat / vec_dstImg.size();
	}
#ifdef _DEBUG
	if (1)
	{
		
		string spath = "E:\\Pictures\\First\\Temp\\hot_sum\\" + to_string(matchMethod);
		MkDir(spath);
		//imwrite(spath + "\\" + to_string(g_j++) + ".jpg", (1.0 - MatSumShow) /** 255*/);
		imwrite(spath + "\\" + to_string(g_j++) + ".jpg", MatSumShow /** 255*/);
	}
#endif // _DEBUG

	
	
	double dMin;
	Point ptMin;
	Mat Mask = cv::Mat::ones(MatSum.size(), CV_8UC1);
	roiToMask(Mask, Point(0, 0), Point(Mask.cols-1, Mask.rows / 2));
	findDoubleMin(MatSum, dMin, ptMin, Mask);
	cv::rectangle(Src, ptMin, cv::Point(ptMin.x + szMean.width, ptMin.y + szMean.height), cv::Scalar(0, 255, 0), 2, 8);

	roiToMask(Mask, Point(ptMin.x - szMean.width/2, ptMin.y -szMean.height),
		Point(ptMin.x + szMean.width / 2, ptMin.y + szMean.height));
	dMin -= dMin; ptMin -= ptMin;
	findDoubleMin(MatSum, dMin, ptMin, Mask);
	cv::rectangle(Src, ptMin, cv::Point(ptMin.x + szMean.width, ptMin.y + szMean.height), cv::Scalar(0, 255, 0), 2, 8);

#ifdef _DEBUG
	if (1)
	{
		string spath = "E:\\Pictures\\First\\Temp\\hot_sum_after";
		MkDir(spath);
		imwrite(spath + "\\" + to_string(g_j++) + ".jpg", Src);
	}
#endif // _DEBUG






	


	



	return true;

}

void Image_fileter(Mat resImg, Mat& output_img)
{
	Mat _resImg;
	if (resImg.channels() != 1) cvtColor(resImg, _resImg, COLOR_BGR2GRAY);   //转为gray图
	/*均衡化处理*/
	equalizeHist(_resImg, output_img);//均衡化就是把直方图每个灰度级进行归一化处理，根据相应的灰度值修正原图的每个像素。
	/*高斯滤波*/   //高斯滤波就是每一个像素点的值都有其本身和领域内的其他像素值经过加权平均后得到的
	Mat Kernel = getStructuringElement(cv::MORPH_RECT, Size(5, 5), Point(-1, -1));  //getStructuringElement()可以生成形态学操作中用到的核  矩形  5*5 个1
	GaussianBlur(output_img, output_img, Size(5, 5), 0);  //高斯处理
	/*求梯度*/
	morphologyEx(output_img, output_img, cv::MORPH_GRADIENT, Kernel); // 形态学梯度保留物体的边缘轮廓  膨胀与腐蚀之差
}

//矩阵扩大填值
//填的是浮点float型
bool MatExternRB(cv::Mat& Src, cv::Mat &Dst,/*cv::Size szAimSize,*/float val)
{
	if (Src.empty()) return false;
	if (Dst.empty()) return false;

	for (size_t x = 0; x < Dst.cols; x++)
	{
		for (size_t y = 0; y < Dst.rows; y++)
		{
			if(x<Src.cols && y<Src.rows)
			{ 
				Dst.at<float>(y, x) = Src.at<float>(y, x);
			}
			else
			{
				Dst.at<float>(y, x) = val;
			}
		}
	}

	return true;
}

//
bool findDoubleMin(cv::Mat& src, double& dMin, cv::Point& ptMin, cv::Mat& mask)
{
	if (src.empty())return false;
	if (mask.empty())return false;
	if (src.size() != mask.size())return false;

	dMin = numeric_limits<double>::max();
	for (size_t x = 0; x < src.cols; x++)
	{
		for (size_t y = 0; y < src.rows; y++)
		{
			if (mask.at<uchar>(y, x)>0 && src.at<float>(y, x) < dMin)
			{
				dMin = src.at<float>(y, x);
				ptMin.x = x;
				ptMin.y = y;

			}
		}
	}
	return true;
}
//
bool findDoubleMax(cv::Mat& src, double &dMax,cv::Point &ptMax, cv::Mat &mask)
{
	if (src.empty())return false;
	if (mask.empty())return false;
	if (src.size() != mask.size())return false;

	dMax = numeric_limits<double>::min();
	for (size_t x = 0; x < src.cols; x++)
	{
		for (size_t y = 0; y < src.rows; y++)
		{
			if (mask.at<uchar>(y,x) && src.at<float>(y, x) > dMax)
			{
				dMax = src.at<float>(y, x);
				ptMax.x = x;
				ptMax.y = y;

			}
		}
	}
	return true;
}


void roiToMask(cv::Mat& Src, Point p1, Point p2)
{
	roiToVal(Src, p1, p2, 0);
}

void roiToVal(cv::Mat& Src, Point p1, Point p2, int val)
{
	int minx = p1.x > p2.x ? p2.x : p1.x; minx = minx > 0 ? minx : 0;
	int maxx = p1.x > p2.x ? p1.x : p2.x; maxx = maxx > 0 ? maxx : 0;
	int miny = p1.y > p2.y ? p2.y : p1.y; miny = miny > 0 ? miny : 0;
	int maxy = p1.y > p2.y ? p1.y : p2.y; maxy = maxy > 0 ? maxy : 0;

	for (int x = minx; x < maxx && x < Src.cols; x++)
	{
		for (int y = miny; y < maxy && y < Src.rows; y++)
		{
			Src.at<uchar>(y, x) = val;
		}
	}
}

bool cvMatch(cv::Mat &Temp, string Filename,string ownname)
{
	if (Temp.empty())return false;
	if (Filename.empty())return false;
	cv::Mat img1 = imread(Filename);//cv::imread(sPath + "\\car_1.jpg");
	cv::Mat img2 = Temp;//cv::imread(sPath + "\\Temp\\car_1_temp.jpg");

	//步骤二：创建一个空画布用来绘制匹配结果
	cv::Mat dstImg;
	dstImg.create(img1.dims, img1.size, img1.type());
	//cv::imshow("createImg", dstImg);

	//步骤三：匹配，最后一个参数为匹配方式，共有6种，详细请查阅函数介绍
	cv::matchTemplate(img1, img2, dstImg, 0);

	//步骤四：归一化图像矩阵，可省略
	cv::normalize(dstImg, dstImg, 0, 1, 32);

	//步骤五：获取最大或最小匹配系数
	//首先是从得到的 输出矩阵中得到 最大或最小值（平方差匹配方式是越小越好，所以在这种方式下，找到最小位置）
	//找矩阵的最小位置的函数是 minMaxLoc函数
	cv::Point minPoint;
	cv::Point maxPoint;
	double* minVal = 0;
	double* maxVal = 0;
	dstImg = 1 - dstImg;//翻转灰度
	cv::minMaxLoc(dstImg, minVal, maxVal, &minPoint, &maxPoint);


	//步骤六：开始正式绘制
	cv::rectangle(img1, maxPoint, cv::Point(maxPoint.x + img2.cols, maxPoint.y + img2.rows), cv::Scalar(0, 255, 0), 2, 8);

	//选出第二组
	cv::Mat Mask = cv::Mat::ones(dstImg.size(), CV_8UC1) * 255;//把已经找出来的区域给屏蔽掉，可以选择操作Mask,也可以直接在原图上操作
	for (size_t i = maxPoint.x - img2.cols / 2; i < maxPoint.x + img2.cols && i < Mask.cols; i++)
	{
		for (size_t j = maxPoint.y - img2.rows / 2; j < maxPoint.y + img2.rows && j < Mask.rows; j++)
		{
			Mask.at<uchar>(j, i) = 0;
		}
	}
	//正式选出第二组并绘制
	minVal = 0; maxVal = 0; minPoint -= minPoint; maxPoint -= maxPoint;
	cv::minMaxLoc(dstImg, minVal, maxVal, &minPoint, &maxPoint, Mask);
	cv::rectangle(img1, maxPoint, cv::Point(maxPoint.x + img2.cols, maxPoint.y + img2.rows), cv::Scalar(0, 255, 0), 2, 8);

	string sPath = "E:\\Pictures\\First\\";
	MkDir(sPath + "car");
	imwrite(sPath + "car\\" +ownname, img1);

	return true;
}


int premain()//用分割的方法的主函数
{
	Mat mImage, mImage2;

#ifdef COMPANY
	String sPath("E:\\Pictures\\First");
	String sName("E:\\Pictures\\First\\car_9.jpg");
#endif // COMPANY

#ifdef USEROWN
	String sPath("F:\\Pictures\\First");
	String sName("F:\\Pictures\\First\\car_9.jpg");
#endif // USEROWN

	//Dir(sPath, emJpg);
	std::vector<String> vsFileNames, vsFileOwnNames;
	getFiles2(sPath, vsFileNames, vsFileOwnNames, emJpg);

	stRegment stR;
	stR.spatialRad = 50;  //空间窗口大小
	stR.colorRad = 40;   //色彩窗口大小
	stR.maxPyrLevel = 2;  //金字塔层数

	Algo detect;
	detect.setPath(sPath);
	detect.setRegment(stR);
	for (size_t i = 0; i < vsFileNames.size(); i++)
	{

		detect.setFileName(vsFileNames[i], vsFileOwnNames[i]);
		detect.run();
	}

	return 0;
}

int before()
{
	Mat mImage, mImage2;

#ifdef COMPANY
	String sPath("E:\\Pictures\\First");
	String sName("E:\\Pictures\\First\\car_9.jpg");
#endif // COMPANY

#ifdef USEROWN
	String sPath("F:\\Pictures\\First");
	String sName("F:\\Pictures\\First\\car_9.jpg");
#endif // USEROWN

	//Dir(sPath, emJpg);
	std::vector<String> vsFileNames, vsFileOwnNames;
	getFiles2(sPath, vsFileNames, vsFileOwnNames, emJpg);

	Algo detect;
	for (size_t i = 0; i < vsFileNames.size(); i++)
	{
		
		detect.setFileName(vsFileNames[i], vsFileOwnNames[i]);











		//读取图片
		mImage = imread(vsFileNames[i]);
		cv::GaussianBlur(mImage, mImage2, cv::Size(3, 3), 1.5, 1.5);
		cout << "=========start to deal with " << vsFileNames[i] << "===========" << endl;


		//namedWindow("Display window", WINDOW_AUTOSIZE);
		//imshow("Display window", mImage);

		//查看颜色对比
		//std::vector<Mat> rgbChannels;
		//cv::split(mImage, rgbChannels);
		///*imshow("blue", rgbChannels[0]);
		//imshow("green", rgbChannels[1]);
		//imshow("red", rgbChannels[2]);*/

		//Mat blue, green, red;
		//SingleMatMerge(&rgbChannels[0], &blue);
		//SingleMatMerge(&rgbChannels[1], &green, 1);
		//SingleMatMerge(&rgbChannels[2], &red, 2);
		//imshow("blue", blue);
		//imshow("green", green);
		//imshow("red", red);

		//查看Lab
		//经过初步观察，lab中的a对car_2应该可以作为区分条件
		//对黑色的车效果不佳，甚至很差，很难区分，应该加一个拉伸来看一下
		//还有也许反色之后处理会有不同
		//Mat mLab;
		//std::vector<Mat> LabChannels;
		//cv::cvtColor(mImage, mLab, cv::COLOR_BGR2Lab);
		//imshow("mLab", mLab);
		//cv::split(mLab, LabChannels);
		////imshow("L", LabChannels[0]);
		////imshow("a", LabChannels[1]);
		////imshow("b", LabChannels[2]);
		//imwrite(sPath + "\\Lab\\Lab-" + vsFileOwnNames[i], mLab);
		//imwrite(sPath + "\\Lab\\L-" + vsFileOwnNames[i], LabChannels[0]);
		//imwrite(sPath + "\\Lab\\a-" + vsFileOwnNames[i], LabChannels[1]);
		//imwrite(sPath + "\\Lab\\b-" + vsFileOwnNames[i], LabChannels[2]);

		//查看HSV，H是颜色
		//第九幅图背景偏黑，然后车子也是黑色，效果很差
		/*Mat mHSV;
		std::vector<Mat> HSVChannels;
		cv::cvtColor(mImage, mHSV, cv::COLOR_BGR2HSV);
		imshow("mHSV", mHSV);
		cv::split(mHSV, HSVChannels);
		imshow("H", HSVChannels[0]);
		imshow("S", HSVChannels[1]);
		imshow("V", HSVChannels[2]);
		imwrite(sPath + "\\HSV\\HSV-" + vsFileOwnNames[i], mHSV);
		imwrite(sPath + "\\HSV\\H-" + vsFileOwnNames[i], HSVChannels[0]);
		imwrite(sPath + "\\HSV\\S-" + vsFileOwnNames[i], HSVChannels[1]);
		imwrite(sPath + "\\HSV\\V-" + vsFileOwnNames[i], HSVChannels[2]);*/

		//查看XYZ图，感觉没有什么明显的帮助
		/*Mat mXYZ;
		string XYZpath = sPath + "\\XYZ";
		std::vector<Mat> XYZChannels;
		cv::cvtColor(mImage, mXYZ, cv::COLOR_BGR2XYZ);
		imshow("mXYZ", mXYZ);
		cv::split(mXYZ, XYZChannels);
		imshow("X", XYZChannels[0]);
		imshow("Y", XYZChannels[1]);
		imshow("Z", XYZChannels[2]);
		MkDir(XYZpath);
		imwrite(sPath + "\\XYZ\\XYZ-" + vsFileOwnNames[i], mXYZ);
		imwrite(sPath + "\\XYZ\\X-" + vsFileOwnNames[i], XYZChannels[0]);
		imwrite(sPath + "\\XYZ\\Y-" + vsFileOwnNames[i], XYZChannels[1]);
		imwrite(sPath + "\\XYZ\\Z-" + vsFileOwnNames[i], XYZChannels[2]);*/

		//Mat mYCrCb;
		//string YCrCbpath = sPath + "\\YCrCb";
		//std::vector<Mat> YCrCbChannels;
		//cv::cvtColor(mImage, mYCrCb, cv::COLOR_BGR2YCrCb);
		//imshow("mYCrCb", mYCrCb);
		//cv::split(mYCrCb, YCrCbChannels);
		////imshow("L", LabChannels[0]);
		////imshow("a", LabChannels[1]);
		////imshow("b", LabChannels[2]);
		//MkDir(YCrCbpath);
		//imwrite(sPath + "\\YCrCb\\YCrCb-" + vsFileOwnNames[i], mYCrCb);
		//imwrite(sPath + "\\YCrCb\\Y-" + vsFileOwnNames[i], YCrCbChannels[0]);
		//imwrite(sPath + "\\YCrCb\\Cr-" + vsFileOwnNames[i], YCrCbChannels[1]);
		//imwrite(sPath + "\\YCrCb\\Cb-" + vsFileOwnNames[i], YCrCbChannels[2]);

		//直接上边界呢？
		/*Mat mCanny;
		string Cannypath = sPath + "\\Canny";
		string Gaussianpath = sPath + "\\Gaussian";
		cvtColor(mImage, mImage2, COLOR_BGR2GRAY);
		cv::GaussianBlur(mImage2, mImage2, cv::Size(3, 3), 1.5, 1.5);
		MkDir(Gaussianpath);
		MkDir(Cannypath);
		imwrite(sPath + "\\Gaussian\\" + vsFileOwnNames[i], mImage2);
		for (uchar uMin = 10; uMin < 250; uMin += 10)
		{
			for (uchar uMax = uMin > 50 ? uMin : 50; uMax < 250; uMax += 10)
			{
				cv::Canny(mImage2, mCanny, uMin, uMax);
				imwrite(sPath + "\\Canny\\" + to_string(uMin) + to_string(uMax) + vsFileOwnNames[i], mCanny);
				cout << "Canny:uMin = " << to_string(uMin) << ", uMax = " << to_string(uMax) << endl;
			}
		}*/


		//Mat img = imread(argv[1]); //读入图像，RGB三通道  
		//imshow("原图像", mImage);
		Mat mRegment; //分割后图像
		int spatialRad = 50;  //空间窗口大小
		int colorRad = 40;   //色彩窗口大小
		int maxPyrLevel = 2;  //金字塔层数
		cout << "do pyrMeanShiftFiltering" << endl;
		pyrMeanShiftFiltering(mImage2, mRegment, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波
		//imshow("res", mRegment);
		RNG rng = theRNG();
		Mat mask(mRegment.rows + 2, mRegment.cols + 2, CV_8UC1, Scalar::all(0));  //掩模
		int colorId(1);
		for (int y = 0; y < mRegment.rows; y++)
		{
			for (int x = 0; x < mRegment.cols; x++)
			{
				if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理
				{
					//Scalar newVal(rng(256), rng(256), rng(256));
					//Scalar newVal(rng(2)*255, rng(2)*255, 125/*rng(256)*/);//分割成只有四种颜色，方便二值化，每个通道只有两种颜色，前景和背景，

					Scalar newVal(rng(4) * 61 + 61, 125, 125);
					/*Scalar newVal(colorId, 125, 125);
					colorId++;
					colorId = colorId == 256 ? 0 : colorId;*/

					floodFill(mRegment, mask, Point(x, y), newVal, 0, Scalar::all(10), Scalar::all(10), /*flags =*/ 8); //执行漫水填充
					//cout << "do floodFill" << endl;
				}
			}
		}
		MkDir(sPath + "\\Pyr");
		imwrite(sPath + "\\Pyr\\" + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + vsFileOwnNames[i], mRegment);
		std::vector<Mat> vmRegment;
		Mat mThresh, mDst = cv::Mat::zeros(mImage.size(), CV_8UC1);
		string sbgr[] = { "B","G","R" };
		cv::split(mRegment, vmRegment);
		vector<vector<Point>> iter_g_contours;
		iter_g_contours.clear();
		for (int k = 0; k < 1/*vmRegment.size()*/; k++)
		{

			for (int m = 0; m < /*256*/4; m++)
			{
				uchar keyVal = (m + 1) * 61;
				Mat mDealGray = Mat::zeros(vmRegment[k].size(), CV_8UC1);
				DelSingleValMat(&vmRegment[k], &mDealGray, keyVal);
				MkDir(sPath + "\\Pyr\\single");
				imwrite(sPath + "\\Pyr\\single\\" + to_string(/*m*/(m + 1) * 61) + "-" + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + vsFileOwnNames[i],
					mDealGray);
				mDst -= mDst;
				//SingleConnect(mDealGray, &mDst, iter_g_contours);
				MkDir(sPath + "\\Pyr\\after");
				imwrite(sPath + "\\Pyr\\after\\" + to_string(keyVal) + to_string(k) + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + vsFileOwnNames[i],
					mDst);

				//检测边缘和进行霍夫圆变换
				Mat mhCanny;
				cv::Canny(mDst, mhCanny, 125, 250);//canny
				vector<Vec3f> circles;
				HoughCircles(mDst/*mhCanny*/, circles, HOUGH_GRADIENT, 1, 30, 100, 10, /*minRadius =*/ 35, /*maxRadius*/90);
				cout << "the number of circles is" << circles.size() << endl;
				//第五个参数 是圆心与圆心之间的距离 
				//第六个参数 就设为默认值就OK
				//第七个参数这个根据你的图像中的圆  大小设置，如果圆越小，则设置越小
				//第八个和第九个参数 是你检测圆 最小半径和最大半径是多少  这个是经验值
				for (size_t ci = 0; ci < circles.size(); ci++)
				{
					Vec3f cc = circles[ci];
					cout << "=" << endl << cc << endl;//查看图像中圆的信息
					circle(mDst, Point(cc[0], cc[1]), cc[2], 125, 2, LINE_AA);//标记出圆
					circle(mDst, Point(cc[0], cc[1]), 2, 125, 2, LINE_AA);//标记出圆心(这里把圆的半径设为2，并把标记线的粗细设为2，刚好画出一个实心的圆心)
				}
				MkDir(sPath + "\\Pyr\\after\\circle");
				imwrite(sPath + "\\Pyr\\after\\circle\\" + to_string((m + 1) * 61) + to_string(k) + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + vsFileOwnNames[i],
					mDst);
			}
		}
		cout << "the" << vsFileOwnNames[i] << "is done;" << endl;


	}

	//waitKey(0);
	std::cout << "Hello World!\n";

	return 0;
}


void testflood()
{
	Mat img = imread("*.jpg"); //读入图像，RGB三通道  
	imshow("原图像", img);
	Mat res; //分割后图像
	int spatialRad = 50;  //空间窗口大小
	int colorRad = 50;   //色彩窗口大小
	int maxPyrLevel = 2;  //金字塔层数
	pyrMeanShiftFiltering(img, res, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波
	imshow("res", res);
	RNG rng = theRNG();
	Mat mask(res.rows + 2, res.cols + 2, CV_8UC1, Scalar::all(0));  //掩模
	for (int y = 0; y < res.rows; y++)
	{
		for (int x = 0; x < res.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				floodFill(res, mask, Point(x, y), newVal, 0, Scalar::all(5), Scalar::all(5)); //执行漫水填充
			}
		}
	}
	imshow("meanShift图像分割", res);
	waitKey();
	//return 0;
}

void testConnectAnalyze(cv::Mat& src)
{
	//Mat src = imread(argv[1]);

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	Mat thr;
	//cvtColor(src, thr, COLOR_BGR2GRAY); //Convert to gray
	//threshold(thr, thr, 125, 255, THRESH_BINARY); //Threshold the gray
	//bitwise_not(thr, thr); //这里先变反转颜色
	thr = src;

	vector<vector<Point> > contours; // Vector for storing contours

	findContours(thr, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image
	//int nMinArea = src.size().width * src.size().width * 0.2;
	contours.erase(std::remove_if(contours.begin(), contours.end(),
		[](const std::vector<cv::Point>& c) {return cv::contourArea(c) < 200; }), contours.end());//去除小的连通域

	for (size_t i = 0; i < contours.size(); i++) // iterate through each contour.
	{
		double area = contourArea(contours[i]);  //  Find the area of contour

		if (area > largest_area)
		{
			largest_area = area;
			largest_contour_index = i;               //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}
	cout << "contours:" << contours.size() << "largest_contour_index" << largest_contour_index << endl;

	drawContours(src, contours, largest_contour_index, Scalar(0, 255, 0), 2); // Draw the largest contour using previously stored index.

	imshow("result", src);
}



//单通道的连通图处理
void SingleConnect(cv::Mat grayImage, cv::Mat* dst)
{
	cout << "goto SingleConnect" << endl;
	Mat thresholdImage;
	//Mat grayImage;
	//cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, thresholdImage, 125, 255, THRESH_BINARY);
	//Mat resultImage;
	//thresholdImage.copyTo(resultImage);
	vector< vector< Point> > contours;  //用于保存所有轮廓信息
	vector< vector< Point> > contours2; //用于保存面积不足100的轮廓
	vector<Point> tempV;				//暂存的轮廓

	findContours(thresholdImage, contours, RETR_CCOMP, CHAIN_APPROX_NONE);
	//cv::Mat labels;
	//int N = connectedComponents(resultImage, labels, 8, CV_16U);
	//findContours(labels, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//轮廓按照面积大小进行升序排序
	sort(contours.begin(), contours.end(), ascendSort);//降序排序
	vector<vector<Point> >::iterator itc = contours.begin();
	int i = 0;
	while (itc != contours.end())
	{
		//获得轮廓的矩形边界
		/*Rect rect = boundingRect(*itc);
		int x = rect.x;
		int y = rect.y;
		int w = rect.width;
		int h = rect.height;*/
		//绘制轮廓的矩形边界
		//cv::rectangle(*dst/*srcImage*/, rect, { 0, 0, 255 }, 1);
		//保存图片
		//char str[10];
		//sprintf(str, "%d.jpg", i++);
		//cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		//waitKey(1000);

		if (itc->size() < 100)
		{
			//把轮廓面积不足100的区域，放到容器contours2中，
			/*tempV.push_back(Point(x, y));
			tempV.push_back(Point(x, y + h));
			tempV.push_back(Point(x + w, y + h));
			tempV.push_back(Point(x + w, y));
			contours2.push_back(tempV);*/
			/*也可以直接用：contours2.push_back(*itc);代替上面的5条语句*/
			//contours2.push_back(*itc);

			//删除轮廓面积不足100的区域，即用黑色填充轮廓面积不足100的区域：
			//cv::drawContours(*dst/*srcImage*/, contours2, -1, Scalar(0, 0, 0), FILLED);

			contours.erase(itc, contours.end());//直接删掉后面的小的
			cout << "delete index = " << i << endl;
			break;
		}

		//保存图片
		//sprintf(str, "%d.jpg", i++);
		//cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		//cv::waitKey(100);
		tempV.clear();
		++i;
		++itc;
	}
	//contours.clear();
	//findContours(*dst/*thresholdImage*/, contours, RETR_CCOMP, CHAIN_APPROX_NONE);
	for (size_t t = 0; t < contours.size(); t++) cv::drawContours(*dst/*srcImage*/, contours, t, 255, FILLED);
	cout << "the new contours's size is:" << contours.size() << endl;
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单



// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件



