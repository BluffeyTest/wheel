
#include "Algo.h"


using namespace std;
using namespace cv;

bool ascendSort(std::vector<cv::Point> a, std::vector<cv::Point> b) {
	return a.size() > b.size();
}

void SingleMatMerge(cv::Mat* InputArray, cv::Mat* OutputArray, int nChannel/* = 0*/)
{
	std::vector<cv::Mat> vChannels;
	for (uchar i = 0; i < 3; i++)
	{
		if (i == nChannel)vChannels.push_back(*InputArray);
		else vChannels.push_back(cv::Mat::zeros(InputArray->size(), CV_8UC1));
	}
	cv::merge(vChannels, *OutputArray);
}

void PreAnalyze() {};

Algo::Algo()
{
}

Algo::~Algo()
{
}

void Algo::PreProress()
{
	this->m_SrcImage = imread(this->m_stFile.fileHoleName);
	cv::GaussianBlur(this->m_SrcImage, this->m_GuImage, cv::Size(3, 3), 1.5, 1.5);
}

void Algo::setPath(string sPath)
{
	this->m_stFile.filePath = sPath;
}

void Algo::setFileName(string fileHoleName, string fileOwnName)
{
	this->m_stFile.fileHoleName = fileHoleName;
	this->m_stFile.flieName = fileOwnName;
}

void Algo::setFileName()
{


}

void Algo::Regment()
{
	int spatialRad = this->m_Regement.spatialRad;
	int colorRad = this->m_Regement.colorRad;
	int maxPyrLevel = this->m_Regement.maxPyrLevel;
	cout << "do pyrMeanShiftFiltering" << endl;
	pyrMeanShiftFiltering(m_GuImage, m_RegImage, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波
	//imshow("res", mRegment);
	RNG rng = theRNG();
	Mat mask(m_RegImage.rows + 2, m_RegImage.cols + 2, CV_8UC1, Scalar::all(0));  //掩模
	int colorId(1);
	for (int y = 0; y < m_RegImage.rows; y++)
	{
		for (int x = 0; x < m_RegImage.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理
			{
				//Scalar newVal(rng(256), rng(256), rng(256));
				//Scalar newVal(rng(2)*255, rng(2)*255, 125/*rng(256)*/);//分割成只有四种颜色，方便二值化，每个通道只有两种颜色，前景和背景，

				Scalar newVal(rng(4) * 61 + 61, 125, 125);
				/*Scalar newVal(colorId, 125, 125);
				colorId++;
				colorId = colorId == 256 ? 0 : colorId;*/

				floodFill(m_RegImage, mask, Point(x, y), newVal, 0, Scalar::all(10), Scalar::all(10), /*flags =*/ 8); //执行漫水填充
				//cout << "do floodFill" << endl;
			}
		}
	}
	string sPath = this->m_stFile.filePath;
	string sOwnName = this->m_stFile.flieName;
	MkDir(sPath + "\\Pyr");
	imwrite(sPath + "\\Pyr\\" + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + sOwnName, m_RegImage);
	cout << "do pyrMeanShiftFiltering done" << endl;
}

void Algo::FindAllCircles()
{
#ifdef _DEBUG
	cout << "FindAllCircles" << endl;
#endif // _DEBUG

	m_all_Circles.clear();//清除所有的圆的信息


	string sPath = this->m_stFile.filePath;
	string sOwnName = this->m_stFile.flieName;
	int spatialRad = this->m_Regement.spatialRad;
	int colorRad = this->m_Regement.colorRad;
	int maxPyrLevel = this->m_Regement.maxPyrLevel;

	std::vector<Mat> vmRegment;
	Mat mThresh, mDst = cv::Mat::zeros(this->m_SrcImage.size(), CV_8UC1);
	string sbgr[] = { "B","G","R" };
	cv::split(m_RegImage, vmRegment);
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
			imwrite(sPath + "\\Pyr\\single\\" + to_string(keyVal) + "-" + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + sOwnName,
				mDealGray);
			mDst -= mDst;
			SingleConnect(mDealGray, &mDst, iter_g_contours);
			MkDir(sPath + "\\Pyr\\after");
			imwrite(sPath + "\\Pyr\\after\\" + to_string(keyVal) + "-" + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + sOwnName,
				mDst);

			//检测边缘和进行霍夫圆变换
			Mat mhCanny;
			cv::Canny(mDst, mhCanny, 125, 250);//canny
			vector<Vec3f> circles;
			HoughCircles(mDst/*mhCanny*/, circles, HOUGH_GRADIENT, 1, 20, 100, 10, /*minRadius =*/ 35, /*maxRadius*/100);
			cout << "the number of circles is" << circles.size() << endl;
			//第五个参数 是圆心与圆心之间的距离 
			//第六个参数 就设为默认值就OK
			//第七个参数这个根据你的图像中的圆  大小设置，如果圆越小，则设置越小
			//第八个和第九个参数 是你检测圆 最小半径和最大半径是多少  这个是经验值
			vector<stCircle> vCircles;
			for (size_t ci = 0; ci < circles.size(); ci++)
			{
				Vec3f cc = circles[ci];
				vCircles.push_back(stCircle(Point(cc[0], cc[1]), cc[2]));
				//cout << "=" << endl << cc << endl;//查看图像中圆的信息
				circle(mDst, Point(cc[0], cc[1]), cc[2], 125, 2, LINE_AA);//标记出圆
				circle(mDst, Point(cc[0], cc[1]), 2, 125, 2, LINE_AA);//标记出圆心(这里把圆的半径设为2，并把标记线的粗细设为2，刚好画出一个实心的圆心)
			}
			m_all_Circles.push_back(vCircles);
			vCircles.clear();
			MkDir(sPath + "\\Pyr\\after\\circle");
			imwrite(sPath + "\\Pyr\\after\\circle\\" + to_string((m + 1) * 61) + to_string(spatialRad) + to_string(colorRad) + to_string(maxPyrLevel) + sOwnName,
				mDst);
		}
	}

#ifdef _DEBUG
	cout << "FindAllCircles End" << endl;
#endif // _DEBUG
}

//寻找圆对
//目前主要采集，基本在一条水平线上的，圆心间距大于三倍直径的（起码也要两倍），、、后面可以改成是圆心连线平行于车身轮廓
void Algo::findCieclePair()
{
	vector<vector<stCircle>>& vvCircles = this->m_all_Circles;
	vector<pair<stCircle, stCircle>> vpCCPair;
	vpCCPair.clear();
	int errHight(10);//高度差异范围
	int errR(20);//半径差异
	double dScale(3);//倍数
	for (size_t i = 0; i < vvCircles.size();i++)
	{
		for(size_t j=0;j< vvCircles[i].size();j++)
		{
			for (size_t k = 0;k< vvCircles.size();k++)
			{
				for (size_t m = 0; m < vvCircles[k].size(); m++)
				{
					if (i == k && j == m)continue;
					stCircle stC1 = vvCircles[i][j];
					stCircle stC2 = vvCircles[k][m];
					int minHoriDist = dScale * (stC1.dR + stC2.dR);
					if(isHoriCircleDistPair(stC1, stC2, errHight, minHoriDist, errR) )
						vpCCPair.push_back(pair<stCircle, stCircle>(stC1, stC2));
				}
			}
		}
	}




}
//判断是否是临近的圆，是否需要合并
bool isSameCircle(stCircle& stC1, stCircle& stC2, int errCenter, int errR)
{
	if (abs(stC1.ptCenter.x - stC2.ptCenter.x) <= errCenter
		&& abs(stC1.ptCenter.y - stC2.ptCenter.y) <= errCenter
		&& abs(stC1.dR - stC2.dR) <= errR
		)
		return true;
	return false;
}

//水平上面相聚一定距离的圆
bool isHoriCircleDistPair(stCircle& stC1, stCircle& stC2, int errHight, int minHoriDist, int errR)
{
	if (abs(stC1.ptCenter.x - stC2.ptCenter.x) >= minHoriDist//水平距离超出最小值
		&& abs(stC1.ptCenter.y - stC2.ptCenter.y) <= errHight //高度在误差内
		&& abs(stC1.dR - stC2.dR) <= errR //半径差距不太大
		)
		return true;
	return false;
}






//单通道的连通图处理
void Algo::SingleConnect(cv::Mat grayImage, cv::Mat* dst, vector<vector<cv::Point>>& g_contours)
{
	cout << "goto SingleConnect" << endl;
	Mat thresholdImage;
	threshold(grayImage, thresholdImage, 1, 255, THRESH_BINARY);
	vector< vector< Point> > contours;  //用于保存所有轮廓信息
	//vector< vector< Point> > contours2; //用于保存面积不足100的轮廓
	vector<Point> tempV;				//暂存的轮廓

	findContours(thresholdImage, contours, RETR_CCOMP, CHAIN_APPROX_NONE);

	//轮廓按照面积大小进行升序排序
	sort(contours.begin(), contours.end(), ascendSort);//降序排序
	vector<vector<Point> >::iterator itc = contours.begin();
	int i = 0;
	while (itc != contours.end())
	{

		if (itc->size() < 100)
		{
			contours.erase(itc, contours.end());//直接删掉后面的小的
			cout << "delete index = " << i << endl;
			break;
		}

		tempV.clear();
		++i;
		++itc;
	}
	g_contours.insert(g_contours.end(), contours.begin(), contours.end());
	for (size_t t = 0; t < contours.size(); t++) cv::drawContours(*dst/*srcImage*/, contours, t, 255, FILLED);
	cout << "the new contours's size is:" << contours.size() << endl;
}

void Algo::run()
{
	PreProress();//读入和滤波
	Regment();//分割
	FindAllCircles();//找到分割后图里面的所有的圆
	//checkCircles();//检查符合要求的圆
	//提取所得圆的roi并进行预处理，预处理是指进行新的分割和二值化。

}



void DelSingleValMat(cv::Mat* Src, cv::Mat* Dst, int val)
{
	for (size_t i = 0; i < Src->rows; i++)
	{
		for (size_t j = 0; j < Src->cols; j++)
		{
			if (Src->at<uchar>(i, j) == val)
			{
				Dst->at<uchar>(i, j) = Src->at<uchar>(i, j);
			}
			else
			{
				Dst->at<uchar>(i, j) = 0;
			}

		}
	}
}
