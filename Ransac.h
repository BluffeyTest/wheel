#pragma once

#ifndef RANSAC_H
#define RANSAC_H
#include <opencv2/opencv.hpp>
#include <ctime>

#include"Struct2d.h"

using namespace cv;

/**
 * Ransac 参数.
 */
typedef struct stRansacPara
{
	enum emFitType
	{
		RASANC_SEG_LINE = 0,
		RASANC_SEG_LINES = 1,
		RASANC_SEG_CIRCLE = 2,
		RASANC_SEG_CIRCLES = 3,
	};
	emFitType type;//拟合类型
	double dInner;//内点距离
	double dScale;//最小拟合点数比例，若为0，则取最大，否则一般取至少0.5
	int nIters;//最大迭代次数
}RansacPara;

class Ransac
{
public:
	Ransac();
	~Ransac();

private:
	RansacPara m_Para;//参数

	std::vector<Point> m_vec_SrcPoints;///原始点集
	std::vector<Point> m_vec_Points;///实际处理过程当中的点集

	std::vector<stCircle> m_vec_Circles;///检测出来的圆
	std::vector<stSegLine> m_vec_SegLines;//检测出来的线段

	stCircle m_Circle;//返回单个圆
	stSegLine m_SegLine;//返回的单个线段

	stCircle m_CurrentCircle;//正在检测的单个圆
	stSegLine m_CurrentSegLine;//正在检测的单个线段

	//bool m_bVecCircle; //是否返回多个圆
	//bool m_bVecSegLine;//是否返回多个线段

public:

	bool InputPoints(std::vector<Point> &vec_pts);//传入点集
	bool InputPara(RansacPara& stR);//传入参数
	bool Run();//运行
	bool FitSegLine();//拟合直线
	bool FitSegLines();//拟合多条直线；
	bool FitCircle();//拟合圆；
	bool FitCircles();//拟合多个圆

	bool GetResult();//这个写法不好，传出不单一
};














#endif // !RANSAC_H