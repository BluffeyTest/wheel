#pragma once
#ifndef STRUCT2D_H
#define STRUCT2D_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

using namespace std;
using namespace cv;
struct stCircle
{
	Point ptCenter;
	double dR;
	stCircle():ptCenter(Point(0,0)),dR(0.0){}
	stCircle(Point pt,double r):ptCenter(pt),dR(r){}

	inline bool operator==(const stCircle &stC) const;
	
};
struct stArc
{
	stCircle Circle;
	double dStartAngle;//��ʼ�Ƕȣ�������
	double dEndAngle;	//�����ǶȻ�����
};








































#endif // !STRUCT2D_H
