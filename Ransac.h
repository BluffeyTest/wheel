#pragma once

#ifndef RANSAC_H
#define RANSAC_H
#include <opencv2/opencv.hpp>
#include <ctime>

#include"Struct2d.h"

using namespace cv;

/**
 * Ransac ����.
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
	emFitType type;//�������
	double dInner;//�ڵ����
	double dScale;//��С��ϵ�����������Ϊ0����ȡ��󣬷���һ��ȡ����0.5
	int nIters;//����������
}RansacPara;

class Ransac
{
public:
	Ransac();
	~Ransac();

private:
	RansacPara m_Para;//����

	std::vector<Point> m_vec_SrcPoints;///ԭʼ�㼯
	std::vector<Point> m_vec_Points;///ʵ�ʴ�����̵��еĵ㼯

	std::vector<stCircle> m_vec_Circles;///��������Բ
	std::vector<stSegLine> m_vec_SegLines;//���������߶�

	stCircle m_Circle;//���ص���Բ
	stSegLine m_SegLine;//���صĵ����߶�

	stCircle m_CurrentCircle;//���ڼ��ĵ���Բ
	stSegLine m_CurrentSegLine;//���ڼ��ĵ����߶�

	//bool m_bVecCircle; //�Ƿ񷵻ض��Բ
	//bool m_bVecSegLine;//�Ƿ񷵻ض���߶�

public:

	bool InputPoints(std::vector<Point> &vec_pts);//����㼯
	bool InputPara(RansacPara& stR);//�������
	bool Run();//����
	bool FitSegLine();//���ֱ��
	bool FitSegLines();//��϶���ֱ�ߣ�
	bool FitCircle();//���Բ��
	bool FitCircles();//��϶��Բ

	bool GetResult();//���д�����ã���������һ
};














#endif // !RANSAC_H