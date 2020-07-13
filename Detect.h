#pragma once
#ifndef DETECT_H
#define DETECT_H
#include"Defines.h"
#include"Struct2d.h"
#include"Ransac.h"

/**
 * Detect类的检测参数，就是一个框，只要监测数据都往里面装.
 */
struct stDetect
{
	int k;
};

/**
 * 检测类的结果，也是一个框，检测结果都可以往里面装.
 */
struct stDetectResulr
{

};

/**
 * 检测类，用于实现对单个图像的检测.
 */
class Detect
{
public:
	Detect();
	~Detect();

private:
	vector<Mat*> m_pTemps;//传入的所有模板图像
	Mat* m_pSrc;//传入的原始图像的指针

	stDetect* m_pPara;//检测参数
	stDetectResulr* m_pResult;//检测结果

	vector<Mat*> m_pMatchScore;//模板匹配的评价矩阵

	Mat** m_ppSegment;//分割后的图像
	
	//roi中间结果
	Mat* m_pRoiGray;
	Mat* m_pRoiBinary;
	Mat* m_psobelX;
	Mat* m_psobelY;
	Mat* m_pCanny;

	Mat* m_pMask;//屏蔽区域

public:
	bool InputTemps(vector<Mat*> *m_pTemps);//传入模板
	bool InputImage(Mat* Src);//传入原图像
	bool SetPara(stDetect* stD);//传入参数
	bool GetResult(stDetectResulr* stDr);//提取结果

private:




};



























#endif // !DETECT_H

