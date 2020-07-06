#pragma once
#ifndef ALGO_H
#define ALGO_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>


#include"Struct2d.h"
#include"FileList.h"

//#include"../ransac/ransac_line2d.h"
//#include"../ransac/ransac_circle2d.h"
//#include"../ransac/ransac_ellipse2d.h"


struct stFileName
{
	emImageFileType emType;//文件类型
	string filePath;//文件路径
	string flieName;//文件名,自身名字
	string fileHoleName;//文件名，包含路径和格式
};

struct stRegment
{
	int spatialRad;  //空间窗口大小
	int colorRad;   //色彩窗口大小
	int maxPyrLevel;  //金字塔层数
};

/*
@brief 单通道补足成三通道，其余的两通道补零
@param cv::Mat* InputArray 输入的单通道图像
@param cv::Mat* OutputArray 输出的多通道图像
@param int nChannel 单通道所占据的通道位置

*/
void SingleMatMerge(cv::Mat* InputArray, cv::Mat* OutputArray, int nChannel = 0);
bool ascendSort(std::vector<cv::Point> a, std::vector<cv::Point> b);
void DelSingleValMat(cv::Mat* Src, cv::Mat* Dst, int val); //取出某一个值的点作为前景图
bool isHoriCircleDistPair(stCircle& stC1, stCircle& stC2, int errHight, int minHoriDist, int errR);

class Algo
{
public:
	Algo();
	~Algo();

public:
	void PreProress();//预处理
	void Init();
	void setPath(string sPath);
	void setFileName(string fileHoleName, string fileOwnName);
	void setRegment(stRegment stR) { this->m_Regement = stR; }
	void setFileName();
	void Regment();
	void FindAllCircles();
	void findCieclePair();
	void findCiecleArea();
	//找到所有的圆，可以预见这个函数后期必然添加大量的重载
	void SingleConnect(cv::Mat grayImage, cv::Mat* dst, vector<vector<cv::Point>>& g_contours);//单图像连通区域
	void run();

	

private:
	stFileName m_stFile;   //存储文件名
	Mat m_SrcImage;			//原始读入图像
	Mat m_GuImage;			//经过高斯变换的图像

	stRegment m_Regement;	//均值平移分割参数
	Mat	m_RegImage;			//均值分割后额图像


	vector<vector<stCircle>> m_all_Circles;		//所有前后径图像当中检测出来的圆，方便后面用来进行对比


	int m_mRegs;// = 4;
};

















#endif // !ALGO_H
