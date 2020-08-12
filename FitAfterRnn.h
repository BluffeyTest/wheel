#pragma once
#ifndef FIT_AFTERRNN_H
#define FIT_AFTERRNN_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>


#include"Struct2d.h"
#include"FileList.h"

namespace fitAfterRnn
{
	struct stFileName
	{
		emImageFileType emType;//文件类型
		string filePath;//文件路径
		string flieName;//文件名,自身名字
		string fileHoleName;//文件名，包含路径和格式

		string labelHoleName;//标签文件全名，用来兼容神经网络给的标签图
		string lebelName;//标签文件自己的名字
	};


	class FitAfterRnn
	{
	public:
		FitAfterRnn();
		~FitAfterRnn();


	public:
		void PreProress();//预处理
		void setPath(string sPath);
		void setFileName(string fileHoleName, string fileOwnName);
		void setLabelName(string fileHoleName, string fileOwnName);
		//void setRegment(stRegment stR) { this->m_Regement = stR; }
		void setFileName();
		void Regment();
		void FindAllCircles();
		void findCieclePair();
		void findCiecleArea();
		//找到所有的圆，可以预见这个函数后期必然添加大量的重载
		void SingleConnect(cv::Mat grayImage, cv::Mat *dst, vector<vector<cv::Point>> &g_contours);//单图像连通区域
		void run();

	private:
		void FindLabelPoints();

	private:
		stFileName m_stFile;   //存储文件名
		Mat m_SrcImage;			//原始读入图像
		Mat m_GuImage;			//经过高斯变换的图像
		Mat m_MeanImage;			//经过高斯变换的图像

		Mat m_Labels;			//标签矩阵

		//stRegment m_Regement;	//均值平移分割参数
		Mat	m_RegImage;			//均值分割后额图像

		vector<vector<Point>> vvLabelPts; //标签点集


		vector<vector<stCircle>> m_all_Circles;		//所有前后径图像当中检测出来的圆，方便后面用来进行对比


		int m_mRegs;// = 4;
	};



}







#endif