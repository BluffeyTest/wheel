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
		emImageFileType emType;//�ļ�����
		string filePath;//�ļ�·��
		string flieName;//�ļ���,��������
		string fileHoleName;//�ļ���������·���͸�ʽ

		string labelHoleName;//��ǩ�ļ�ȫ��������������������ı�ǩͼ
		string lebelName;//��ǩ�ļ��Լ�������
	};


	class FitAfterRnn
	{
	public:
		FitAfterRnn();
		~FitAfterRnn();


	public:
		void PreProress();//Ԥ����
		void setPath(string sPath);
		void setFileName(string fileHoleName, string fileOwnName);
		void setLabelName(string fileHoleName, string fileOwnName);
		//void setRegment(stRegment stR) { this->m_Regement = stR; }
		void setFileName();
		void Regment();
		void FindAllCircles();
		void findCieclePair();
		void findCiecleArea();
		//�ҵ����е�Բ������Ԥ������������ڱ�Ȼ��Ӵ���������
		void SingleConnect(cv::Mat grayImage, cv::Mat *dst, vector<vector<cv::Point>> &g_contours);//��ͼ����ͨ����
		void run();

	private:
		void FindLabelPoints();

	private:
		stFileName m_stFile;   //�洢�ļ���
		Mat m_SrcImage;			//ԭʼ����ͼ��
		Mat m_GuImage;			//������˹�任��ͼ��
		Mat m_MeanImage;			//������˹�任��ͼ��

		Mat m_Labels;			//��ǩ����

		//stRegment m_Regement;	//��ֵƽ�Ʒָ����
		Mat	m_RegImage;			//��ֵ�ָ���ͼ��

		vector<vector<Point>> vvLabelPts; //��ǩ�㼯


		vector<vector<stCircle>> m_all_Circles;		//����ǰ��ͼ���м�������Բ����������������жԱ�


		int m_mRegs;// = 4;
	};



}







#endif