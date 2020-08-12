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
	emImageFileType emType;//�ļ�����
	string filePath;//�ļ�·��
	string flieName;//�ļ���,��������
	string fileHoleName;//�ļ���������·���͸�ʽ

	string labelHoleName;//��ǩ�ļ�ȫ��������������������ı�ǩͼ
	string lebelName;//��ǩ�ļ��Լ�������
};

struct stRegment
{
	int spatialRad;  //�ռ䴰�ڴ�С
	int colorRad;   //ɫ�ʴ��ڴ�С
	int maxPyrLevel;  //����������
};

/*
@brief ��ͨ���������ͨ�����������ͨ������
@param cv::Mat* InputArray ����ĵ�ͨ��ͼ��
@param cv::Mat* OutputArray ����Ķ�ͨ��ͼ��
@param int nChannel ��ͨ����ռ�ݵ�ͨ��λ��

*/
void SingleMatMerge(cv::Mat* InputArray, cv::Mat* OutputArray, int nChannel = 0);
bool ascendSort(std::vector<cv::Point> a, std::vector<cv::Point> b);
void DelSingleValMat(cv::Mat* Src, cv::Mat* Dst, int val); //ȡ��ĳһ��ֵ�ĵ���Ϊǰ��ͼ
bool isHoriCircleDistPair(stCircle& stC1, stCircle& stC2, int errHight, int minHoriDist, int errR);

class Algo
{
public:
	Algo();
	~Algo();

public:
	void PreProress();//Ԥ����
	//void Init();
	void setPath(string sPath);
	void setFileName(string fileHoleName, string fileOwnName);
	void setRegment(stRegment stR) { this->m_Regement = stR; }
	void setFileName();
	void Regment();
	void FindAllCircles();
	void findCieclePair();
	void findCiecleArea();
	//�ҵ����е�Բ������Ԥ������������ڱ�Ȼ��Ӵ���������
	void SingleConnect(cv::Mat grayImage, cv::Mat* dst, vector<vector<cv::Point>>& g_contours);//��ͼ����ͨ����
	void run();



	

private:
	stFileName m_stFile;   //�洢�ļ���
	Mat m_SrcImage;			//ԭʼ����ͼ��
	Mat m_GuImage;			//������˹�任��ͼ��
	Mat m_MeanImage;			//������˹�任��ͼ��

	stRegment m_Regement;	//��ֵƽ�Ʒָ����
	Mat	m_RegImage;			//��ֵ�ָ���ͼ��

	

	vector<vector<stCircle>> m_all_Circles;		//����ǰ��ͼ���м�������Բ����������������жԱ�


	int m_mRegs;// = 4;
};

/**
 * .
 */
class AlgoAfterRnn
{
public:
	AlgoAfterRnn();
	~AlgoAfterRnn();


public:
	void PreProress();//Ԥ����
	void setPath(string sPath);
	void setFileName(string fileHoleName, string fileOwnName);
	void setLabelName(string fileHoleName, string fileOwnName);
	void setRegment(stRegment stR) { this->m_Regement = stR; }
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

	stRegment m_Regement;	//��ֵƽ�Ʒָ����
	Mat	m_RegImage;			//��ֵ�ָ���ͼ��

	vector<vector<Point>> vvLabelPts; //��ǩ�㼯


	vector<vector<stCircle>> m_all_Circles;		//����ǰ��ͼ���м�������Բ����������������жԱ�


	int m_mRegs;// = 4;
};

















#endif // !ALGO_H
