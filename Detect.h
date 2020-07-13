#pragma once
#ifndef DETECT_H
#define DETECT_H
#include"Defines.h"
#include"Struct2d.h"
#include"Ransac.h"

/**
 * Detect��ļ�����������һ����ֻҪ������ݶ�������װ.
 */
struct stDetect
{
	int k;
};

/**
 * �����Ľ����Ҳ��һ���򣬼����������������װ.
 */
struct stDetectResulr
{

};

/**
 * ����࣬����ʵ�ֶԵ���ͼ��ļ��.
 */
class Detect
{
public:
	Detect();
	~Detect();

private:
	vector<Mat*> m_pTemps;//���������ģ��ͼ��
	Mat* m_pSrc;//�����ԭʼͼ���ָ��

	stDetect* m_pPara;//������
	stDetectResulr* m_pResult;//�����

	vector<Mat*> m_pMatchScore;//ģ��ƥ������۾���

	Mat** m_ppSegment;//�ָ���ͼ��
	
	//roi�м���
	Mat* m_pRoiGray;
	Mat* m_pRoiBinary;
	Mat* m_psobelX;
	Mat* m_psobelY;
	Mat* m_pCanny;

	Mat* m_pMask;//��������

public:
	bool InputTemps(vector<Mat*> *m_pTemps);//����ģ��
	bool InputImage(Mat* Src);//����ԭͼ��
	bool SetPara(stDetect* stD);//�������
	bool GetResult(stDetectResulr* stDr);//��ȡ���

private:




};



























#endif // !DETECT_H

