#pragma once

#ifndef WHEEL_H
#define WHEEL_H
#include"Algo.h"
#include"Struct2d.h"
bool IsBiger(float *a, float *b);

void SingleConnect(cv::Mat grayImage, cv::Mat* dst);
//void SingleConnect(cv::Mat grayImage, cv::Mat* dst, vector<vector<cv::Point>> &g_contours);
void testflood();
//void DelSingleValMat(cv::Mat *Src, cv::Mat *Dst, int val);
bool testCircle2dMulti(cv::Mat* BinarySrc, std::vector<stCircle>* vCircles);
bool cvMatch(cv::Mat& Temp, string Filename);
bool cvMatch(cv::Mat& Temp, string Filename, string ownname);
void getTemps(string TempPath, std::vector<cv::Mat>& Temps);
bool cvSameSzMatchs(std::vector<cv::Mat>& Temps, string Filename);
bool FindTen(Mat& Src, Mat& DataMat, std::vector<Point>& vec_Pts, int matchMethod);
bool slidSumMean(cv::Mat& Src, Mat& Dst, cv::Size sz);
void findBestPair(cv::Point& p_first_best, cv::Point& p_second_best, cv::Size& sz_first_best, cv::Size& sz_second_best, std::vector<std::pair<double, double>> vpMax, std::vector<std::pair<Point, Point>> vpPoint, std::vector<cv::Size> vDstSize, int matchMethod);
bool cvMatchs(std::vector<cv::Mat>& Temps, string Filename);
//bool addDstImgAndFindBox(std::vector<cv::Mat>& vec_dstImg, std::vector<cv::Size>& vec_dstSize, std::vector<Point> vptDst);
bool addDstImgAndFindBox(Mat &Src, std::vector<cv::Mat>& vec_dstImg, std::vector<cv::Size>& vec_dstSize,cv::Size szMean, int matchMethod);
void Image_fileter(Mat resImg, Mat& output_img);
bool MatExternRB(cv::Mat& Src, cv::Mat& Dst, float val);
bool findDoubleMin(cv::Mat& src, double& dMin, cv::Point& ptMin, cv::Mat& mask);
bool findDoubleMax(cv::Mat& src, double &dMax, cv::Point &ptMax, cv::Mat &mask);
void roiToMask(cv::Mat& Src, Point p1, Point p2);
void roiToVal(cv::Mat& Src, Point p1, Point p2, int val);




















#endif // !WHEEL_H
