#pragma once

#ifndef WHEEL_H
#define WHEEL_H
#include"Algo.h"
#include"Struct2d.h"
#include"Ransac.h"
bool IsBiger(float *a, float *b);
bool cvSameSzMatchs(std::vector<cv::Mat> &Temps, string Filename);
void SingleConnect(cv::Mat grayImage, cv::Mat* dst);
//void SingleConnect(cv::Mat grayImage, cv::Mat* dst, vector<vector<cv::Point>> &g_contours);
void testflood();
//void DelSingleValMat(cv::Mat *Src, cv::Mat *Dst, int val);
bool testCircle2dMulti(cv::Mat* BinarySrc, std::vector<stCircle>* vCircles);
bool cvMatch(cv::Mat& Temp, string Filename);
bool cvMatch(cv::Mat& Temp, string Filename, string ownname);
void getTemps(string TempPath, std::vector<cv::Mat>& Temps);

bool FindCircleMain(Mat& Src, Mat& Dst, std::vector<Point>& vec_Pts, std::vector<stCircle>& vec_Circle, cv::Size sz);
bool FindROICircle(Mat& Src, Mat& Dst, std::vector<Point>& vec_Centers, std::vector<stCircle>& vecCircle, cv::Size sz);
bool DrawPoint(Mat& Img, Point& pt, Scalar color, char flag);
bool drawCrossPoint(Mat& Img, Point pt, Scalar color = Scalar(0,0,255), int halfSize = 2);
bool GetPointsFromCanny(Mat& Canny, std::vector<Point>& vec_Points);
bool GetPointsFromBinary(Mat& Binary, std::vector<Point>& vec_Points);
bool drawPointsRoi(Mat& Src, Mat& Dst, std::vector<Point>& vec_Centers);
bool FindPointsRoi(std::vector<Point>& vec_Pts, int minDis, int maxDis, std::vector<Point>& vec_Centers, std::vector<cv::Size>& vec_Size, int& ks);
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

