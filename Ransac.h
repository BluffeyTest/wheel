#pragma once

#ifndef RANSAC_H
#define RANSAC_H
#include <opencv2/opencv.hpp>
#include <ctime>

std::vector<cv::Point2f> getPointPositions(cv::Mat binaryImage);
float verifyCircle(cv::Mat dt, cv::Point2f center, float radius, std::vector<cv::Point2f>& inlierSet);
inline void getCircle(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3, cv::Point2f& center, float& radius);

#endif // !RANSAC_H

