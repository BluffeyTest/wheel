#include"Ransac.h"

float verifyCircle(cv::Mat dt, cv::Point2f center, float radius, std::vector<cv::Point2f>& inlierSet)
{
    unsigned int counter = 0;
    unsigned int inlier = 0;
    float minInlierDist = 2.0f;
    float maxInlierDistMax = 100.0f;
    float maxInlierDist = radius / 25.0f;
    if (maxInlierDist < minInlierDist) maxInlierDist = minInlierDist;
    if (maxInlierDist > maxInlierDistMax) maxInlierDist = maxInlierDistMax;

    // choose samples along the circle and count inlier percentage
    for (float t = 0; t < 2 * CV_PI; t += 0.05f)
    {
        counter++;
        float cX = radius * cos(t) + center.x;
        float cY = radius * sin(t) + center.y;

        if (cX < dt.cols)
            if (cX >= 0)
                if (cY < dt.rows)
                    if (cY >= 0)
                        if (dt.at<float>(cY, cX) < maxInlierDist)//从这儿看出来，这种情况弄的是一个饼，而不是一个圆圈，我需要的是一个圆圈，而不是一个饼
                        {
                            inlier++;
                            inlierSet.push_back(cv::Point2f(cX, cY));
                        }
    }

    return (float)inlier / float(counter);
}


inline void getCircle(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3, cv::Point2f& center, float& radius)
{
    float x1 = p1.x;
    float x2 = p2.x;
    float x3 = p3.x;

    float y1 = p1.y;
    float y2 = p2.y;
    float y3 = p3.y;

    // PLEASE CHECK FOR TYPOS IN THE FORMULA :)
    center.x = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2);
    center.x /= (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

    center.y = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1);
    center.y /= (2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

    radius = sqrt((center.x - x1) * (center.x - x1) + (center.y - y1) * (center.y - y1));
}



std::vector<cv::Point2f> getPointPositions(cv::Mat binaryImage)
{
    std::vector<cv::Point2f> pointPositions;

    for (unsigned int y = 0; y < binaryImage.rows; ++y)
    {
        //unsigned char* rowPtr = binaryImage.ptr<unsigned char>(y);
        for (unsigned int x = 0; x < binaryImage.cols; ++x)
        {
            //if(rowPtr[x] > 0) pointPositions.push_back(cv::Point2i(x,y));
            if (binaryImage.at<unsigned char>(y, x) > 0) pointPositions.push_back(cv::Point2f(x, y));
        }
    }

    return pointPositions;
}



int ransacmain()
{
    clock_t starttime, endtime;
    starttime = clock();
    cv::Mat color = cv::imread("1.jpg");
    cv::Mat gray;

    // convert to grayscale
    // you could load as grayscale if you want, but I used it for (colored) output too
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);


    cv::Mat mask;

    float canny1 = 100;
    float canny2 = 20;

    cv::Mat canny;
    cv::Canny(gray, canny, canny1, canny2);
    //cv::imshow("canny",canny);

    mask = canny;



    std::vector<cv::Point2f> edgePositions;
    edgePositions = getPointPositions(mask);

    // create distance transform to efficiently evaluate distance to nearest edge
    cv::Mat dt;
    cv::distanceTransform(255 - mask, dt, cv::DIST_L1/*CV_DIST_L1*/, 3);

    //TODO: maybe seed random variable for real random numbers.

    unsigned int nIterations = 0;

    cv::Point2f bestCircleCenter;
    float bestCircleRadius;
    float bestCirclePercentage = 0;
    float minRadius = 10;   // TODO: ADJUST THIS PARAMETER TO YOUR NEEDS, otherwise smaller circles wont be detected or "small noise circles" will have a high percentage of completion

    //float minCirclePercentage = 0.2f;
    float minCirclePercentage = 0.05f;  // at least 5% of a circle must be present? maybe more...

    int maxNrOfIterations = edgePositions.size();   // TODO: adjust this parameter or include some real ransac criteria with inlier/outlier percentages to decide when to stop
    printf("%d\n", maxNrOfIterations);
    for (unsigned int its = 0; its < maxNrOfIterations; ++its)
    {
        //RANSAC: randomly choose 3 point and create a circle:
        //TODO: choose randomly but more intelligent, 
        //so that it is more likely to choose three points of a circle. 
        //For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
        unsigned int idx1 = rand() % edgePositions.size();
        unsigned int idx2 = rand() % edgePositions.size();
        unsigned int idx3 = rand() % edgePositions.size();

        // we need 3 different samples:
        if (idx1 == idx2) continue;
        if (idx1 == idx3) continue;
        if (idx3 == idx2) continue;

        // create circle from 3 points:
        cv::Point2f center; float radius;
        getCircle(edgePositions[idx1], edgePositions[idx2], edgePositions[idx3], center, radius);

        // inlier set unused at the moment but could be used to approximate a (more robust) circle from alle inlier
        std::vector<cv::Point2f> inlierSet;

        //verify or falsify the circle by inlier counting:
        //这儿只是把在该圆心的的一定距离内的店筛选出来了，但是实际上这个点完全可能在是在园上而不是圆内，所以接下来的代码其实不一定奏效
        float cPerc = verifyCircle(dt, center, radius, inlierSet);

        // update best circle information if necessary
        if (cPerc >= bestCirclePercentage)
            if (radius >= minRadius)
            {
                bestCirclePercentage = cPerc;
                bestCircleRadius = radius;
                bestCircleCenter = center;
            }

    }

    std::cout << "bestCirclePerc: " << bestCirclePercentage << std::endl;
    std::cout << "bestCircleRadius: " << bestCircleRadius << std::endl;

    // draw if good circle was found
    if (bestCirclePercentage >= minCirclePercentage)
        if (bestCircleRadius >= minRadius);
    cv::circle(color, bestCircleCenter, bestCircleRadius, cv::Scalar(255, 255, 0), 1);
    std::cout << "the used time is: " << clock() - starttime << std::endl;

    cv::imshow("output", color);
    cv::imshow("mask", mask);
    //cv::imwrite("../outputData/1_circle_normalized.png", normalized);
    cv::waitKey(0);

    return 0;
}



