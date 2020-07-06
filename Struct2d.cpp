#include"Struct2d.h"

stCircle::stCircle(Point& pt1, Point& pt2, Point& pt3)
{
    double x1 = pt1.x, x2 = pt2.x, x3 = pt3.x;
    double y1 = pt1.y, y2 = pt2.y, y3 = pt3.y;
    double a = x1 - x2;
    double b = y1 - y2;
    double c = x1 - x3;
    double d = y1 - y3;
    double e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0;
    double f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0;
    double det = b * c - a * d;
    
    //三点共线
    if (fabs(det) < 1e-5)
    {
        dR = -1;
        ptCenter = Point(0, 0);
    }

    double x0 = -(d * e - b * f) / det;
    double y0 = -(a * f - c * e) / det;
    dR = hypot(x1 - x0, y1 - y0);
    ptCenter = Point(x0, y0);
}

inline bool stCircle::operator==(const stCircle &stC) const
{
    if (this->ptCenter.x == stC.ptCenter.x
        && this->ptCenter.y == stC.ptCenter.y
        && this->dR == stC.dR)
        return true;
    return false;
}

inline bool stCircle::Cross(stCircle& stC) const
{
    double dDist = sqrt(pow(this->ptCenter.x - stC.ptCenter.x, 2) + pow(this->ptCenter.y - stC.ptCenter.y, 2));
    if (dDist == (this->dR + stC.dR))
        return true;
    return false;
}

//圆与直线相交
inline bool stCircle::Cross(stGenLine& stG) const
{
    double dDistance = stG.FromPoint(this->ptCenter);
    if (this->dR < dDistance)
        return true;
    return false;
}

//待优化
inline bool stCircle::Cross(stSegLine& stS) const
{
    stGenLine stG = stGenLine(stS);
    double d1 = sqrt(pow(stS.pt1.x - this->ptCenter.x, 2) + pow(stS.pt1.y - this->ptCenter.y, 2));///<线段第一个点到圆心的距离
    double d2 = sqrt(pow(stS.pt2.x - this->ptCenter.x, 2) + pow(stS.pt2.y - this->ptCenter.y, 2));///<线段第二个点到圆心的距离
    double d3 = stG.FromPoint(this->ptCenter);
    if (d1 < this->dR
        || d2 < this->dR)
        return true;
    if()
    return false;
}

stGenLine::stGenLine(Point &p1, Point &p2)
{
    this->da =(double) (p1.y - p2.y);
    this->db = (double)(p2.x - p1.x);
    this->dc = (double)(p1.x*p2.y - p2.x*p1.y);
}

stGenLine::stGenLine(stSegLine& stS)
{
    Point &p1 = stS.pt1;
    Point &p2 = stS.pt2;
    this->da = (double)(p1.y - p2.y);
    this->db = (double)(p2.x - p1.x);
    this->dc = (double)(p1.x * p2.y - p2.x * p1.y);
}

inline double stGenLine::FromPoint(Point &pt) const
{  
    double dDistance = fabs(this->da * pt.x + this->db * pt.y + this->dc) / sqrt(pow(this->da, 2) + pow(this->db, 2));
    return dDistance;
}

inline bool stGenLine::operator==(const stGenLine& stG) const
{
    double dScale = this->da / stG.da;
    if (this->da == stG.da * dScale
        && this->db == stG.db * dScale
        && this->dc == stG.dc * dScale
        )
        return true;
    return false;
}

inline bool stSegLine::operator==(stSegLine& stS) const
{
    if (this->pt1 == stS.pt1
        && this->pt2 == stS.pt2)
        return true;
    return false;
}

inline void stSegLine::GetGenLine(stGenLine& stG) const
{
    stG.da = (double)(this->pt1.y - this->pt2.y);
	stG.db = (double)(this->pt2.x - this->pt1.x);
	stG.dc = (double)(this->pt1.x * this->pt2.y - this->pt2.x * this->pt1.y);
}


