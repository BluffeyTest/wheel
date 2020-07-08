#include"Ransac.h"


Ransac::Ransac()
{
}

Ransac::~Ransac()
{
}

bool Ransac::InputPoints(std::vector<Point>& vec_pts)
{
    if (vec_pts.empty()) return false;
    m_vec_SrcPoints.assign(vec_pts.begin(), vec_pts.end() - 1);
    m_vec_Points.assign(vec_pts.begin(), vec_pts.end() - 1);
    return true;
}

bool Ransac::InputPara(RansacPara& stR)
{
    m_Para = stR;
    return false;
}

bool Ransac::Run()
{
    switch (m_Para.type)
    {
    case RansacPara::RASANC_SEG_LINE:FitSegLine(); break;
    case RansacPara::RASANC_SEG_LINES:FitSegLines(); break;
    case RansacPara::RASANC_SEG_CIRCLE:FitCircle(); break;
    case RansacPara::RASANC_SEG_CIRCLES:FitCircles(); break;
    default:
        break;
    }
    return false;
}

bool Ransac::FitCircle()
{
    for(size_t i = 0;i<m_Para.nIters;i++)
    {
    }
}
