#include"Struct2d.h"

inline bool stCircle::operator==(const stCircle &stC) const
{
    if (this->ptCenter.x == stC.ptCenter.x
        && this->ptCenter.y == stC.ptCenter.y
        && this->dR == stC.dR)
        return true;
    return false;
}
