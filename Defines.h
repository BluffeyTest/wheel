#pragma once
#ifndef DEFINES_H
#define DEFINES_H

#include<string>

//定义公司电脑，主要是公司电脑和个人电脑上用的路径不一样
//如果没有定义公司电脑，那就定义自己的电脑，每次切换就屏蔽一下定义公司电脑的模块就好了
#define COMPANY		
#ifndef COMPANY
#define USEROWN
#endif // !COMPANY

#ifdef COMPANY
std::string g_sResultDir = "E:\\Result";
#else
std::string g_sResultDir = "F:\\Result";
#endif // COMPANY

#define GAOYUAN











#endif // !DEFINES_H

