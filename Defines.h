#pragma once
#ifndef DEFINES_H
#define DEFINES_H

//定义公司电脑，主要是公司电脑和个人电脑上用的路径不一样
//如果没有定义公司电脑，那就定义自己的电脑，每次切换就屏蔽一下定义公司电脑的模块就好了
#define COMPANY		
#ifndef COMPANY
#define USEROWN
#endif // !COMPANY

#ifdef COMPANY
string g_sResultDir = "E:\\Result";
#else
string g_sResultDir = "F:\\Result";
#endif // COMPANY












#endif // !DEFINES_H

