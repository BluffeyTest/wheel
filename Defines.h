#pragma once
#ifndef DEFINES_H
#define DEFINES_H

#include<string>

//���幫˾���ԣ���Ҫ�ǹ�˾���Ժ͸��˵������õ�·����һ��
//���û�ж��幫˾���ԣ��ǾͶ����Լ��ĵ��ԣ�ÿ���л�������һ�¶��幫˾���Ե�ģ��ͺ���
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

