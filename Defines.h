#pragma once
#ifndef DEFINES_H
#define DEFINES_H

//���幫˾���ԣ���Ҫ�ǹ�˾���Ժ͸��˵������õ�·����һ��
//���û�ж��幫˾���ԣ��ǾͶ����Լ��ĵ��ԣ�ÿ���л�������һ�¶��幫˾���Ե�ģ��ͺ���
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

