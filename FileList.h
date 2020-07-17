#pragma once
#ifndef FILELIST_H
#define FILELIST_H

#include <iostream>
#include <string>
#include <io.h>
#include <direct.h>
#include<vector>

using namespace std;

enum emImageFileType
{
    emJpg = 0,
    emBmp = 1,

    emLog =10

};

//struct _finddata_t
//{
//    unsigned attrib;
//    //_A_ARCH���浵��
//    //_A_HIDDEN�����أ�
//    //_A_NORMAL��������
//    //_A_RDONLY��ֻ����
//    //_A_SUBDIR���ļ��У�
//    //_A_SYSTEM��ϵͳ��
//    time_t time_create;
//    //��������
//    time_t time_access;
//    //����������
//    time_t time_write;
//    //����޸�����
//    _fsize_t size;
//    //�ļ���С
//    char name[_MAX_FNAME];
//    //�ļ����� _MAX_FNAME��ʾ�ļ�����󳤶�
//};

//��������������Ԥ�������ͬʱ�ܹ���������ϵͳ��
void Dir(string path, emImageFileType emtype, bool bExt/* = false*/);


void getFiles2(string path, vector<string>& files, vector<string>& ownname, emImageFileType emtype, bool bExt = false);


/*
@brief ���򴴽��ļ���
*/
void MkDir(string sDir);


#endif // !FILELIST_H
