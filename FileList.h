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
//    //_A_ARCH（存档）
//    //_A_HIDDEN（隐藏）
//    //_A_NORMAL（正常）
//    //_A_RDONLY（只读）
//    //_A_SUBDIR（文件夹）
//    //_A_SYSTEM（系统）
//    time_t time_create;
//    //创建日期
//    time_t time_access;
//    //最后访问日期
//    time_t time_write;
//    //最后修改日期
//    _fsize_t size;
//    //文件大小
//    char name[_MAX_FNAME];
//    //文件名， _MAX_FNAME表示文件名最大长度
//};

//这个函数最好能用预处理该来同时能够兼容两个系统。
void Dir(string path, emImageFileType emtype, bool bExt/* = false*/);


void getFiles2(string path, vector<string>& files, vector<string>& ownname, emImageFileType emtype, bool bExt = false);


/*
@brief 检查或创建文件夹
*/
void MkDir(string sDir);


#endif // !FILELIST_H
