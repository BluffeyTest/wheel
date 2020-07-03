#include"FileList.h"

#include <iostream>
#include <string>
#include <io.h>
#include <direct.h>

using namespace std;

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
void Dir(string path, emImageFileType emtype, bool bExt = false)
{
    //long hFile = 0;//win7
    intptr_t hFile = 0;//win10
    struct _finddata_t fileInfo;
    string pathName, exdName;
    switch (emtype)
    {
    case emJpg:exdName = "\\*.jpg";
        break;
    case emBmp:exdName = "\\*.bmp";
        break;
    default:
        break;
    }
    // \\* 代表要遍历所有的类型,如改成\\*.jpg表示遍历jpg类型文件
    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileInfo)) == -1) {
        return;
    }
    do
    {
        //判断文件的属性是文件夹还是文件
        cout << fileInfo.name << (fileInfo.attrib & _A_SUBDIR ? "[folder]" : "[file]") << endl;
    } while (_findnext(hFile, &fileInfo) == 0);
    _findclose(hFile);
    return;
}

void getFiles2(string path, vector<string>& files, vector<string>& ownname, emImageFileType emtype, bool bExt /*= false*/)
{
    /*files存储文件的路径及名称(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
     ownname只存储文件的名称(eg.     data1.txt)*/

     //文件句柄  
    //long   hFile = 0;//win7
    intptr_t   hFile = 0;//win10
    //文件信息  
    struct _finddata_t fileinfo;
    string p, exdName;
    switch (emtype)
    {
    case emJpg:exdName = "\\*.jpg";
        break;
    case emBmp:exdName = "\\*.bmp";
        break;
    default:
        break;
    }

    if ((hFile = _findfirst(p.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //如果是目录,且有迭代需求,迭代之  
            //如果不是,加入列表  
            if ((fileinfo.attrib & _A_SUBDIR) && bExt)
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles2(p.assign(path).append("\\").append(fileinfo.name), files, ownname, emtype, bExt);
            }
            else
            {
                files.push_back(path + "\\" + fileinfo.name);
                ownname.push_back(fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

/*
@brief 检查或创建文件夹
*/
void MkDir(string sDir)
{
    if (0 != _access(sDir.c_str(), 0))
    {
        // if this folder not exist, create a new one.
        _mkdir(sDir.c_str());   // 返回 0 表示创建成功，-1 表示失败
    }
}









