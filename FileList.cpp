#include"FileList.h"

#include <iostream>
#include <string>
#include <io.h>
#include <direct.h>

using namespace std;

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
    // \\* ����Ҫ�������е�����,��ĳ�\\*.jpg��ʾ����jpg�����ļ�
    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileInfo)) == -1) {
        return;
    }
    do
    {
        //�ж��ļ����������ļ��л����ļ�
        cout << fileInfo.name << (fileInfo.attrib & _A_SUBDIR ? "[folder]" : "[file]") << endl;
    } while (_findnext(hFile, &fileInfo) == 0);
    _findclose(hFile);
    return;
}

void getFiles2(string path, vector<string>& files, vector<string>& ownname, emImageFileType emtype, bool bExt /*= false*/)
{
    /*files�洢�ļ���·��������(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
     ownnameֻ�洢�ļ�������(eg.     data1.txt)*/

     //�ļ����  
    //long   hFile = 0;//win7
    intptr_t   hFile = 0;//win10
    //�ļ���Ϣ  
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
            //�����Ŀ¼,���е�������,����֮  
            //�������,�����б�  
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
@brief ���򴴽��ļ���
*/
void MkDir(string sDir)
{
    if (0 != _access(sDir.c_str(), 0))
    {
        // if this folder not exist, create a new one.
        _mkdir(sDir.c_str());   // ���� 0 ��ʾ�����ɹ���-1 ��ʾʧ��
    }
}









