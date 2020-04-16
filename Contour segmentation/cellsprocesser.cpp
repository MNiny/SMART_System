#include "cellsprocesser.h"
#include "imageedgefinder.h"
#include "imagetool.h"
#include <iostream>
#include <QDebug>

using namespace std;

int GetVersion(char *versionStr, int maxVersionStrLength)
{
    int result = 0;

    const char *version = "1.0.0.0";
    if (static_cast<unsigned long long>(maxVersionStrLength) > strlen(version))
    {
        strncpy_s(versionStr, static_cast<unsigned long long>(maxVersionStrLength), version, strlen(version));
        result = 1;
    }

    return result;
}

int GetInnerVersion()
{
    if (VER_REVISIONSVN!=-1 && VER_REVISIONSVN_IF_NOT_MIX !=-1)
    {
        return VER_REVISIONSVN;
    }
    else
    {
        return -1;
    }

}

int GetMethodCount()
{
    return 7;
}

int SetMethodName(char*methodName, int maxNameLength, const char* name)
{
    int result = 0;
    if (static_cast<unsigned long long>(maxNameLength) > strlen(name))
    {
        strncpy_s(methodName, static_cast<unsigned long long>(maxNameLength), name, strlen(name));
        result = 1;
    }

    return result;
}

int GetMethodName(int methodIndex, char *methodName, int maxNameLength)
{
    int result = 0;
    switch (methodIndex)
    {  
    case 0:
        result = SetMethodName(methodName, maxNameLength, "Invasiveness");
        break;
    }

    return result;
}

int FindEdge(unsigned char *buf, int width, int height, int methodIndex, int *points, int *pointsNum)
{
    ImageEdgeFinder finder;
    finder.FindEdge(buf, height, width, methodIndex, points, pointsNum);
    return 1;
}

int CalcFeatures(int *points, int pointsNum, char *features, int maxCharLengh)
{
    ImageEdgeFinder finder;
    finder.CalcFeatures(points, pointsNum, features, maxCharLengh);
    return 1;
}

double CalcClarity(unsigned char *buf, int width, int height)
{
    ImageTool imageTool;
    cv::Mat src(height, width, CV_8UC3);
    imageTool.BufToMat(buf, height, width, src);

    return imageTool.CalcTenengrad(src);
}

