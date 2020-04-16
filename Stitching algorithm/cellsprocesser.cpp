#include "cellsprocesser.h"
#include "imagestitcher.h"
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


int StitchImagesByPath(const char *path[], int picColNum, int picRowNum, double overlapedRatio, unsigned char *buf, int canvasWidth, int canvasHeight, int *validWidth, int *validHeight)
{
    QStringList pathList;

    for (int i = 0; i < picColNum * picRowNum; ++i)
    {
        pathList.append(path[i]);
    }

    ImageStitcher imageStitcher;
    imageStitcher.StitchImagesByPath(pathList, picColNum, picRowNum, overlapedRatio, buf, canvasWidth, canvasHeight, validWidth, validHeight);
    return 1;
}

double CalcClarity(unsigned char *buf, int width, int height)
{
    ImageTool imageTool;
    cv::Mat src(height, width, CV_8UC3);
    imageTool.BufToMat(buf, height, width, src);

    return imageTool.CalcTenengrad(src);
}

