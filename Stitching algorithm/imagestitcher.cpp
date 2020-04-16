#include "imagestitcher.h"
#include "imagetool.h"

#include <QDebug>

using namespace cv;

ImageStitcher::ImageStitcher()
{

}

ImageStitcher::~ImageStitcher()
{

}

bool ImageStitcher::StitchImagesByPath(const QStringList &srcPathList,
                                       int picColNum,
                                       int picRowNum,
                                       double overlapedRatio,
                                       unsigned char *buf,
                                       int canvasWidth,
                                       int canvasHeight,
                                       int *validWidth,
                                       int *validHeight)
{
    *validWidth = 0;
    *validHeight = 0;

    cv::Mat dst(canvasHeight, canvasWidth, CV_8UC3);
    bool result = this->ImageMosacBack(srcPathList, overlapedRatio, picColNum, picRowNum, dst, validWidth, validHeight);

    ImageTool imageTool;
    result = result && imageTool.MatToBuf(dst, buf, canvasHeight, canvasWidth);

    return result;
}

bool ImageStitcher::ImageMosacBack(const QStringList &srcPathList,
                                   double overlapedRatio,
                                   int picColNum,
                                   int picRowNum,
                                   cv::Mat &dst,
                                   int *validWidth,
                                   int *validHeight)
{
    bool result = false;

    ImageTool imageTool;
    double realRatio = 1 - overlapedRatio;

    QList<Point> offsetList;
    for (int i = 0; i < picColNum * picRowNum; ++i)
    {
        offsetList.append(Point(0, 0));
    }

    //获取第一张图像
    Mat image;
    imageTool.OpenImage(srcPathList[0], image);

    int height = image.rows;
    int width = image.cols;

    //拼接结果图像
    Mat resultImg, tempImg;
    resultImg.create(height * picRowNum, width * picColNum, CV_8UC3);
    tempImg.create(height * picRowNum, width * picColNum, CV_8UC3);
    image.copyTo(resultImg(Rect(0, 0, width, height)));
    //依次读取其他图像
    Mat resultImage;

    bool flag = 1;
    for (int i = 1; i < picColNum * picRowNum; i++)
    {
        Mat newImage;
        imageTool.OpenImage(srcPathList[i], newImage);

        if (!newImage.data)
        {
            flag = 0;
            break;
        }

        if (i >= 1 && i < picColNum)
        {
            offsetList[i].x = offsetList[i - 1].x + static_cast<int>(width * realRatio);
            offsetList[i].y = offsetList[i - 1].y;
        }
        else if (i % picColNum == 0)
        {
            offsetList[i].x = offsetList[i - picColNum].x;
            offsetList[i].y = offsetList[i - picColNum].y + static_cast<int>(height * realRatio);
        }
        else
        {
            offsetList[i].x = offsetList[i - picColNum].x;
            offsetList[i].y = offsetList[i - 1].y;
        }

        newImage.copyTo(tempImg(Rect(offsetList[i].x, offsetList[i].y, width, height)));

        //拼接
        OptimizeSeamBack(i + 1, resultImg, tempImg, picColNum, picRowNum, width, height, offsetList, resultImg);
    }


    if (flag)
    {
        int holeWidth = width * picColNum - static_cast<int>(width * overlapedRatio * (picColNum - 1));
        int holeHeight = height * picRowNum - static_cast<int>(height * overlapedRatio * (picRowNum - 1));

        if (holeWidth <= dst.size().width && holeHeight <= dst.size().height)
        {
            result = true;
            Rect validRect = Rect(0, 0, holeWidth, holeHeight);

            Mat roi = dst(validRect);
            Mat mask(roi.rows, roi.cols, roi.depth(), Scalar(1));
            resultImg(validRect).copyTo(roi, mask);

            *validHeight = holeHeight;
            *validWidth = holeWidth;
        }
    }

    return result;
}

void ImageStitcher::OptimizeSeamBack(int index, Mat& img, Mat& trans, int picColNum, int picRowNum, int width, int height, const QList<Point> &offsetList, Mat& dst)
{
    double k = 0;
    //横向:向左拼接
    if (index >= 2 && index <= picColNum)
    {
        for (int i = offsetList[index - 1].y; i <= offsetList[index - 2].y + height; i++)
        {
            if (i < 0)
                continue;
            uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
            uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = offsetList[index - 1].x; j < offsetList[index - 2].x + width; j++)
            {
                k = 1.0*(j - offsetList[index - 1].x) / (offsetList[index - 2].x + width - offsetList[index - 1].x);
                if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
                {
                    k = 1;
                }
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
            for (int j = offsetList[index - 2].x + width; j < offsetList[index - 1].x + width; j++)
            {
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
        }
    }
    else if (index % picColNum == 1)
    {
        int startC = MAX(offsetList[index - 1].x, offsetList[index - picColNum - 1].x);
        int endC = MIN(offsetList[index - 1].x + width, offsetList[index - picColNum - 1].x + width);

        for (int i = offsetList[index - 1].y; i <= offsetList[index - picColNum - 1].y + height; i++)
        {
            uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
            uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            k = 1.0*(i - offsetList[index - 1].y) / (offsetList[index - picColNum - 1].y + height - offsetList[index - 1].y);
            for (int j = startC; j < endC; j++)
            {
                if (j < 0)
                    continue;
                if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
                {
                    k = 1;
                }
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
        }
        for (int i = offsetList[index - picColNum - 1].y + height; i < offsetList[index - 1].y + width; i++)
        {
            uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
            uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = startC; j < endC; j++)
            {
                if (j < 0)
                    continue;
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
        }
    }
    else
    {
        int endR = 0;
        if (index % picColNum == 0)
            endR = MAX(offsetList[index - picColNum - 2].y + height, offsetList[index - picColNum - 1].y + height);
        else
            endR = MAX(offsetList[index - picColNum - 1].y + height, MAX(offsetList[index - picColNum - 2].y + height, offsetList[index - picColNum].y + height));
        //上面矩形
        for (int i = offsetList[index - 1].y; i <= endR; i++)
        {
            uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
            uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            k = 1.0*(i - offsetList[index - 1].y) / (endR - offsetList[index - 1].y);
            for (int j = offsetList[index - 2].x + width; j < offsetList[index - 1].x + width; j++)
            {
                if (j < 0)
                    continue;
                if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
                {
                    k = 1;
                }
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
        }
        for (int i = offsetList[index - 1].y; i <= offsetList[index - 1].y + height; i++)
        {
            uchar* p = img.ptr<uchar>(i);  //获取第i行的首地址
            uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = offsetList[index - 1].x; j < offsetList[index - 2].x + width; j++)
            {
                if (j < 0)
                    continue;
                k = 1.0*(j - offsetList[index - 1].x) / (offsetList[index - 2].x + width - offsetList[index - 1].x);
                if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
                {
                    k = 1;
                }
                d[j * 3] = int(p[j * 3] * (1 - k) + t[j * 3] * k);
                d[j * 3 + 1] = int(p[j * 3 + 1] * (1 - k) + t[j * 3 + 1] * k);
                d[j * 3 + 2] = int(p[j * 3 + 2] * (1 - k) + t[j * 3 + 2] * k);
            }
        }
        Rect remain;
        Mat m_remain;
        remain = Rect(offsetList[index - 2].x + width, endR, offsetList[index - 1].x - offsetList[index - 2].x, offsetList[index - 1].y + height - endR - 1);
        m_remain = trans(remain);
        m_remain.copyTo(dst(remain));
    }
}
