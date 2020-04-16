#ifndef IMAGESTITCHER_H
#define IMAGESTITCHER_H

#include "opencv2/opencv.hpp"

#include <QString>

class ImageStitcher
{
public:
    ImageStitcher();
    virtual ~ImageStitcher();

    bool StitchImagesByPath(const QStringList &srcPathList,
                            int picColNum,
                            int picRowNum,
                            double overlapedRatio,
                            unsigned char *buf,
                            int canvasWidth,
                            int canvasHeight,
                            int *validWidth,
                            int *validHeight);

private:
    bool ImageMosacBack(const QStringList &srcPathList, double overlapedRatio, int picColNum, int picRowNum, cv::Mat &dst, int *validWidth, int *validHeight);
    void OptimizeSeamBack(int index, cv::Mat &img, cv::Mat &trans, int picColNum, int picRowNum, int width, int height, const QList<cv::Point> &offsetList, cv::Mat &dst);
};

#endif // IMAGESTITCHER_H
