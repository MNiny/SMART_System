
#ifndef IMAGETOOL_H
#define IMAGETOOL_H

#include "opencv2/opencv.hpp"

#include <QString>

#define MAX_EDGE_POINT_NUM 65535

using namespace std;

class ImageTool
{
public:
    ImageTool();

    void ToGray(const cv::Mat &src, cv::Mat &dst);
    void ToHSV(const cv::Mat &src, cv::Mat &dst);

    void Open(const cv::Mat &src, cv::Mat &dst, int param);
    void Sobel(const cv::Mat &src, cv::Mat &dst);
    void KMeans(const cv::Mat &src, cv::Mat &dst, int clusterCount);
    void HsvThreshold(const cv::Mat &src, cv::Mat &dst);
    void RiddlerCalvard(const cv::Mat &src, cv::Mat &dst);

    void GetContours(const cv::Mat &src, vector<vector<cv::Point> > &contours);
    void FindFloodFilledEdge(const cv::Mat &temp, vector<vector<cv::Point> > &contours, vector<cv::Vec4i> &hierarchy);

    void KMeansToGray(const cv::Mat &labels, const cv::Mat &centers, cv::Mat &dst);

    void ShowImage(cv::Mat &mat, const QString &title);

    // type transfer
    void ToPoints(int *points, int pointsNum, vector<cv::Point> &edge);
    void BufToMat(const unsigned char *buf, int rows, int cols, cv::Mat &dst);
    bool MatToBuf(const cv::Mat &src, unsigned char *buf, int rows, int cols);

    // file operations
    void OpenImage(const QString &fileName, cv::Mat &image);
    void SaveImage(const QString &fileName, const cv::Mat &image);

    double CalcTenengrad(const cv::Mat &image);

private:
    static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
    {
        return (cv::contourArea(contour1) > cv::contourArea(contour2));
    }

    void GrayToFloat32(const cv::Mat &src, cv::Mat &dst);
    void PrintMat(const cv::Mat &mat);
    float GetMingGray(const cv::Mat &mat);
    void FillImageByOneGray(const cv::Mat &src, cv::Mat &dst, uchar gray);
};

#endif // IMAGETOOL_H

