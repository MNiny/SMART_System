#ifndef IMAGEREPORTER_H
#define IMAGEREPORTER_H

#include "opencv2/opencv.hpp"

#include <QString>

using namespace std;

enum FeatureType
{
    Area,
    MinRectRatio,
    MinEnclosingCircleRadius,
    DisOfCenterToSide,
    MinMaxCircleRatio,
};

enum AlgType
{
    Canny,
    AdaptiveThreshold,
    KMeans,
    Otsu,
    Hsv,
    RiddlerCalvard,
};

struct TryMethodItem
{
    AlgType algType;
    int param;

    TryMethodItem(AlgType type, int p);
};

class ImageEdgeFinder
{
public:
    ImageEdgeFinder();
    void FindEdge(unsigned char *buf, int rows, int cols, int methodIndex, int *points, int *pointsNum);

    static QString GetFeatureName(FeatureType featureType);
    void CalcFeatures(int *points, int pointsNum, char *features, int maxCharLengh);

private:
    void OnSobelWithAdaptiveThresholdForContours(const cv::Mat &src, vector<vector<cv::Point> > &contours, int param);
    void OnCannyForContours(const cv::Mat &src, vector<vector<cv::Point> > &contours, int param);

    void Process(const cv::Mat &src, const vector<TryMethodItem> &tryMethodList, vector<cv::Point> &edge, QMap<FeatureType, double> &features);
    void CalcFeatures(vector<cv::Point> &contour, QMap<FeatureType, double> &features);
    double CalcRectRatio(cv::Rect rect);
    void OnGetContoursByAlgType(AlgType algType, const cv::Mat &src, vector<vector<cv::Point> > &contours, int param);
    void FindFloodFilledEdge(const cv::Mat &temp, vector<vector<cv::Point> > &contours, vector<cv::Vec4i> &hierarchy);
    void TryGetContoursAndFeatures(AlgType algType, const cv::Mat &src, vector<vector<cv::Point> > &contours, int param, QMap<FeatureType, double> &features);
    bool AreFeaturesCorrect(const QMap<FeatureType, double> &features);
    void BufToMat(unsigned char *buf, int rows, int cols, cv::Mat &src);
    void GetMethodByIndex(vector<TryMethodItem> &methodList, int methodIndex);
    void GetPointsForOutput(const vector<cv::Point> &edge, int *points, int *pointsNum, int rows, int cols);

    int m_scaledHeight;
    int m_scaledWidth;
};

#endif // IMAGEREPORTER_H



