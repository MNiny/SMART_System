#include "imageedgefinder.h"
#include "imagetool.h"

#include <QDebug>
#include <QLibrary>
#include <vector>
#include <QDateTime>

using namespace std;

ImageEdgeFinder::ImageEdgeFinder() :
    m_scaledHeight(1376),
    m_scaledWidth(1040)
{
}

void ImageEdgeFinder::FindEdge(unsigned char *buf, int rows, int cols, int methodIndex, int *points, int *pointsNum)
{
    ImageTool imageTool;
    cv::Mat src(rows, cols, CV_8UC3);
    imageTool.BufToMat(buf, rows, cols, src);

    vector<TryMethodItem> methodList;
    this->GetMethodByIndex(methodList, methodIndex);

    QMap<FeatureType, double> dstData;
    vector<cv::Point> edge;

    this->Process(src, methodList, edge, dstData);

    this->GetPointsForOutput(edge, points, pointsNum, rows, cols);
}

void ImageEdgeFinder::CalcFeatures(int *points, int pointsNum, char *features, int maxCharLengh)
{
    vector<cv::Point> edge;
    ImageTool imageTool;
    imageTool.ToPoints(points, pointsNum, edge);

    QMap<FeatureType, double> featureMap;
    this->CalcFeatures(edge, featureMap);

    QString featureString;
    QMap<FeatureType, double>::iterator itea = featureMap.begin();
    for (; itea != featureMap.end(); ++itea)
    {
        featureString += this->GetFeatureName(itea.key());
        featureString += ":";

        featureString += QString::number(itea.value(), 'f', 4);
        featureString += ",";
    }

    strncpy_s(features,
              static_cast<rsize_t>(maxCharLengh - 1),
              featureString.toLatin1().data(),
              static_cast<rsize_t>(featureString.length()));
}

void ImageEdgeFinder::GetPointsForOutput(const vector<cv::Point> &edge, int *points, int *pointsNum, int rows, int cols)
{
    double scalex = 0.0;
    double scaley = 0.0;

    scalex = cols / 1376.0;
    scaley = rows / 1040.0;

    if (edge.size() < MAX_EDGE_POINT_NUM)
    {
        *pointsNum = static_cast<int>(edge.size());

        for (vector<cv::Point>::size_type i = 0; i < edge.size(); ++i)
        {
            points[2 * i] = edge[i].x * scalex;
            points[(2 * i) + 1] = edge[i].y * scaley;
        }
    }
}

void ImageEdgeFinder::GetMethodByIndex(vector<TryMethodItem> &methodList, int methodIndex)
{
    switch (methodIndex)
    {
    case 0:
        methodList.push_back(TryMethodItem(Canny, 50)); // param是canny的参数
        methodList.push_back(TryMethodItem(Canny, 40));
        methodList.push_back(TryMethodItem(Canny, 30));
        methodList.push_back(TryMethodItem(AdaptiveThreshold, 4));
        methodList.push_back(TryMethodItem(AdaptiveThreshold, 8));
        break;

    default:
        break;
    }
}

QString ImageEdgeFinder::GetFeatureName(FeatureType featureType)
{
    QString featureName;
    switch (featureType)
    {
    case Area:
        featureName = "Area";
        break;

    case MinRectRatio:
        featureName = "Min rect ratio";
        break;

    case MinEnclosingCircleRadius:
        featureName = "Circumscribed Circle Radius";
        break;

    case DisOfCenterToSide:
        featureName = "Inscribed Circle Radius";
        break;

    case MinMaxCircleRatio:
        featureName = "Min max circle ratio";
        break;
    }

    return featureName;
}

void ImageEdgeFinder::Process(const cv::Mat &src,
                              const vector<TryMethodItem> &tryMethodList,
                              vector<cv::Point> &edge,
                              QMap<FeatureType, double> &features)
{
    cv::Mat temp;
    src.copyTo(temp);

    vector<TryMethodItem>::const_iterator iter = tryMethodList.begin();
    for (; iter != tryMethodList.end(); ++iter)
    {
        vector<vector<cv::Point> > contours;

        this->TryGetContoursAndFeatures(iter->algType, temp, contours, iter->param, features);

        if (this->AreFeaturesCorrect(features))
        {
            edge.insert(edge.end(), contours[0].begin(), contours[0].end());
            break;
        }
    }
}

bool ImageEdgeFinder::AreFeaturesCorrect(const QMap<FeatureType, double> &features)
{
    bool result = true;

    result = result && features.contains(MinRectRatio) && features[MinRectRatio] > 0.4;
    result = result && features.contains(MinEnclosingCircleRadius) && features[MinEnclosingCircleRadius] > 40;
    result = result && features.contains(DisOfCenterToSide) && features[DisOfCenterToSide] > 20;
    result = result && features.contains(MinMaxCircleRatio) && features[MinMaxCircleRatio] > 0.3;

    return result;
}

void ImageEdgeFinder::TryGetContoursAndFeatures(AlgType algType,
                                              const cv::Mat &src,
                                              vector<vector<cv::Point> > &contours,
                                              int param,
                                              QMap<FeatureType, double> &features)
{
    features.clear();

    this->OnGetContoursByAlgType(algType, src, contours, param);

    if (contours.size() > 0 && contours[0].size() > 0)
    {
        this->CalcFeatures(contours[0], features);
    }
}

void ImageEdgeFinder::OnGetContoursByAlgType(AlgType algType,
                                           const cv::Mat &src,
                                           vector<vector<cv::Point> > &contours,
                                           int param)
{
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(m_scaledHeight, m_scaledWidth), (0, 0), (0, 0), CV_INTER_AREA);

    switch (algType)
    {
    case Canny:
        this->OnCannyForContours(dst, contours, param);
        break;

    case AdaptiveThreshold:
        this->OnSobelWithAdaptiveThresholdForContours(dst, contours, param);
        break;


    }
}

void ImageEdgeFinder::CalcFeatures(vector<cv::Point> &contour, QMap<FeatureType, double> &features)
{
    if (contour.size() > 0)
    {
        double area = 0.0;
        area = cv::contourArea(contour);
        features.insert(Area, area);

        cv::RotatedRect minRect = cv::minAreaRect(contour);

        features.insert(MinRectRatio, this->CalcRectRatio(minRect.boundingRect()));

        // Min enclosing circle.
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);
        features.insert(MinEnclosingCircleRadius, static_cast<double>(radius));
        double disOfCenterToSide = cv::pointPolygonTest(contour, center, true);
        features.insert(DisOfCenterToSide, disOfCenterToSide);

        double minMaxCircleRatio = disOfCenterToSide / static_cast<double>(radius);
        features.insert(MinMaxCircleRatio, minMaxCircleRatio);
    }
}

double ImageEdgeFinder::CalcRectRatio(cv::Rect rect)
{
    double ratio = 1.0;

    if (rect.width > rect.height)
    {
        ratio = static_cast<double>(rect.height) / rect.width;
    }
    else
    {
        ratio = static_cast<double>(rect.width) / rect.height;
    }

    return ratio;
}


void ImageEdgeFinder::OnSobelWithAdaptiveThresholdForContours(const cv::Mat &src,
                                    vector<vector<cv::Point> > &contours,
                                    int param)
{
    ImageTool imageTool;

    cv::Mat grayImage;
    imageTool.ToGray(src, grayImage);

    cv::Mat atImage;
    cv::adaptiveThreshold(grayImage, atImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C + cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 7);

    cv::Mat openImage;
    imageTool.Open(atImage, openImage, param);

    cv::Mat sobelcombine;
    imageTool.Sobel(openImage, sobelcombine);

    imageTool.GetContours(sobelcombine, contours);
}

void ImageEdgeFinder::OnCannyForContours(const cv::Mat &src,
                                           vector<vector<cv::Point> > &contours,
                                           int param)
{
    ImageTool imageTool;

    cv::Mat grayImage;
    imageTool.ToGray(src, grayImage);
    cv::Mat cannyImage;

    cv::Canny(grayImage, cannyImage, param, param * 3, 3);

    cv::Mat linedImage;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(cannyImage, linedImage, CV_MOP_CLOSE, element);

    imageTool.GetContours(linedImage, contours);
}


TryMethodItem::TryMethodItem(AlgType type, int p)
{
    this->algType = type;
    this->param = p;
}
