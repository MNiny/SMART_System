#include "imagetool.h"

#include <QDebug>
#include <QFile>

ImageTool::ImageTool()
{

}

void ImageTool::ShowImage(cv::Mat &mat, const QString &title)
{
    bool show = true;

    if (show)
    {
        cvNamedWindow(title.toLatin1().data(), CV_WINDOW_AUTOSIZE);
        cv::imshow(title.toLatin1().data(), mat);
        cv::waitKey(0);
    }
}

void ImageTool::ToGray(const cv::Mat &src, cv::Mat &dst)
{
    cv::cvtColor(src, dst, CV_BGR2GRAY);
}

void ImageTool::ToHSV(const cv::Mat &src, cv::Mat &dst)
{
    cv::cvtColor(src, dst, CV_BGR2HSV);
}

void ImageTool::Open(const cv::Mat &src, cv::Mat &dst, int param)
{
    cv::Mat element = cv::Mat::ones(param, param, CV_8UC1); // cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
    cv::morphologyEx(src, dst, CV_MOP_OPEN, element);
}

void ImageTool::Sobel(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat sobelx;
    cv::Mat sobelAbsGradx;
    cv::Sobel(src, sobelx, CV_64F, 1, 0);
    cv::convertScaleAbs(sobelx, sobelAbsGradx);

    cv::Mat sobely;
    cv::Mat sobelAbsGrady;
    cv::Sobel(src, sobely, CV_64F, 0, 1);
    cv::convertScaleAbs(sobely, sobelAbsGrady);

    cv::bitwise_or(sobelAbsGradx, sobelAbsGrady, dst);
}

void ImageTool::KMeans(const cv::Mat &src, cv::Mat &dst, int clusterCount)
{
    cv::Mat samples(src.cols * src.rows, 1, CV_32FC1);
    this->GrayToFloat32(src, samples);

    cv::Mat labels(src.cols * src.rows, 1, CV_32FC1);
    cv::Mat centers(clusterCount, 1, samples.type());
    cv::kmeans(samples, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    cv::Mat kmImage(src.size(), src.type());
    this->KMeansToGray(labels, centers, kmImage);

    uchar minGray = static_cast<uchar>(this->GetMingGray(centers));

    cv::Mat mdImage;
    cv::medianBlur(kmImage, mdImage, 5);

    cv::Mat openImage;
    cv::Mat element = cv::Mat::ones(5, 5, CV_8UC1); // cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
    cv::morphologyEx(mdImage, openImage, CV_MOP_OPEN, element, cv::Point(-1, -1), 2);

    cv::Mat clusteredMat(src.rows, src.cols, CV_8UC1);
    clusteredMat.setTo(cv::Scalar::all(0));

    this->FillImageByOneGray(openImage, clusteredMat, minGray);

    dst = clusteredMat;
}


void ImageTool::HsvThreshold(const cv::Mat &src, cv::Mat &dst)
{
    int hmin = 0, smin = 0, vmin = 0;
    int hmax = 70, smax = 255, vmax = 105;
    cv::inRange(src, cv::Scalar(hmin, smin, vmin), cv::Scalar(hmax, smax, vmax), dst);
}

void ImageTool::RiddlerCalvard(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat gaublurImage;
    cv::GaussianBlur(src, gaublurImage, cv::Size(5, 5), 0, 0);

    cv::threshold(gaublurImage, dst, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
}


void ImageTool::FillImageByOneGray(const cv::Mat &src, cv::Mat &dst, uchar gray)
{
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar *idata = src.ptr<uchar>(i);
        uchar *pdata = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            if (gray != idata[j])
            {
                pdata[j] = 0;
            }
            else
            {
                pdata[j] = 255;
            }
        }
    }
}

void ImageTool::GrayToFloat32(const cv::Mat &src, cv::Mat &dst)
{
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar *idata = src.ptr<uchar>(i);
        float *pdata = dst.ptr<float>(0);
        for (int j = 0; j < src.cols; ++j)
        {
            pdata[i*src.cols + j] = idata[j];
        }
    }
}

void ImageTool::KMeansToGray(const cv::Mat &labels, const cv::Mat &centers, cv::Mat &dst)
{
    const float *centerData = centers.ptr<float>(0);

    for (int i = 0; i < dst.rows; ++i)
    {
        const int *srcRow = labels.ptr<int>(0);
        uchar *dstRow = dst.ptr<uchar>(i);
        for (int j = 0; j < dst.cols; ++j)
        {
            float gray = centerData[srcRow[i * dst.cols + j]];
            dstRow[j] = static_cast<uchar>(gray);
        }
    }
}

float ImageTool::GetMingGray(const cv::Mat &mat)
{
    float min = 255;
    for (int i = 0; i < mat.rows; ++i)
    {
        const float *idata = mat.ptr<float>(i);

        for (int j = 0; j < mat.cols; ++j)
        {
            if (min > idata[j])
            {
                min = idata[j];
            }
        }
    }

    return min;
}

void ImageTool::PrintMat(const cv::Mat &mat)
{
    int k = 0;
    for (int i = 0; i < mat.rows; ++i)
    {
        const uchar *idata = mat.ptr<uchar>(i);

        for (int j = 0; j < mat.cols; ++j)
        {
            qDebug() << idata[j];
            k++;
        }
    }
}

void ImageTool::BufToMat(const unsigned char *buf, int rows, int cols, cv::Mat &dst)
{
    for (int i = 0; i < rows; ++i)
    {
        uchar *idata = dst.ptr<uchar>(i);

        for (int j = 0; j < cols; ++j)
        {
            int index = i * cols + j;
            idata[3 * j] = buf[3 * index];
            idata[3 * j + 1] = buf[3 * index + 1];
            idata[3 * j + 2] = buf[3 * index + 2];
        }
    }
}

bool ImageTool::MatToBuf(const cv::Mat &src, unsigned char *buf, int rows, int cols)
{
    bool result = false;
    if (src.rows == rows && src.cols == cols)
    {
        for (int i = 0; i < rows; ++i)
        {
            const uchar *idata = src.ptr<uchar>(i);

            for (int j = 0; j < cols; ++j)
            {
                int index = i * cols + j;
                buf[3 * index] = idata[3 * j];
                buf[3 * index + 1] = idata[3 * j + 1];
                buf[3 * index + 2] = idata[3 * j + 2];
            }
        }

        result = true;
    }

    return result;
}


void ImageTool::GetContours(const cv::Mat &src, vector<vector<cv::Point> > &contours)
{
    vector<cv::Vec4i> hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    std::sort(contours.begin(), contours.end(), ContoursSortFun);
}

void ImageTool::FindFloodFilledEdge(const cv::Mat &temp,
                                        vector<vector<cv::Point> > &contours,
                                        vector<cv::Vec4i> &hierarchy)
{
    if (contours.size() > 0 && contours[0].size() > 0)
    {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[0], center, radius);
        double disOfCenterToSide = cv::pointPolygonTest(contours[0], center, true);

        if (disOfCenterToSide > 10)
        {
            cv::Mat blackMat = cv::Mat::zeros(temp.size(), temp.type());
            cv::drawContours(blackMat, contours, 0, cv::Scalar(255), 1, 8, hierarchy);

            cv::floodFill(blackMat, cv::Point((int)center.x, (int)center.y), cv::Scalar(255), nullptr, cv::Scalar(100, 100, 100), cv::Scalar(100, 100, 100));

            cv::drawContours(blackMat, contours, 0, cv::Scalar(0), 1, 8, hierarchy);

            cv::cvtColor(blackMat, blackMat, CV_BGR2GRAY);
            //contours.clear();
            //hierarchy.clear();
            cv::Mat atImage;
            cv::threshold(blackMat, atImage, 20, 255, cv::THRESH_BINARY);

            cv::findContours(atImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::sort(contours.begin(), contours.end(), ContoursSortFun);
        }
    }
}

void ImageTool::ToPoints(int *points, int pointsNum, vector<cv::Point> &edge)
{
    if (pointsNum < MAX_EDGE_POINT_NUM)
    {

        for (int i = 0; i < pointsNum; ++i)
        {
            int x = points[2 * i];
            int y = points[(2 * i) + 1];
            edge.push_back(cv::Point(x, y));
        }
    }
}

void ImageTool::OpenImage(const QString &fileName, cv::Mat &image)
{
    QFile file(fileName);
    if (file.open(QIODevice::ReadOnly))
    {
        QByteArray byteArray = file.readAll();
        std::vector<char> data(byteArray.data(), byteArray.data() + byteArray.size());
        image = cv::imdecode(cv::Mat(data), 1);

        file.close();
    }

    cv::resize(image, image, cv::Size(1500, 1000));
}

void ImageTool::SaveImage(const QString &fileName, const cv::Mat &image)
{
    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly))
    {
        std::vector<uchar> data;
        cv::imencode(".jpg", image, data);
        QByteArray buf;
        for (std::vector<uchar>::iterator itea = data.begin(); itea != data.end(); ++itea)
        {
            buf.push_back(static_cast<char>(*itea));
        }

        file.write(buf);
        file.close();
    }
}

double ImageTool::CalcTenengrad(const cv::Mat &image)
{
    cv::Mat grayImage;
    this->ToGray(image, grayImage);

    cv::Mat sobelx;
    cv::Mat sobelAbsGradx;
    cv::Sobel(grayImage, sobelx, CV_64F, 1, 0);
    cv::convertScaleAbs(sobelx, sobelAbsGradx);

    cv::Mat sobely;
    cv::Mat sobelAbsGrady;
    cv::Sobel(grayImage, sobely, CV_64F, 0, 1);
    cv::convertScaleAbs(sobely, sobelAbsGrady);

    cv::Mat weightedDst;
    cv::addWeighted(sobelAbsGradx, 0.5, sobelAbsGrady, 0.5, 0, weightedDst);

    double meas = cv::mean(weightedDst)[0];

    return  meas;
}
