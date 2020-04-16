#ifndef CELLSPROCESSER_H
#define CELLSPROCESSER_H

#define EXPORT_DLL _declspec(dllexport)

extern "C" EXPORT_DLL int GetVersion(char *versionStr, int maxVersionStrLength);

extern "C" EXPORT_DLL int GetInnerVersion();

extern "C" EXPORT_DLL int GetMethodCount();

extern "C" EXPORT_DLL int GetMethodName(int methodIndex, char *methodName, int maxNameLengh);

extern "C" EXPORT_DLL int FindEdge(unsigned char *buf, int width, int height, int methodIndex, int *points, int *pointsNum);

extern "C" EXPORT_DLL int CalcFeatures(int *points, int pointsNum, char *features, int maxCharLengh);

extern "C" EXPORT_DLL double CalcClarity(unsigned char *buf, int width, int height);

#endif // CELLSPROCESSER_H
