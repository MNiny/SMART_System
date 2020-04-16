#ifndef CELLSPROCESSER_H
#define CELLSPROCESSER_H

#define EXPORT_DLL _declspec(dllexport)

extern "C" EXPORT_DLL int GetVersion(char *versionStr, int maxVersionStrLength);

extern "C" EXPORT_DLL int StitchImagesByPath(const char *path[],
                                             int picColNum,
                                             int picRowNum,
                                             double overlapedRatio,
                                             unsigned char *buf,
                                             int canvasWidth,
                                             int canvasHeight,
                                             int *validWidth,
                                             int *validHeight);

extern "C" EXPORT_DLL double CalcClarity(unsigned char *buf, int width, int height);

#endif // CELLSPROCESSER_H
