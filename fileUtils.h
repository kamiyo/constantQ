#pragma once

#include <fstream>
#include <sstream>

#include "typedefs.h"

template <typename T>
bool WriteArray(std::ofstream &file, const T *data, const size_t length)
{
    if (file.is_open())
    {
        file.write((const char *)data, sizeof(T) * length);
        file.flush();
        if (file.bad())
        {
            std::cout << "bad" << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    return false;
}

template <typename T>
bool WriteSparseMatrix(Eigen::SparseMatrix<T> &matrix, const u32 samplerate, const u32 binsPerOctave, double minF, double maxF)
{
    std::stringstream filename_base;
    filename_base << "CQ_" << samplerate << ".dat";
    std::ofstream file(filename_base.str().c_str(), std::fstream::out | std::fstream::binary);
    const u32 rows = (u32)matrix.innerSize();
    const u32 cols = (u32)matrix.outerSize();
    const u32 nonZeros = (u32)matrix.nonZeros();
    const u32 outerSize = cols + 1;
    const float minFfloat = (float)minF;
    const float maxFfloat = (float)maxF;
    std::cout << "rows: " << rows << std::endl;
    std::cout << "cols: " << cols << std::endl;
    std::cout << "nonZeros: " << nonZeros << std::endl;
    std::cout << "outerSize: " << outerSize << std::endl;
    std::cout << "minFfloat: " << minFfloat << std::endl;
    std::cout << "maxFfloat: " << maxFfloat << std::endl;
    file.write((const char *)&samplerate, sizeof(u32));
    file.write((const char *)&binsPerOctave, sizeof(u32));
    file.write((const char *)&minFfloat, sizeof(float));
    file.write((const char *)&maxFfloat, sizeof(float));
    file.write((const char *)&rows, sizeof(u32));
    file.write((const char *)&cols, sizeof(u32));
    file.write((const char *)&nonZeros, sizeof(u32));  // size of values & inner pointer array
    file.write((const char *)&outerSize, sizeof(u32)); // size of outer pointer array

    bool success1 = WriteArray(file, matrix.valuePtr(), matrix.nonZeros());
    file.flush();

    bool success2 = WriteArray(file, matrix.innerIndexPtr(), matrix.nonZeros());
    file.flush();

    bool success3 = WriteArray(file, matrix.outerIndexPtr(), matrix.outerSize() + 1);
    file.flush();
    file.close();

    return success1 && success2 && success3;
}

template <typename T>
bool WriteFIRFilter(Eigen::Array<T, Eigen::Dynamic, 1> &coeffs, Eigen::Array<T, Eigen::Dynamic, 1> &deltas, const u32 numCrossings, const u32 samplesPerCrossing, const double cutoffCycle, const double beta)
{
    std::ofstream file("fir.dat", std::fstream::out | std::fstream::binary);
    const float cutoffFloat = (float)cutoffCycle;
    const float betaFloat = (float)beta;
    file.write((const char *)&numCrossings, sizeof(u32));
    file.write((const char *)&samplesPerCrossing, sizeof(u32));
    file.write((const char *)&cutoffFloat, sizeof(float));
    file.write((const char *)&betaFloat, sizeof(float));

    bool success1 = WriteArray(file, coeffs.data(), coeffs.rows());
    file.flush();
    bool success2 = WriteArray(file, deltas.data(), deltas.rows());
    file.flush();
    file.close();

    return success1 && success2;
}