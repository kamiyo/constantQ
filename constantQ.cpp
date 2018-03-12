// Constant Q filter generator

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/FFT>
#include <cmath>
#include <iostream>
#include <vector>
#ifdef __linux__
#include <x86intrin.h>
#elif _WIN32
#include <intrin.h>
#endif

#include <omp.h>
#include "typedefs.h"
#include "windowFunctions.h"
#include "fileUtils.h"

template <class T>
void genKernel(Eigen::SparseMatrix<T> &matrix, const u32 samplerate, const u32 binsPerOctave, const double minF, const double maxF, const double thresh = 0.001)
{
    const double ratio = pow(2, 1. / (double)binsPerOctave);
    const double Q = 1. / (ratio - 1);
    const u32 K = (u32)ceil(binsPerOctave * log2(maxF / minF));
    const double omega_min = 2. * PI * minF / (double)samplerate;
    ArrayXd omegas(K);
    omegas.setLinSpaced(0, K - 1);
    omegas = omega_min * omegas.unaryExpr([&ratio](const double element) {
        return pow(ratio, element);
    });
    ArrayXui Nkcqs(K);
    Nkcqs = omegas.unaryExpr([&Q](const double omega) {
        return (u32)ceil(2. * PI * Q / omega);
    });
    const u32 nfft = next_power2((u32)Nkcqs.maxCoeff());
    std::vector<Eigen::Triplet<T>> values;
    values.reserve(K * nfft);

#pragma omp parallel for
    for (u32 k = 0; k < K; k++)
    {
        const u32 N = Nkcqs[k];
        VectorXcd fftInput = VectorXcd::Zero(nfft);
        VectorXcd window = VectorXcd::LinSpaced(N, std::complex<double>(0, 0), std::complex<double>(N - 1, 0));
        window = window.unaryExpr([&n = N, &o = omegas[k] ](const std::complex<double> e) {
            return hamming(25. / 46., e.real(), n) * exp(std::complex<double>(0, 1) * o * e);
        });
        fftInput.block(0, 0, N, 1) = window;
        VectorXcd kernel;
        FFTd fft;
        fft.fwd(kernel, fftInput);
        kernel /= (double)nfft;
        for (u32 i = 0; i < (u32)kernel.size() / 2; i++)
        {
            if (abs(kernel[i]) >= thresh)
            {
#pragma omp critical
                values.push_back(Eigen::Triplet<T>(i, k, (T)abs(kernel[i])));
            }
        }
    }
    matrix.resize(nfft / 2, K);
    matrix.setFromTriplets(values.begin(), values.end());
    matrix.makeCompressed();
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "usage: ./constantQ [samplerate] bins/octave [minFreq] maxFreq" << std::endl;
        return 1;
    }
    if (argc <= 4)
    {
        // generate kernels for all common samplerates
        // for JS, we have to make sure FFT size is limited to 16384 =(
        const u32 binsPerOctave = atoi(argv[1]);
        double minF = 0, maxF;
        if (argc == 3)
        {
            maxF = atof(argv[2]);
        }
        else
        {
            minF = atof(argv[2]);
            maxF = atof(argv[3]);
        }
        const double ratio = pow(2., 1. / 12.);
        ArrayXd pianoFrequencies(100);
        pianoFrequencies.setLinSpaced(0, 99);
        pianoFrequencies = pianoFrequencies.unaryExpr([&ratio](const double number) {
            return 440 * pow(ratio, number - 48);
        });
        for (const u32 sr : COMMON_RATES)
        {
            Eigen::SparseMatrix<float> sparse;
            const double ratio = pow(2, 1. / (double)binsPerOctave);
            const double Q = 1. / (ratio - 1);
            const double adjMinF = std::max(Q * sr / 32768, minF);
            std::cout << "Adjusted minimum frequency: " << adjMinF << std::endl;
            double closestMinF = 0;
            double closestMaxF = 0;
            for (u32 i = 0; i < pianoFrequencies.size(); i++)
            {
                if (pianoFrequencies[i] < adjMinF)
                {
                    continue;
                }
                else
                {
                    closestMinF = pianoFrequencies[i];
                    break;
                }
            }
            for (u32 i = 0; i < pianoFrequencies.size(); i++)
            {
                if (pianoFrequencies[i] < maxF)
                {
                    continue;
                }
                else
                {
                    closestMaxF = pianoFrequencies[i];
                    break;
                }
            }
            std::cout << "Closest min, max frequency: " << closestMinF << ", " << closestMaxF << std::endl;
            genKernel(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
            bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
            if (!success)
            {
                std::cerr << "write sparse matrix failed" << std::endl;
                return 1;
            }
        }
        return 0;
    }
    else
    {
        const u32 sr = atoi(argv[1]);
        const u32 binsPerOctave = atoi(argv[2]);
        const double minF = atof(argv[3]);
        const double maxF = atof(argv[4]);
        Eigen::SparseMatrix<float> sparse;
        genKernel(sparse, sr, binsPerOctave, minF, maxF);
        bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, minF, maxF);
        if (!success)
        {
            std::cerr << "write sparse matrix failed" << std::endl;
            return 1;
        }
        return 0;
    }
}