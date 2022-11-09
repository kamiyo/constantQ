// Constant Q filter generator

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>
#include <cmath>
#include <iostream>
#include <vector>
#ifdef __linux__
#include <x86intrin.h>
#elif _WIN32
#include <intrin.h>
#endif

#include "args.hxx"
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

std::tuple<double, double> quantizeFrequencies(const ArrayXd &pianoFrequencies, const u32 sampleRate, const u32 binsPerOctave, const double minFreq, const double maxFreq, const u32 fftSize)
{
    Eigen::SparseMatrix<float> sparse;
    const double ratio = pow(2, 1. / (double)binsPerOctave);
    const double Q = 1. / (ratio - 1);
    const double adjMinF = std::max(Q * sampleRate / fftSize, minFreq);
    std::cout << "Adjusted minimum frequency: " << adjMinF << std::endl;
    double closestMinF = 0;
    double closestMaxF = 0;
    for (u32 i = 0; i < pianoFrequencies.size(); i++)
    {
        if (pianoFrequencies[i] > adjMinF)
        {
            if (i == 0)
            {
                closestMinF = pianoFrequencies[i];
                break;
            }
            double up = pianoFrequencies[i] - adjMinF;
            double down = adjMinF - pianoFrequencies[i - 1];
            closestMinF = (up < down) ? pianoFrequencies[i] : pianoFrequencies[i - 1];
            break;
        }
        else
        {
            continue;
        }
    }
    for (u32 i = 0; i < pianoFrequencies.size(); i++)
    {
        if (pianoFrequencies[i] > maxFreq)
        {
            if (i == pianoFrequencies.size() - 1)
            {
                closestMaxF = pianoFrequencies.tail(1)[0];
            }
            double up = pianoFrequencies[i] - maxFreq;
            double down = maxFreq - pianoFrequencies[i - 1];
            closestMaxF = (up < down) ? pianoFrequencies[i] : pianoFrequencies[i - 1];
            break;
        }
        else
        {
            continue;
        }
    }
    return {closestMinF, closestMaxF};
}

int main(int argc, char **argv)
{
    args::ArgumentParser parser("A program to generate Constant-Q filter.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Group required(parser, "These are required:", args::Group::Validators::All);
    args::Group _min(parser, "These are exclusive and optional:", args::Group::Validators::AtMostOne);
    args::Group _max(parser, "One required:", args::Group::Validators::Xor);
    args::ValueFlag<u32> _sampleRate(parser, "sampleRate", "The sample rate of the data you will apply this transform to", {'r', "sample-rate"});
    args::ValueFlag<u32> _binsPerOctave(required, "binsPerOctave", "How many bins per octave", {'b', "bins-per-octave"});
    args::ValueFlag<u32> _fftSize(parser, "fftSize", "Size of the FFT (2 * fft bins)", {'z', "fft-size"});
    args::ValueFlag<float> _minFreq(_min, "minFreq", "Frequency of lowest bin", {'n', "min-freq"});
    args::ValueFlag<u32> _minNote(_min, "minNote", "Note number of lowest bin, 1-88", {"min-note"});
    args::ValueFlag<float> _maxFreq(_max, "maxFreq", "Frequency of highest bin", {'x', "max-freq"});
    args::ValueFlag<u32> _maxNote(_max, "maxNote", "Note number of highest bin, 1-88", {"max-note"});

    try
    {
        parser.ParseCLI(argc, argv);

        // Setup piano frequencies for quantizing
        const double ratio = pow(2., 1. / 12.);
        ArrayXd pianoFrequencies(100);
        pianoFrequencies.setLinSpaced(0, 99);
        pianoFrequencies = pianoFrequencies.unaryExpr([&ratio](const double number) {
            return 440 * pow(ratio, number - 48);
        });

        const u32 binsPerOctave = args::get(_binsPerOctave);
        double minFreq = (_minFreq) ? args::get(_minFreq) :
                            (_minNote) ? pianoFrequencies[args::get(_minNote) - 1] : 0;
        double maxFreq = (_maxFreq) ? args::get(_maxFreq) : pianoFrequencies[args::get(_maxNote) - 1];
        const u32 fftSize = (_fftSize) ? next_power2(args::get(_fftSize)) : 32768u;

        if (_sampleRate)
        {
            const u32 sr = args::get(_sampleRate);
            auto [closestMinF, closestMaxF] = quantizeFrequencies(pianoFrequencies, sr, binsPerOctave, minFreq, maxFreq, fftSize);
            Eigen::SparseMatrix<float> sparse;
            genKernel(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
            bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
            if (!success)
            {
                std::cerr << "write sparse matrix failed" << std::endl;
                return 1;
            }
        }
        else
        {
            for (const u32 sr : COMMON_RATES)
            {
                auto [closestMinF, closestMaxF] = quantizeFrequencies(pianoFrequencies, sr, binsPerOctave, minFreq, maxFreq, fftSize);
                std::cout << "Closest min, max frequency: " << closestMinF << ", " << closestMaxF << std::endl;
                Eigen::SparseMatrix<float> sparse;
                genKernel(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
                bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
                if (!success)
                {
                    std::cerr << "write sparse matrix failed" << std::endl;
                    return 1;
                }
            }
        }
        return 0;
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::Error& e)
    {
        std::cerr << e.what() << std::endl << parser;
        return 1;
    }

    // if (argc <= 4)
    // {
    //     // generate kernels for all common samplerates
    //     // for JS, we have to make sure FFT size is limited to 16384 =(
    //     const u32 binsPerOctave = atoi(argv[1]);
    //     double minF = 0, maxF;
    //     if (argc == 3)
    //     {
    //         maxF = atof(argv[2]);
    //     }
    //     else
    //     {
    //         minF = atof(argv[2]);
    //         maxF = atof(argv[3]);
    //     }
    //     const double ratio = pow(2., 1. / 12.);
    //     ArrayXd pianoFrequencies(100);
    //     pianoFrequencies.setLinSpaced(0, 99);
    //     pianoFrequencies = pianoFrequencies.unaryExpr([&ratio](const double number) {
    //         return 440 * pow(ratio, number - 48);
    //     });
    //     for (const u32 sr : COMMON_RATES)
    //     {
    //         auto [closestMinF, closestMaxF] = quantizeFrequencies(pianoFrequencies, sampleRate, binsPerOctave, minFreq, maxFreq, fftSize);
    //         std::cout << "Closest min, max frequency: " << closestMinF << ", " << closestMaxF << std::endl;
    //         genKernel(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
    //         bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, closestMinF, closestMaxF);
    //         if (!success)
    //         {
    //             std::cerr << "write sparse matrix failed" << std::endl;
    //             return 1;
    //         }
    //     }
    //     return 0;
    // }
    // else
    // {
    //     const u32 sr = atoi(argv[1]);
    //     const u32 binsPerOctave = atoi(argv[2]);
    //     const double minF = atof(argv[3]);
    //     const double maxF = atof(argv[4]);
    //     Eigen::SparseMatrix<float> sparse;
    //     genKernel(sparse, sr, binsPerOctave, minF, maxF);
    //     bool success = WriteSparseMatrix<float>(sparse, sr, binsPerOctave, minF, maxF);
    //     if (!success)
    //     {
    //         std::cerr << "write sparse matrix failed" << std::endl;
    //         return 1;
    //     }
    //     return 0;
    // }
}