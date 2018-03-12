#include "fir.h"
#include <cmath>
#include <iostream>

#include "fileUtils.h"

static constexpr double EPSILON = 1e-21;

template <typename T>
double FIR<T>::ModBesselZeroth(const double x)
{
    double sum = 1.;
    double factorialStore = 1.;
    double halfX = x / 2.;
    double previous = 1.;
    do
    {
        double temp = halfX / factorialStore;
        temp *= temp;
        previous *= temp;
        factorialStore += 1.0;
        sum += previous;
    } while (previous >= EPSILON * sum);
    return sum;
}

template <typename T>
void FIR<T>::PopulateFilterDeltas()
{
    m_deltas.block(0, 0, m_filterSize - 1, 1) = m_coeffs.block(1, 0, m_filterSize - 1, 1) - m_coeffs.block(0, 0, m_filterSize - 1, 1);
    m_deltas[m_filterSize - 1] = -m_coeffs[m_filterSize - 1];
}

template <typename T>
void FIR<T>::PopulateFilterCoeffs()
{
    double invModBesselZerothBeta = 1. / ModBesselZeroth(m_kaiserBeta);
    double invSize = 1. / (double)(m_wingSize - 1);
    double invSamplesPerCrossing = 1. / m_samplesPerCrossing;
    ArrayXd offsetArray = ArrayXd::LinSpaced(m_wingSize - 1, 1, m_wingSize - 1);
    ArrayXd sincArray = PI * invSamplesPerCrossing * offsetArray;
    sincArray = sin(2 * sincArray * m_cutoffCycle) / sincArray;
    ArrayXd radicand = offsetArray * invSize;
    radicand = 1.0 - pow(radicand, 2);
    double localBeta = m_kaiserBeta;
    ArrayXd filterDouble(m_filterSize);
    filterDouble.block(m_center + 1, 0, m_wingSize - 1, 1) = sincArray * invModBesselZerothBeta * radicand.unaryExpr([&localBeta](const double rad) {
        return ModBesselZeroth(localBeta * sqrt(rad));
    });
    filterDouble[m_center] = 2. * m_cutoffCycle;
    filterDouble.block(0, 0, m_wingSize - 1, 1) = filterDouble.block(m_center + 1, 0, m_wingSize - 1, 1).reverse();
    const double dcGain = 2. * m_cutoffCycle;
    const double invGain = 1. / dcGain;
    filterDouble *= invGain;
    m_coeffs = filterDouble.cast<T>();
}

int main(int argc, char **argv)
{
    FIR<float>* fir;
    if (argc == 1) {
        fir = new FIR<float>(7, 32);
    } else if (argc == 2) {
        fir = new FIR<float>((u32)atoi(argv[1]));
    } else if (argc == 3) {
        fir = new FIR<float>((u32)atoi(argv[1]), (u32)atoi(argv[2]));
    } else if (argc == 4) {
        fir = new FIR<float>((u32)atoi(argv[1]), (u32)atoi(argv[2]), atof(argv[3]));
    } else if (argc == 5) {
        fir = new FIR<float>((u32)atoi(argv[1]), (u32)atoi(argv[2]), atof(argv[3]), atof(argv[4]));
    }
    bool success = WriteFIRFilter(fir->m_coeffs, fir->m_deltas, fir->m_numCrossings, fir->m_samplesPerCrossing, fir->m_cutoffCycle, fir->m_kaiserBeta);
    if (!success)
    {
        std::cerr << "write filter coeffs/deltas failed" << std::endl;
        return 1;
    }
    else
    {
        return 0;
    }
}