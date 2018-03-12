#pragma once

#include "typedefs.h"
#include <iostream>

template <typename T>
class FIR
{
  public:
    FIR(u32 numCrossings = 17, u32 samplesPerCrossing = 512, double cutoffCycle = 0.5, double beta = 7.0)
        : m_numCrossings(numCrossings), m_samplesPerCrossing(samplesPerCrossing), m_wingSize(samplesPerCrossing * (numCrossings - 1) / 2), m_filterSize(m_wingSize * 2 - 1), m_center(m_wingSize - 1), m_cutoffCycle(cutoffCycle), m_kaiserBeta(beta)
    {
        std::cout << m_numCrossings << std::endl;
        std::cout << m_samplesPerCrossing << std::endl;
        std::cout << m_wingSize << std::endl;
        std::cout << m_filterSize << std::endl;
        std::cout << m_center << std::endl;
        m_coeffs.resize(m_filterSize);
        m_deltas.resize(m_filterSize);

        PopulateFilterCoeffs();
        PopulateFilterDeltas();
    }

    const u32 m_numCrossings;
    const u32 m_samplesPerCrossing;
    const u32 m_wingSize;
    const u32 m_filterSize;
    const u32 m_center;

    Eigen::Array<T, Eigen::Dynamic, 1> m_coeffs;
    Eigen::Array<T, Eigen::Dynamic, 1> m_deltas;

    const double m_kaiserBeta;
    const double m_cutoffCycle;

  private:
    static double ModBesselZeroth(const double x);
    void PopulateFilterCoeffs();
    void PopulateFilterDeltas();
};