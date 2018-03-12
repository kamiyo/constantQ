#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/FFT>

typedef unsigned int u32;
typedef int s32;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::MatrixXcd MatrixXcd;
typedef Eigen::SparseMatrix<float> SparseXf;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXcd VectorXcd;
typedef Eigen::ArrayXd ArrayXd;
typedef Eigen::ArrayXi ArrayXi;
typedef Eigen::Array<unsigned int, Eigen::Dynamic, 1> ArrayXui;
typedef Eigen::Triplet<float> Tripletf;
typedef Eigen::FFT<double> FFTd;
static constexpr u32 COMMON_RATES[] = {44100, 48000, 96000, 192000};

static constexpr double PI = 3.14159265358979323846;