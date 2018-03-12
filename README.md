# Contents

* General
* Constant Q
* Finite Impulse Response filter generator (windowed sinc)

# General

Eigen library must be present for compilation to work. You can download at http://eigen.tuxfamily.org/index.php?title=Main_Page

# Constant Q

This folder contains C++ code for generating the Constant Q transform matrix.

Compile with gcc or clang, for example:
```
clang++ -Ipath_to_Eigen_library -std=c++11 ./constantQ.cpp -o constantQ
```
Fastest executable I've created is with:
```
clang++ -Ipath_to_Eigen_library -std=c++14 -march=native -O3 -fopenmp ./constantQ.cpp -o constantQ
```
Make sure to install libomp-dev for OpenMP to work.

Program takes three versions:

```
./ConstantQ samplerate binsPerOctave minFrequency maxFrequency
./ConstantQ binsPerOctave minFrequency maxFrequency
./ConstantQ binsPerOctave maxFrequency
```
The version without samplerate will calculate filters for all common samplerates: 44100, 48000, 96000, 192000.

The version without samplerate or minFrequency will calculate optimum minFrequency

Example:
```
./ConstantQ 12 4434
```

## .dat file header and format.

constantQ outputs the matrix in compressed sparse format. The .dat file is structured as follows.

| Size in bytes | Field |
| --- | --- |
| 31 | Header |
| variable | Value |
| variable | Inner pointers |
| variable | Outer pointers |

To get the size of the variable sections, see the header structure:

Header: 32 bytes

| Byte offset | Type | Field |
| --- | --- | --- |
| 0-3 | uint_32t | Sample rate (Hz) |
| 4-7 | uint_32t | Bins per octave |
| 8-11 | float | Lowest frequency |
| 12-15 | float | Highest frequency |
| 16-19 | uint_32t | # of rows |
| 20-23 | uint_32t | # of columns |
| 24-27 | uint_32t | size of inner pointers (# values) |
| 28-31 | uint_32t | size of outer pointers |

# Finite Impulse Response Filter (windowed sinc)

Compile with gcc or clang, for example:
```
clang++ -Ipath_to_Eigen_library -std=c++11 ./fir.cpp -o fir
```

Program takes 5 versions
```
./fir
./fir numCrossings
./fir numCrossings samplesPerCrossing
./fir numCrossings samplesPerCrossing cutoffCycles
./fir numCrossings samplesPerCrossing cutoffCycles kaiserBeta
```
The first variant will generate fir.dat with 7 crossings, 32 samples per crossing, a cutoff of 0.5, and a beta of 7.0. For the other variants, arguments that are not supplied will default to 512 samples per crossing, 0.5 cutoff, and 7.0 beta.

## .dat file header and format.

Header:

| Byte offset | Type | Field |
| --- | --- | --- |
| 0-3 | uint_32t | Number of crossings |
| 4-7 | uint_32t | Samples per crossing |
| 8-11 | float | Cutoff cycles |
| 12-15 | float | Kaiser beta |

Body:
* filterSize is SamplesPerCrossing * (NumCrossings - 1) - 1

| Bytes | Type | Field |
| --- | --- | --- |
| filterSize * sizeof(float) | float | Filter Coefficients |
| filterSize * sizeof(float) | float | Delta Coefficients (filter[i + 1] - filter [i]) |