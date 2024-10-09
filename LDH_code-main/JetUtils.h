#pragma once
#include "Eigen/Core"
#include "Eigen/Dense"

// Define RRMat3 if not defined elsewhere
typedef Eigen::Matrix<double, 3, 3> RRMat3;  // Adjust type as needed
typedef Eigen::Matrix<double, 3, 1> RRVec3;  // Define RRVec3 similarly
typedef Eigen::Matrix<double, 2, 1> RRVec2;

#define RRMat3T Eigen::Matrix<T, 3, 3>
#define RRVec3T Eigen::Matrix<T, 3, 1>
#define RRVec2T Eigen::Matrix<T, 2, 1>

template <typename T>
void Mat3ToJet(const RRMat3& inA, RRMat3T& out) {
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            out(row, col) = T(inA(row, col));
}

template <typename T>
void Vec3ToJet(const RRVec3& inA, RRVec3T& out) {
    for (int row = 0; row < 3; row++)
        out(row) = T(inA(row));
}
