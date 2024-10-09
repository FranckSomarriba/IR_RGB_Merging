#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#define GLOG_NO_ABBREVIATED_SEVERITIES

#include "ceres/types.h"
#include "ceres/ceres.h"
#include "ceres/solver.h"
#include "ceres/rotation.h"
#include "ceres/loss_function.h"

#include "JetUtils.h"
#include "RedRiverOptimizationCostFunctors.h"

namespace RedRiver {

typedef Eigen::Matrix<double, 3, 3> RRMat3;  // Ensure matching type definitions
typedef Eigen::Matrix<double, 3, 1> RRVec3;  // Ensure matching type definitions

class SingleCameraPoseOptimization {
public:
    typedef std::vector<std::pair<RRVec2, RRVec3>> Correspondences2D3D;
    struct ConstantParameters {
        Correspondences2D3D *correspondences;
        RRMat3 K; // intrinsics 
    };
    enum Method {
        OPENCV
    };

    std::chrono::milliseconds optimize(Correspondences2D3D& correspondences, RRMat3 K, RRVec3& r, RRVec3& t, RRVec3& r_delta, RRVec3& t_delta, Method method=OPENCV);
};

template <typename T>
void swapXYAxesHomography(RRMat3T& H, size_t image_width, size_t image_height) {
    // make sure it is normalized
    H /= H(2, 2);
    // swap x and y
    RRMat3T H_swapXY;
    H_swapXY << 0, 1, 0 , 1, 0, 0, 0, 0, 1;
    // reverse (put back!) the new x axis
    //flipYAxisHomography<T>(H_swapXY, image_height);
    H = H_swapXY * H;
}

template <typename T>
void flipXAxisHomography(RRMat3T& H, size_t image_width) {
    // make sure it is normalized
    H /= H(2, 2);
    // reverse X axis
    RRMat3T H_reveresX;
    H_reveresX << -1, 0, image_width-1, 0, 1, 0 , 0, 0, 1;
    H = H_reveresX * H;
}

template <typename T>
void flipYAxisHomography(RRMat3T& H, size_t image_height) {
    // make sure it is normalized
    H /= H(2, 2);
    // reverse Y axis
    RRMat3T H_reveresY;
    H_reveresY << 1, 0, 0, 0, -1, image_height-1, 0, 0, 1;
    H = H_reveresY * H;
}

class DynamicHomographyStitchingUsingCeres {
public:
    // call this to optimize the rotationVector values and obtain a homography (H_A2B)
    std::chrono::milliseconds optimize(const std::vector<RRVec3>& xA, const RRMat3& kA, const std::vector<RRVec3>& xB, const RRMat3& kB, RRVec3& rotationVector, RRMat3& H_A2B, bool optimize_focal_length, bool verbose);
    std::chrono::milliseconds optimize(const std::vector<RRVec3>& xA, const RRMat3& kA, const std::vector<RRVec3>& xB, const RRMat3& kB, RRVec3& rotationVector, RRMat3& H_A2B, bool verbose);

    double getMedianReprojectionError(const std::vector<RRVec3> xA, RRMat3& kA, const std::vector<RRVec3> xB, const RRMat3 kB, const RRMat3 H_A2B);
};

class DynamicHomographyStitching4Cam {
public:
    std::chrono::milliseconds optimize(
        std::vector<std::pair<RRVec3, RRVec3>>* xPairs, RRMat3* K,
        RRVec3& rv_BR, RRVec3& rv_BL, RRVec3& rv_TL,
        bool optimizeFocals
    );
};

}
