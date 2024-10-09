#include "RedRiverIndividualCameraAdjustment.h"
#include <chrono>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

namespace RedRiver {

class IntermediateCallback : public ceres::IterationCallback {
public:
    IntermediateCallback(double* rotation_vector) : rotation_vector_(rotation_vector) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        std::cout << "Iteration " << summary.iteration << ":\n";
        std::cout << "Rotation Vector: [" 
                  << rotation_vector_[0] << ", " 
                  << rotation_vector_[1] << ", " 
                  << rotation_vector_[2] << "]\n";
        return ceres::SOLVER_CONTINUE;
    }

private:
    double* rotation_vector_;
};

std::chrono::milliseconds DynamicHomographyStitchingUsingCeres::optimize(
    const std::vector<RRVec3>& xA, const RRMat3& kA, const std::vector<RRVec3>& xB, const RRMat3& kB, RRVec3& rotationVector, RRMat3& H_A2B, bool optimize_focal_length, bool verbose) 
{
    if (!optimize_focal_length) {
        auto startTime = std::chrono::high_resolution_clock::now();

        ceres::Problem problem;

        double optimizationBlock_rotation[3];
        for (int i = 0; i < 3; i++) {
            optimizationBlock_rotation[i] = rotationVector(i);
        }

        RRMat3 kA_inv = kA.inverse();
        for (int i = 0; i < xA.size(); i++) {
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RRHomographyStitchingCostFunctor, 2, 3>(
                new RRHomographyStitchingCostFunctor(xA[i], kA_inv, xB[i], kB)
            );

            ceres::LossFunction* loss_function = new ceres::CauchyLoss(1);
            problem.AddResidualBlock(cost_function, loss_function, optimizationBlock_rotation);
        }

        const bool enforce_LowerUpperBound = false;
        if (enforce_LowerUpperBound) {
            const double angle_bounds[3][2] = { {8, 11}, {-1.0, 1.0}, {-1.0, 1} };
            for (int i = 0; i < 3; i++) {
                problem.SetParameterLowerBound(optimizationBlock_rotation, i, deg2rad(angle_bounds[i][0]));
                problem.SetParameterUpperBound(optimizationBlock_rotation, i, deg2rad(angle_bounds[i][1]));
            }
        }
        // if (enforce_LowerUpperBound) {
        // const double angle_bounds[3][2] = {{8, 11}, {-1.0, 1.0}, {-1.0, 1.0}};
        //     for (int i = 0; i < 3; ++i) {
        //         problem.SetParameterLowerBound(optimizationBlock_rotation, i, angle_bounds[i][0] * M_PI / 180.0);
        //         problem.SetParameterUpperBound(optimizationBlock_rotation, i, angle_bounds[i][1] * M_PI / 180.0);
        //     }
        // }   

        ceres::Solver::Options solver_options;
        solver_options.minimizer_progress_to_stdout = verbose;
        solver_options.num_threads = 16;
        solver_options.max_num_iterations = 20;  // Ensure the maximum number of iterations is 20
        solver_options.function_tolerance = 1e-10;  // Set to a very low value to prevent early stopping
        solver_options.gradient_tolerance = 1e-10;  // Set to a very low value to prevent early stopping
        solver_options.parameter_tolerance = 1e-10; // Set to a very low value to prevent early stopping

        IntermediateCallback callback(optimizationBlock_rotation);
        solver_options.callbacks.push_back(&callback);

        ceres::Solver::Summary summary;
        ceres::Solve(solver_options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        for (int i = 0; i < 3; i++) {
            rotationVector(i) = optimizationBlock_rotation[i];
        }

        // Calculate the final homography matrix using the optimized rotation vector
        Eigen::Matrix<double, 3, 3> R;
        ceres::AngleAxisToRotationMatrix(optimizationBlock_rotation, R.data());
        H_A2B = kB * R * kA_inv;

        auto stopTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);

        return duration;
    }
    // If optimize_focal_length is true, return a default value (e.g., 0 milliseconds)
    return std::chrono::milliseconds(0);
}

double DynamicHomographyStitchingUsingCeres::getMedianReprojectionError(
    const std::vector<RRVec3> xA, RRMat3& kA, const std::vector<RRVec3> xB, const RRMat3 kB, const RRMat3 H_A2B) 
{
    std::vector<double> errors;
    for (int i = 0; i < xA.size(); i++) {
        RRVec3 xB_pred = H_A2B * xA[i];
        xB_pred /= xB_pred(2);
        RRVec2 residual = xB[i].head(2) - xB_pred.head(2);
        double e = residual.norm();
        errors.push_back(e);
    }
    std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
    double e_median = errors[errors.size() / 2];
    return e_median;
}

} // namespace RedRiver
