#pragma once

#include "ceres/types.h"
#include "ceres/ceres.h"
#include "ceres/solver.h"
#include "ceres/rotation.h"
#include "ceres/loss_function.h"
#include "JetUtils.h"
#include "RedRiverIndividualCameraAdjustment.h" // Ensure this header is included

namespace RedRiver {

	template <typename T>
	void constructInfinitHomographyA2B(const Eigen::Matrix<T, 3, 3>& kA_inverse_T, const Eigen::Matrix<T, 3, 3>& kB_T, const Eigen::Matrix<T, 3, 3>& RR, Eigen::Matrix<T, 3, 3>& HA2B) {
		HA2B = kB_T * RR * kA_inverse_T;
	}

	struct RRBA_PinholeReprojectionError_Extrinsics {
		RRBA_PinholeReprojectionError_Extrinsics(double observed_x, double observed_y, double principal_x, double principal_y, double focal_xy)
			: observed_x(observed_x), observed_y(observed_y), principal_x(principal_x), principal_y(principal_y), focal_xy(focal_xy) {}

		template <typename T>
		bool operator()(const T* const camera,
			const T* const point,
			T* residuals) const {

			//cout << "\n observed_x, observed_y: " << observed_x << "\t" << observed_y << "\n" ;
			const T focal_x = T(focal_xy);
			const T focal_y = T(focal_xy);
			// principal point
			const T u = T(principal_x);
			const T v = T(principal_y);

			T p[3];
			// camera[0,1,2] are the angle-axis rotation.
			ceres::AngleAxisRotatePoint(camera, point, p);

			// camera[3,4,5] are the translation.
			p[0] += camera[3];
			p[1] += camera[4];
			p[2] += camera[5];

			T xp = p[0] / p[2];
			T yp = p[1] / p[2];
			T predicted_x = focal_x * xp + u;
			T predicted_y = focal_y * yp + v;
			residuals[0] = predicted_x - T(observed_x);
			residuals[1] = predicted_y - T(observed_y);

			return true;
		}

		double observed_x;
		double observed_y;
		double principal_x;
		double principal_y;
		double focal_xy;
	};



	struct RRHomographyStitchingCostFunctor {
		RRHomographyStitchingCostFunctor(const RRVec3 xA, const RRMat3 kA_inverse, const RRVec3 xB, const RRMat3 kB)
			: xA(xA), xB(xB), kA_inverse(kA_inverse), kB(kB) {}
		template <typename T>
		bool operator()(const T* rotationVector, T* residuals) const {

			//typedef Eigen::Matrix<T, 3, 3> Mat3;
			//typedef Eigen::Matrix<T, 3, 1> Vec3;

			// Conversions between 3x3 rotation matrix (in column major order) and axis-angle rotation representations
			//  MatrixAdapter<T, row_stride, col_stride> M
			// M(i, j) is equivalent to	 arrary[i * row_stride + j * col_stride]
			// https://ceres-solver.googlesource.com/ceres-solver/+/master/include/ceres/rotation.h
			// R here is defined as row major (row_stride=3)
			ceres::MatrixAdapter<T, 3, 1> R(new T[9]);
			ceres::AngleAxisToRotationMatrix(rotationVector, R);

			//RRMat3 H = kB * R * kA_inverse;

			RRMat3T kA_inverse_T;
			Mat3ToJet<T>(kA_inverse, kA_inverse_T);

			RRMat3T kB_T;
			Mat3ToJet<T>(kB, kB_T);

			RRMat3T RR;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					RR(i, j) = R(i, j);

			RRMat3T H;
			constructInfinitHomographyA2B<T>(kA_inverse_T, kB_T, RR, H);

			RRVec3T xA_T;
			Vec3ToJet<T>(xA, xA_T);

			RRVec3T xB_predict = H * xA_T;
			xB_predict /= xB_predict(2);

			RRVec3T xB_T;
			Vec3ToJet<T>(xB, xB_T);

			residuals[0] = xB_predict(0) - xB_T(0);
			residuals[1] = xB_predict(1) - xB_T(1);

			return true;
		}
		const RRVec3 xA, xB;
		const RRMat3 kA_inverse, kB;
	};

}

