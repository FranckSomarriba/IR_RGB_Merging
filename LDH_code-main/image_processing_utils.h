#pragma once

#include "RedRiverIndividualCameraAdjustment.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// Function to read images
bool readImages(const std::string& image1_path, const std::string& image2_path, cv::Mat& img1, cv::Mat& img2);

// Function to detect and match features
void detectAndMatchFeatures(const cv::Mat& img1, const cv::Mat& img2,
                            std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                            cv::Mat& descriptors1, cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches);

// Function to prepare data for optimization
void prepareOptimizationData(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                             const std::vector<cv::DMatch>& matches,
                             std::vector<RedRiver::RRVec3>& xA, std::vector<RedRiver::RRVec3>& xB);

// Function to print inputs to the optimization function
void printOptimizationInputs(const RedRiver::RRVec3& rotationVector, const RedRiver::RRMat3& H_A2B,
                             const RedRiver::RRMat3& kA, const RedRiver::RRMat3& kB,
                             const std::vector<RedRiver::RRVec3>& xA, const std::vector<RedRiver::RRVec3>& xB);

// Function to save the homography matrix to a file
void saveHomographyMatrix(const RedRiver::RRMat3& H_A2B, const std::string& filename);

// Function to blend and save the images
void blendAndSaveImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& H_A2B_cv, const std::string& filename);

// Helper function to convert radians to degrees
double rad2deg(double radians);

// Function to calculate the rotation error using the trace method
double calculateRotationError(const RedRiver::RRMat3& R1, const RedRiver::RRMat3& R2);
