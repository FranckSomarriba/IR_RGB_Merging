#include "image_processing_utils.h"
#include <cmath> // Include cmath for M_PI


int main() {
    // Hardcoded image paths
    std::string image1_path = "/home/jay/ws/VIS_IR/dh_new/input/Frame0010_cam4_5.jpg";
    std::string image2_path = "/home/jay/ws/VIS_IR/dh_new/input/Frame0010_cam1_5.jpg"; // Use a different image

    // Read images
    cv::Mat img1, img2;
    if (!readImages(image1_path, image2_path, img1, img2)) {
        return -1;
    }

    // Hardcoded intrinsic parameters as an example
    RedRiver::RRMat3 kA, kB;
    kA << 17700, 0, img1.cols / 2,
          0, 17700, img1.rows / 2,
          0, 0, 1;

    kB << 17700, 0, img2.cols / 2,
          0, 17700, img2.rows / 2,
          0, 0, 1;

    // Detect and match features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::DMatch> matches;
    detectAndMatchFeatures(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matches);

    // Sort matches by distance and select the top 10 matches
    std::sort(matches.begin(), matches.end());
    matches.resize(10);

    // Visualize the top 10 matches with thicker lines
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    for (auto& match : matches) {
        cv::line(img_matches, keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt + cv::Point2f(img1.cols, 0), cv::Scalar(0, 255, 0), 5);
    }
    cv::imwrite("./top_10_matches_5.jpg", img_matches);

    // Prepare data for optimization
    std::vector<RedRiver::RRVec3> xA, xB;
    prepareOptimizationData(keypoints1, keypoints2, matches, xA, xB);

    RedRiver::RRVec3 rotationVector(0.0, 0.0, 0.0);
    RedRiver::RRMat3 H_A2B = RedRiver::RRMat3::Zero(); // Initialize to zero matrix

    // Print inputs to the optimization function
    printOptimizationInputs(rotationVector, H_A2B, kA, kB, xA, xB);

    // Create an instance of the optimization class
    RedRiver::DynamicHomographyStitchingUsingCeres stitcher;

    // Perform the optimization
    auto duration = stitcher.optimize(xA, kA, xB, kB, rotationVector, H_A2B, false, true);

    std::cout << "Optimization completed in " << duration.count() << " milliseconds." << std::endl;
    std::cout << "Final rotation vector: " << rotationVector.transpose() << std::endl;

    // Calculate and print the rotation matrix
    RedRiver::RRMat3 rotationMatrix;
    ceres::AngleAxisToRotationMatrix(rotationVector.data(), rotationMatrix.data());
    std::cout << "Rotation matrix: \n" << rotationMatrix << std::endl;

    // Calculate the magnitude of the rotation vector
    double rotation_magnitude = rotationVector.norm();
    std::cout << "Rotation magnitude (in radians): " << rotation_magnitude << std::endl;

    // Convert rotation magnitude to degrees
    double rotation_magnitude_degrees = rad2deg(rotation_magnitude);
    std::cout << "Rotation magnitude (in degrees): " << rotation_magnitude_degrees << std::endl;

    // Calculate the angular error between the optimized rotation and the identity matrix
    RedRiver::RRMat3 identity_matrix = RedRiver::RRMat3::Identity();
    double rotation_error = calculateRotationError(rotationMatrix, identity_matrix);
    std::cout << "Angular error (in degrees) between the optimized rotation and the identity matrix: " << rotation_error << std::endl;

    std::cout << "Final homography matrix (H_A2B): \n" << H_A2B << std::endl;

    // Calculate and print reprojection error
    double reprojectionError = stitcher.getMedianReprojectionError(xA, kA, xB, kB, H_A2B);
    std::cout << "Reprojection Error: " << reprojectionError << std::endl;

    // Convert H_A2B from Eigen::Matrix to cv::Mat using OpenCV function
    cv::Mat H_A2B_cv;
    cv::eigen2cv(H_A2B, H_A2B_cv);

    // Save the homography matrix
    saveHomographyMatrix(H_A2B, "./homography_5.txt");

    // Blend and save images
    blendAndSaveImages(img1, img2, H_A2B_cv, "./combined_image_5.jpg");

    return 0;
}
