#include "image_processing_utils.h"

bool readImages(const std::string& image1_path, const std::string& image2_path, cv::Mat& img1, cv::Mat& img2) {
    img1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return false;
    }
    return true;
}

void detectAndMatchFeatures(const cv::Mat& img1, const cv::Mat& img2,
                            std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
                            cv::Mat& descriptors1, cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches) {
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->match(descriptors1, descriptors2, matches);

    std::sort(matches.begin(), matches.end());
    matches.resize(10);

    // Visualize the top 10 matches with thicker lines
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    for (const auto& match : matches) {
        cv::line(img_matches, keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt + cv::Point2f(img1.cols, 0), cv::Scalar(0, 255, 0), 10); // Increase the thickness here
    }
    cv::imwrite("./top_10_matches_50.jpg", img_matches);
}

void prepareOptimizationData(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                             const std::vector<cv::DMatch>& matches,
                             std::vector<RedRiver::RRVec3>& xA, std::vector<RedRiver::RRVec3>& xB) {
    for (const auto& match : matches) {
        cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
        xA.emplace_back(pt1.x, pt1.y, 1.0);
        xB.emplace_back(pt2.x, pt2.y, 1.0);
    }
}

void printOptimizationInputs(const RedRiver::RRVec3& rotationVector, const RedRiver::RRMat3& H_A2B,
                             const RedRiver::RRMat3& kA, const RedRiver::RRMat3& kB,
                             const std::vector<RedRiver::RRVec3>& xA, const std::vector<RedRiver::RRVec3>& xB) {
    std::cout << "Initial rotation vector: " << rotationVector.transpose() << std::endl;
    std::cout << "Initial homography matrix (H_A2B): \n" << H_A2B << std::endl;
    std::cout << "kA matrix: \n" << kA << std::endl;
    std::cout << "kB matrix: \n" << kB << std::endl;
    std::cout << "xA points: \n";
    for (const auto& point : xA) {
        std::cout << point.transpose() << std::endl;
    }
    std::cout << "xB points: \n";
    for (const auto& point : xB) {
        std::cout << point.transpose() << std::endl;
    }
}

void saveHomographyMatrix(const RedRiver::RRMat3& H_A2B, const std::string& filename) {
    std::ofstream homography_file(filename);
    if (homography_file.is_open()) {
        homography_file << H_A2B << std::endl;
        homography_file.close();
    }
}

void blendAndSaveImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& H_A2B_cv, const std::string& filename) {
    cv::Mat img1_color, img2_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);

    cv::Mat warped_img1;
    cv::warpPerspective(img1_color, warped_img1, H_A2B_cv, cv::Size(img1.cols + img2.cols, img1.rows));

    cv::Mat result = warped_img1.clone();
    cv::Mat roi(result, cv::Rect(0, 0, img2_color.cols, img2_color.rows));
    cv::addWeighted(roi, 0.5, img2_color, 0.5, 0.0, roi);

    cv::imwrite(filename, result);
}

double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}

double calculateRotationError(const RedRiver::RRMat3& R1, const RedRiver::RRMat3& R2) {
    RedRiver::RRMat3 I1 = R1 * R2.transpose();
    double tr_I1 = I1.trace();
    double tmp = (tr_I1 - 1) / 2.0;
    double ang_error = std::acos(tmp) * 180.0 / M_PI;
    return ang_error;
}
