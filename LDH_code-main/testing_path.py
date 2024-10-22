import os # Used to set up environmnet variables
import cv2 # OpenCV library
import numpy as np
from PyQt5.QtCore import QLibraryInfo, QLibraryInfo
from pathlib import Path # Helps with paths to folders


# Function to read the homography matrix from a text file
def read_homography(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Makes the text file into a 3x3 matrix
        H = np.array([list(map(float, line.strip().split())) for line in lines]) 
    return H

# Set the environment variable for Qt plugin path to prevent compatibility issues
# Qt is a cross-platform software for GUI related functions
# Get the plugin path from PyQt5
qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)

# Set the environment variable dynamically
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

# Set "assets" as the path
images_path = Path("../assets")  

# Load images for resizing step
image0_load = images_path / "city_1.png"
image1_load = images_path / "city_2.png"

# Load the images in color (BGR)
img1 = cv2.imread(image0_load, cv2.IMREAD_COLOR)
img2 = cv2.imread(image1_load, cv2.IMREAD_COLOR)

# # Check images are properly loaded
# cv2.imshow("image1",img1)
# cv2.imshow("image2", img2)
# cv2.waitKey(0)


# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

# Match descriptors using BFMatcher and K-nearest neighbor
# BFMatcher = Brute Force Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to find the top matches
good_matches = []
ratio_thresh = 0.75

for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# Check if we have enough matches, if not, adjust ratio threshold
while len(good_matches) < 10 and ratio_thresh < 1.0:
    ratio_thresh += 0.05
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

# Sort matches by distance and select the top 10
good_matches = sorted(good_matches, key=lambda x: x.distance)[:30]

# Extract location of good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Print the pairs of match points
print("Pairs of match points (Image 1 -> Image 2):")
for i in range(len(pts1)):
    print(f"Point {i+1}: {pts1[i]} -> {pts2[i]}")

# Draw the top 10 matches with flags to show match lines clearly
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                              matchColor=(0, 255, 0),  # draw matches in green color
                              singlePointColor=(255, 0, 0),  # draw keypoints in red color
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Resize the image to make it smaller for display
img_matches_resized = cv2.resize(img_matches, (img_matches.shape[1] // 1, img_matches.shape[0] // 1))
cv2.imshow("Matches", img_matches_resized)
cv2.waitKey(0)