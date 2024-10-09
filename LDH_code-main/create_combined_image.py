import os # Used to set up environmnet variables
import cv2 # OpenCV library
import numpy as np

# Function to read the homography matrix from a text file
def read_homography(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Makes the text file into a 3x3 matrix
        H = np.array([list(map(float, line.strip().split())) for line in lines]) 
    return H

# Set the environment variable for Qt plugin path to prevent compatibility issues
# Qt is a cross-platform software for GUI related functions
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/jay/libraries/anaconda3/lib/python3.11/site-packages/cv2/qt/plugins'

# Load the images in color (BGR)
img1 = cv2.imread("//media/jay/Files/RGB_images/VIS_IR/eo/DJI_0444_W.JPG", cv2.IMREAD_COLOR)
img2 = cv2.imread("/media/jay/Files/RGB_images/VIS_IR/ir/DJI_0443_T.JPG", cv2.IMREAD_COLOR)

# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to find the top matches
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
good_matches = sorted(good_matches, key=lambda x: x.distance)[:10]

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

# Read the homography matrix from the text file
H = read_homography("/home/jay/ws/VIS_IR/dh_new/build/homography_5.txt")

# Print the homography matrix
print("Homography matrix:")
print(H)

# Warp image 1 to the perspective of image 2
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# Get the canvas size that can hold both images
corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
corners_img2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

# Transform the corners of img1 using the homography
transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

# Find the combined bounding box
all_corners = np.concatenate((transformed_corners_img1, corners_img2), axis=0)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]

# Adjust homography to account for translation
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
H_adjusted = H_translation @ H

# Warp image 1 to the perspective of image 2 with adjusted homography
warped_img1 = cv2.warpPerspective(img1, H_adjusted, (x_max - x_min, y_max - y_min))

# Create a canvas to hold both images
canvas = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

# Place image 2 on the canvas
canvas[translation_dist[1]:height2 + translation_dist[1], translation_dist[0]:width2 + translation_dist[0]] = img2

# Overlay the warped image 1 on the canvas
mask_warped_img1 = (warped_img1 > 0)
canvas[mask_warped_img1] = warped_img1[mask_warped_img1]

# Save the matches and warped images
cv2.imwrite("/home/jay/ws/VIS_IR/dh_new/output/matches_7.jpg", img_matches)
cv2.imwrite("/home/jay/ws/VIS_IR/dh_new/output/warped_img_5.jpg", warped_img1)
cv2.imwrite("/home/jay/ws/VIS_IR/dh_new/output/panorama_5.jpg", canvas)
