import cv2  # OpenCV Library: Library that helps with Computer Vision Tasks
import numpy as np # Numpy: Allows to do math on arrays
import torch # Pytorch: Helps with dealing with Tensors and Neural Networks
from lightglue import LightGlue, SuperPoint, DISK  # Lightglue Library: Importing the Extractor: Superpoint, and the Matcher: Lightglue
from lightglue.utils import load_image, rbd # load_image: Transforms images into Tensors. rbd: removes batch dimenssion created during neural network
from lightglue import viz2d # viz2d: helps visualize images
import matplotlib.pyplot as plt # help visualize images
from pathlib import Path # Helps with paths to folders
from PIL import Image # Allows for image transformation.

def resize_to_smaller(image1_path, image2_path, output_path):
    """
    Resize the larger image to the same size as the smaller image and save the resized images.

    Parameters:
    - image1_path: Path to the first image file.
    - image2_path: Path to the second image file.
    - output_path: Path to save the resized images.
    """
    # The use of Pillow for this function is to keep the image as a jpg
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    width1, height1 = image1.size
    width2, height2 = image2.size


    if height1 * width1 > height2 * width2:
        resized_image1 = image1.resize((width2, height2), Image.Resampling.LANCZOS)
        resized_image1.save(output_path / 'image0_rz.jpg')
        image2.save(output_path / 'image1_rz.jpg')
    else:
        resized_image2 = image2.resize((width1, height1 ), Image.Resampling.LANCZOS)
        image1.save(output_path / 'image0_rz.jpg')
        resized_image2.save(output_path / 'image1_rz.jpg')

torch.set_grad_enabled(False) # dissable gradient computation since we are not training the model

images_path = Path("assets")  # Set "assets" as the path

# Load extractor and matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sets the device where the tensor would be evaluated
extractor = SuperPoint(max_num_keypoints=4096).eval().to(device) # Sets the max amount of keypoints and the device to evaluate
matcher = LightGlue(features="superpoint").eval().to(device) # Lightglue is set as the matcher, and superpoint as the features


# Load images for resizing step
image0_load = images_path / "IR_3.jpg"
image1_load = images_path / "VIS_3.jpg"

resize_to_smaller(image0_load, image1_load, images_path)


# Transform image into a PyTorch Tensor, for LightGlue Neural Network
image0 = load_image(images_path / "image0_rz.jpg")
image1 = load_image(images_path / "image1_rz.jpg")


# Display images to ensure they are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# Adjust the dimensions from C,H,W to H,W,C
plt.imshow(image0.permute(1, 2, 0).cpu().numpy())  
plt.title('Image 0')
plt.subplot(1, 2, 2)
plt.imshow(image1.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1')
plt.savefig("assets/images/loaded_images.png")
plt.show()


# Extract features from both images
# Extractor is superpoint, this one extract the features from the images
feats0 = extractor.extract(image0.to(device)) # .to(device) moves the tensor to the device (CPU/GPU)
# Both the neural network and the input data needs to be in the same device
feats1 = extractor.extract(image1.to(device))

# Extractor Neural Network adds the batch dimension
matches01 = matcher({"image0": feats0, "image1": feats1}) # This is lightglue matching the features
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] # Remove batch dimension [1, C, H, W] into [C, H, W]

# Extract keypoints and matches
kpts0 = feats0["keypoints"]  # Keypoints from the first image
kpts1 = feats1["keypoints"]  # Keypoints from the second image
matches = matches01["matches"]  # Indices of matched keypoints
scores = matches01["scores"]  # Confidence scores for each match

# Create a boolean mask where scores are greater than 0.9
good_match_mask = scores > 0.6  # This will be a boolean tensor of shape [N_matches]

# Apply the mask to matches and scores
filtered_matches = matches[good_match_mask]
filtered_scores = scores[good_match_mask]

# Retrieve the keypoints corresponding to the filtered matches
m_kpts0 = kpts0[filtered_matches[:, 0]]
m_kpts1 = kpts1[filtered_matches[:, 1]]

# Optionally, print the number of good matches and their scores
print(f'Number of good matches: {filtered_matches.shape[0]}')
print(f'Filtered match scores: {filtered_scores}')



# print(torch.sort(matches, dim=0))

# Display the number of matches
print(f'Number of matches: {matches.shape[0]}') # Returns the amount of rows for this array


#Visualize the matches
axes = viz2d.plot_images([image0, image1]) # It allows to visualize 2D data
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2) # Visualizes the matches
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
plt.savefig("assets/images/matched_keypoints.png")

# Visualize keypoints
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"]) # Visualize the points that were pruned
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)


#Show images
plt.show()

# Extract matched keypoints for homography
src_pts = kpts0[filtered_matches[:, 0]].cpu().numpy()
dst_pts = kpts1[filtered_matches[:, 1]].cpu().numpy()


# Compute the homography matrix
if len(matches) > 4:
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) #This 5 might be affecting the results
    print(f'Homography matrix:\n{H}')
else:
    raise ValueError("Not enough matches to compute homography.")


# Convert images to NumPy arrays
image0_np = image0.permute(1, 2, 0).cpu().numpy()
image1_np = image1.permute(1, 2, 0).cpu().numpy()

# Warp the entire image0 to the plane of image1 using the homography matrix
height0, width0 = image0_np.shape[:2]
height1, width1 = image1_np.shape[:2]
print(f"height0: {height0}, width0: {width0}")
print(f"height1: {height1}, width1: {width1}")

# Get the canvas size that can hold both images
corners_img0 = np.float32([[0, 0], [0, height0], [width0, height0], [width0, 0]]).reshape(-1, 1, 2)
corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)

# Transform the corners of img0 using the homography
transformed_corners_img0 = cv2.perspectiveTransform(corners_img0, H)

# Find the combined bounding box
all_corners = np.concatenate((transformed_corners_img0, corners_img1), axis=0)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation_dist = [-x_min, -y_min]

# Adjust homography to account for translation
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
H_adjusted = H_translation @ H

# Warp image 0 to the perspective of image 1 with adjusted homography
warped_img0 = cv2.warpPerspective(image0_np, H_adjusted, (width1, height1))

# Blending the warped image with the second image using alpha blending
alpha = 0.5  # blending factor
blended_image = cv2.addWeighted(warped_img0, alpha, image1_np, 1 - alpha, 0)

# Display the blended image

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,100)
fontScale              = 3
fontColor              = (255,255,255)
thickness              = 3
lineType               = 2

cv2.putText(blended_image,'VIS-IR', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)


cv2.imshow('Blended Image', blended_image)
cv2.imwrite('assets/images/image_fusion.png', 255*blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


