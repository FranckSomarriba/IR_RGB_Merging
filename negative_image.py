import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps

def resize_to_smaller(image1_path, image2_path, output_path):
    """
    Resize the larger image to the same size as the smaller image and save the resized images.

    Parameters:
    - image1_path: Path to the first image file.
    - image2_path: Path to the second image file.
    - output_path: Path to save the resized images.
    """
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    height1, width1 = image1.size
    height2, width2 = image2.size

    if height1 * width1 > height2 * width2:
        resized_image1 = image1.resize((height2, width2), Image.Resampling.LANCZOS)
        resized_image1.save(output_path / 'image0_rz.jpg')
        image2.save(output_path / 'image1_rz.jpg')
    else:
        resized_image2 = image2.resize((height1, width1), Image.Resampling.LANCZOS)
        image1.save(output_path / 'image0_rz.jpg')
        resized_image2.save(output_path / 'image1_rz.jpg')

def transform_grayscale(image_path, output_path):
    """
    Creates a grayscale copy of an image

    Parameters:
    - image_path: Path to the first image file.
    - output_path: Path to save the resized images.
    """

    image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Save the grayscale image to the output path
    grayscale_image.save(output_path / 'image1_gray.jpg')

def transform_to_negative(image_path, output_path):
    """
    Creates a negative copy of an image.

    Parameters:
    - image_path: Path to the input image file.
    - output_path: Path to save the negative image.
    """

    # Open the image
    image = Image.open(image_path)

    # Convert the image to its negative
    negative_image = ImageOps.invert(image)

    # Save the negative image to the output path
    negative_image.save(output_path / 'image1_negative.jpg')

torch.set_grad_enabled(False)

images_path = Path("assets")

# Load extractor and matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)


# Load images
image0_load = images_path / "IR_3.jpg"
image1_load = images_path / "VIS_3.jpg"

resize_to_smaller(image0_load, image1_load, images_path)

image0 = load_image(images_path / "image0_rz.jpg")
image1 = load_image(images_path / "image1_rz.jpg")


# Display images to ensure they are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image0.permute(1, 2, 0).cpu().numpy())
plt.title('Image 0')
plt.subplot(1, 2, 2)
plt.imshow(image1.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1')
plt.savefig("assets/images/loaded_images.png")
plt.show()

# image1_gray_load = images_path / "image1_rz.jpg"
# transform_grayscale(image1_gray_load, images_path)
# image1_negative_load = images_path / 'image1_gray.jpg'

image1_negative_load = images_path / 'image1_rz.jpg'
image0_negative_load = images_path / "image0_rz.jpg"

transform_to_negative(image1_negative_load,images_path)
image1_negative = load_image(images_path / "image1_negative.jpg")

transform_to_negative(image0_negative_load,images_path)
image0_negative = load_image(images_path / "image1_negative.jpg")


# Display negative image to ensure they are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1')
plt.subplot(1, 2, 2)
plt.imshow(image1_negative.permute(1, 2, 0).cpu().numpy())
plt.title('Image 1-negative')
plt.savefig("assets/images/negative_transform.png")
plt.show()

# Display negative image of IR to ensure they are loaded correctly
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image0.permute(1, 2, 0).cpu().numpy())
plt.title('Image 0')
plt.subplot(1, 2, 2)
plt.imshow(image0_negative.permute(1, 2, 0).cpu().numpy())
plt.title('Image 0-negative')
plt.savefig("assets/images/negative_transform_0.png")
plt.show()



# Extract features from both images
feats0 = extractor.extract(image0_negative.to(device))
feats1 = extractor.extract(image1_negative.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

# Extract keypoints and matches
kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# Display the number of matches
print(f'Number of matches: {matches.shape[0]}')

#Visualize the matches
axes = viz2d.plot_images([image0_negative, image1_negative])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
plt.savefig("assets/images/matched_keypoints_negative.png")

# Visualize keypoints
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

#Show images
plt.show()

# Extract matched keypoints for homography
src_pts = kpts0[matches[:, 0]].cpu().numpy()
dst_pts = kpts1[matches[:, 1]].cpu().numpy()


# Compute the homography matrix
if len(matches) > 4:
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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

cv2.imshow('Blended Image', blended_image)
cv2.imwrite('assets/images/image_fusion_negative.png', 255*blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
