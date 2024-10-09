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
image0_load = images_path / "VIS_1.jpg"
image1_load = images_path / "IR_1.jpg"

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

# Parameters
focal_length = 800  # Example focal length; adjust as per your camera
# height0, width0 = image0.shape[:2]
image_width, image_height = 1280, 1024  # Example image dimensions in pixels
sensor_width, sensor_height = 36, 24  # Example sensor dimensions in mm

# Define the field of view angles (in radians)
alpha_h = 2 * np.arctan(sensor_width / (2 * focal_length))
alpha_v = 2 * np.arctan(sensor_height / (2 * focal_length))

# Initial rotation (identity) and translation (origin)
R_initial = np.eye(3)
t_initial = np.zeros((3, 1))

# Define ground plane characteristics
ground_plane_normal = np.array([0, 0, 1])  # Flat ground plane normal vector
ground_plane_point = np.array([0, 0, 0])  # A point on the ground plane

print("Focal Length:", focal_length)
print("Image Width:", image_width)
print("Image Height:", image_height)
print("R_initial:\n", R_initial)
print("t_initial:\n", t_initial)
print("Ground Plane Normal:", ground_plane_normal)
print("Ground Plane Point:", ground_plane_point)


# New Function Integrations
def camera_to_world_matrix(R, t):
    Rt = -R.T @ t
    return np.vstack((np.hstack((R.T, Rt)), [0, 0, 0, 1]))

def image_plane_corners(f, w, h):
    return [
        np.array([-w / 2, -h / 2, f, 1]),
        np.array([ w / 2, -h / 2, f, 1]),
        np.array([-w / 2,  h / 2, f, 1]),
        np.array([ w / 2,  h / 2, f, 1]),
    ]

def project_to_ground(R, t, ground_plane_normal, ground_plane_point):
    P_inv = camera_to_world_matrix(R, t)
    corners = image_plane_corners(focal_length, image_width, image_height)
    ground_intersections = []

    for corner in corners:
        world_point = P_inv @ corner
        ray_direction = (world_point[:3] - t.flatten())
        ray_direction /= np.linalg.norm(ray_direction)

        t_intersect = np.dot((ground_plane_point - t.flatten()), ground_plane_normal) / np.dot(ray_direction, ground_plane_normal)
        intersection_point = t.flatten() + t_intersect * ray_direction
        ground_intersections.append(intersection_point)
    
    return np.array(ground_intersections)

def adjust_overlap(ground_points_i, overlap_x, overlap_y):
    overlap_point_x = ground_points_i[0] + overlap_x * (ground_points_i[1] - ground_points_i[0])
    overlap_point_y = ground_points_i[2] + overlap_y * (ground_points_i[3] - ground_points_i[2])
    return overlap_point_x, overlap_point_y

def calculate_d_prime(ground_points_j, overlap_point_x, overlap_point_y):
    rho = ground_points_j[0] - overlap_point_x
    rho_norm = rho / np.linalg.norm(rho)
    angle_vertical = alpha_v / 2
    length_CjBj = np.linalg.norm(ground_points_j[0] - t_initial.flatten())
    angle_CjBjDj = np.arccos(np.dot(-rho_norm, (ground_points_j[0] - t_initial.flatten()) / length_CjBj))
    length_BjDj = np.sin(angle_vertical) * length_CjBj / np.sin(np.pi - angle_vertical - angle_CjBjDj)
    D_prime_j = ground_points_j[0] + rho_norm * length_BjDj
    return D_prime_j

def calculate_rotation_translation(overlap_point_x, D_prime_j):
    z_prime = (overlap_point_x - t_initial.flatten()) / np.linalg.norm(overlap_point_x - t_initial.flatten())
    y_axis = (D_prime_j - overlap_point_x) / np.linalg.norm(D_prime_j - overlap_point_x)
    x_axis = np.cross(y_axis, z_prime)
    R_new = np.vstack((x_axis, y_axis, z_prime)).T
    t_new = -R_new @ t_initial.flatten()
    return R_new, t_new

# Apply Light-field Dynamic Homography
ground_points = project_to_ground(R_initial, t_initial, ground_plane_normal, ground_plane_point)
overlap_point_x, overlap_point_y = adjust_overlap(ground_points, overlap_x=0.5, overlap_y=0.5)
D_prime_j = calculate_d_prime(ground_points, overlap_point_x, overlap_point_y)
R_new, t_new = calculate_rotation_translation(overlap_point_x, D_prime_j)

print("New Rotation Matrix (R_new):\n", R_new)
print("New Translation Vector (t_new):\n", t_new)

# Function to apply rotation and translation to keypoints
def apply_rotation_translation(keypoints, R, t):
    # Convert keypoints to homogeneous coordinates (adding a 1 for each point)
    keypoints_homog = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    
    # Apply the rotation and translation
    transformed_points = (R @ keypoints_homog[:, :3].T).T + t.T
    
    # Convert back to inhomogeneous coordinates
    return transformed_points[:, :2] / transformed_points[:, 2:3]

# Extract the original keypoints
src_pts = np.array([kpt.cpu().numpy() for kpt in m_kpts0])  # VIS keypoints in original image
dst_pts = np.array([kpt.cpu().numpy() for kpt in m_kpts1])  # IR keypoints in target image

# Apply the new rotation and translation to the VIS keypoints
transformed_src_pts = apply_rotation_translation(src_pts, R_new, t_new)

# Compute homography based on transformed source points and destination points
H, status = cv2.findHomography(transformed_src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the VIS image to match the IR image perspective using the computed homography
image0_np = image0.permute(1, 2, 0).cpu().numpy()
image1_np = image1.permute(1, 2, 0).cpu().numpy()
warped_img0 = cv2.warpPerspective(image0_np, H, (image1_np.shape[1], image1_np.shape[0]))

# Display or blend the warped image as per your original code
alpha = 0.5
blended_image = cv2.addWeighted(warped_img0, alpha, image1_np, 1 - alpha, 0)

# Display the final blended result
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.title('Blended Image with Light-field Dynamic Homography')
plt.savefig("assets/images/blended_image_dynamic_homography.png")
plt.show()


# # Display the blended image

# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10,100)
# fontScale              = 3
# fontColor              = (255,255,255)
# thickness              = 3
# lineType               = 2

# cv2.putText(blended_image,'VIS-IR', 
#     bottomLeftCornerOfText, 
#     font, 
#     fontScale,
#     fontColor,
#     thickness,
#     lineType)


# cv2.imshow('Blended Image', blended_image)
# cv2.imwrite('assets/images/image_fusion.png', 255*blended_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


