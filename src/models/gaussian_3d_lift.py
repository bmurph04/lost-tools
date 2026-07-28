import torch
import numpy as np
import os
import cv2
from rfdetr.assets.coco_classes import COCO_CLASSES

class Gaussian3DLift:
    def __init__(self, visualize=False):
        pass

    def gaussian_lift_points(self, objects_info, depth, focal_length_px, camera_rot=None, camera_trans=None):
        """
        
        points - (N, 2)
        """

        # If no camera intrinsics, assume camera is origin of world coordinate system
        # Only works if analyzing per-frame and not building unified scene graph
        if camera_rot == None:
            camera_rot = np.eye(3)
        if camera_trans == None:
            camera_trans = np.zeros(3)

        # Convert to numpy
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        depth = depth.squeeze()

        # Infer intrinsics (If no intrinsics, assume principal point is image center)
        height, width = depth.shape
        cx, cy = width / 2.0, height / 2.0
        fx, fy = focal_length_px

        mean_2d_list = []
        cov_2d_list = []
        depth_center_list = []
        valid_object_instances = []

        num_objects = len(objects_info['points'])
        for i in range(num_objects):
            points = objects_info['points'][i]
            object_point_count = objects_info['object_point_counts'][i]
            class_id = objects_info['class_ids'][i]
            confidence = objects_info['confidences'][i]

            # If no points exist for this object, skip
            if len(points) == 0:
                continue

            # Convert to numpy
            if torch.is_tensor(points):
                points = points.detach().cpu().numpy()

            # Compute 2D mean from tracked points
            mean_2d = np.mean(points, axis=0)

            # Compute 2D covariance from tracked points
            centered_points = points - mean_2d
            cov_2d = (centered_points.T @ centered_points) / len(points)

            # Get depth info
            mean_2d_int = np.round(mean_2d).astype(int)
            depth_center = depth[mean_2d_int[1], mean_2d_int[0]]

            mean_2d_list.append(mean_2d)
            cov_2d_list.append(cov_2d)
            depth_center_list.append(depth_center)
            valid_object_instances.append(i)
            
        mean_2d_np = np.array(mean_2d_list)
        cov_2d_np = np.array(cov_2d_list)        
        depth_center_np = np.array(depth_center_list)[:, None]

        # Unproject 2d centroid to 3d mean
        x = (mean_2d_np[:, 0] - cx) / fx
        y = (mean_2d_np[:, 1] - cy) / fy
        z = np.ones_like(x)
        camera_coords = (np.stack((x, y, z), axis=-1) * depth_center_np)[..., None]

        # Transform to world space
        camera_trans_col = np.array(camera_trans).reshape(1, 3, 1)
        means_3d = (camera_rot[None, ...] @ camera_coords + camera_trans_col).squeeze(-1)

        # Unproject 2d covariance to 3d covariance
        M = len(valid_object_instances)
        J = np.zeros((M, 2, 3))
        J[:, 0, 0] = fx / depth_center_np[:, 0]
        J[:, 1, 1] = fy / depth_center_np[:, 0]
        J[:, 0, 2] = -x * fx / (depth_center_np[:, 0] ** 2)
        J[:, 1, 2] = -y * fy / (depth_center_np[:, 0] ** 2)

        J_inv = np.linalg.pinv(J)

        covs_3d = J_inv @ cov_2d_np @ J_inv.transpose(0, 2, 1)

        # Depth uncertainty regularization
        covs_3d[:, 2, 2] += (covs_3d[:, 0, 0] + covs_3d[:, 1, 1]) / 2.0

        # Rotate covariance into world orientation
        covs_3d = camera_rot[None, ...] @ covs_3d @ camera_rot[None, ...].transpose(0, 2, 1)

        return means_3d, covs_3d, valid_object_instances

    def visualize_3d_gaussians_on_image(
        self,
        image_input,
        means_3d,
        covs_3d,
        valid_indices,
        focal_length,
        output_path,
        camera_rot=None,
        camera_trans=None,
        labels=None,
        std_scale=2.0  # k=2 corresponds to ~95% confidence ellipse
    ):
        """
        Projects 3D Gaussians back onto a 2D image plane and saves the visualized image.

        Args:
            image_input: Path to RGB .jpg/.png image, or an already loaded OpenCV uint8 image array (H, W, 3).
            means_3d: (M, 3) Array/Tensor of 3D means in world or camera space.
            covs_3d: (M, 3, 3) Array/Tensor of 3D covariance matrices.
            valid_indices: List or array of surviving object IDs/indices.
            focal_length: Focal length
            output_path: Path string where the annotated .jpg image will be saved.
            camera_rot: Optional (3, 3) camera rotation matrix (if means/covs are in world space).
            camera_trans: Optional (3, 1) or (3,) camera translation vector.
            labels: Optional list/array of string names or class IDs matching valid_indices.
            std_scale: Factor multiplying standard deviations (2.0 = ~95% confidence bounds).
        """
        # Load Image if a path string was passed
        if isinstance(image_input, (str, os.PathLike)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Could not load image at path: {image_input}")
        else:
            img = image_input.copy()

        height, width = img.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        fx, fy = focal_length

        # Convert PyTorch tensors to NumPy arrays if necessary
        if torch.is_tensor(means_3d):
            means_3d = means_3d.detach().cpu().numpy()
        if torch.is_tensor(covs_3d):
            covs_3d = covs_3d.detach().cpu().numpy()

        if len(means_3d) == 0:
            cv2.imwrite(output_path, img)
            print(f"[Visualization] No Gaussians to render. Saved blank canvas to {output_path}")
            return

        # Handle World-to-Camera coordinate transformation
        if camera_rot is not None and camera_trans is not None:
            R = camera_rot.detach().cpu().numpy() if torch.is_tensor(camera_rot) else np.array(camera_rot)
            T = camera_trans.detach().cpu().numpy() if torch.is_tensor(camera_trans) else np.array(camera_trans)
            if T.ndim == 1:
                T = T[:, None]

            # Transform World space means back to Camera space: P_cam = R^T * (P_world - T)
            R_inv = R.T
            means_cam = (R_inv @ (means_3d.T - T)).T  # (M, 3)
            
            # Transform World space covariances: Cov_cam = R^T * Cov_world * R
            covs_cam = R_inv[None, ...] @ covs_3d @ R[None, ...]
        else:
            means_cam = means_3d
            covs_cam = covs_3d

        # Seed distinct colors for each valid object
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(max(len(valid_indices) + 1, 100), 3)).tolist()

        # Iterate over each 3D Gaussian
        for idx, (mean_3d, cov_3d, orig_idx) in enumerate(zip(means_cam, covs_cam, valid_indices)):
            X, Y, Z = mean_3d

            # Ignore points behind or too close to the camera lens
            if Z <= 0.1:
                continue

            # 1. Project 3D Mean -> 2D Pixel Centroid
            u = int(np.round((X * fx / Z) + cx))
            v = int(np.round((Y * fy / Z) + cy))

            # Check if projected center lies reasonably near the frame
            if not (-100 <= u <= width + 100 and -100 <= v <= height + 100):
                continue

            # 2. Construct 2D Projection Jacobian Matrix J (2 x 3)
            J = np.array([
                [fx / Z, 0.0,    -(X * fx) / (Z ** 2)],
                [0.0,    fy / Z, -(Y * fy) / (Z ** 2)]
            ])

            # 3. Project 3D Covariance -> 2D Image Plane Covariance (2 x 2)
            cov_2d = J @ cov_3d @ J.T

            # 4. Eigendecomposition of 2D Covariance for Ellipse Parameters
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)

            # Ensure positive eigenvalues for valid sqrt
            eigenvalues = np.maximum(eigenvalues, 1e-6)

            # Sort eigenvalues in descending order
            order = eigenvalues.argsort()[::-1]
            evals = eigenvalues[order]
            evecs = eigenvectors[:, order]

            # Calculate Semi-Major and Semi-Minor axis lengths in pixels
            axis_major = int(np.round(std_scale * np.sqrt(evals[0])))
            axis_minor = int(np.round(std_scale * np.sqrt(evals[1])))

            # Calculate rotation angle in degrees
            angle_rad = np.arctan2(evecs[1, 0], evecs[0, 0])
            angle_deg = int(np.round(np.degrees(angle_rad)))

            color = colors[int(orig_idx) % len(colors)]

            # 5. Draw 2D Confidence Ellipse
            cv2.ellipse(
                img,
                center=(u, v),
                axes=(axis_major, axis_minor),
                angle=angle_deg,
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # Draw Center Point
            cv2.circle(img, (u, v), radius=4, color=(0, 0, 255), thickness=-1)
        
            # Optional Text Label
            label_text = f"Obj {orig_idx}"
            if labels is not None and idx < len(labels):
                class_name = COCO_CLASSES[labels[idx]]
                label_text = f"{class_name} (#{orig_idx})"

            cv2.putText(
                img,
                label_text,
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            # cv2.putText(
            #     img,
            #     label_text,
            #     (u + 6, v - 6),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     color,
            #     1,
            #     cv2.LINE_AA
            # )

        # Ensure output directory exists and save image
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(output_path, img)