import torch
import numpy as np
import os
import cv2
from rfdetr.assets.coco_classes import COCO_CLASSES
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class Gaussian3DLift:
    def __init__(self):
        pass

    def gaussian_lift_points(self, points_list, depth, focal_length, optical_center, camera_rot, camera_pos):
        """
        X-right, Y-up, Z-forward
        """

        # If no camera intrinsics, assume camera is origin of world coordinate system
        # Only works if analyzing per-frame and not building unified scene graph
        if camera_rot is None:
            camera_rot = np.eye(3)
        if camera_pos is None:
            camera_pos = np.zeros(3)

        # X-right, Y-down, Z-forward --> X-right, Y-up, Z-forward
        camera_to_canonical = np.diag([1.0, -1.0, 1.0])

        # Convert to numpy
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        depth = depth.squeeze()

        # Infer intrinsics (If no intrinsics, assume principal point is image center)
        height, width = depth.shape
        cx, cy = optical_center
        fx, fy = focal_length

        mean_2d_list = []
        cov_2d_list = []
        center_depth_2d_list = []
        point_clouds_list = []

        num_objects = len(points_list)
        for i in range(num_objects):
            points = points_list[i]

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

            # Get the depth of the center of the object
            mean_2d_int = np.round(mean_2d).astype(int)
            center_depth_2d = depth[mean_2d_int[1], mean_2d_int[0]]

            # Throw error if object center depth is invalid
            if center_depth_2d <= 0:
                raise RuntimeError(f"Object {i} returned an invalid depth for its center: {center_depth_2d=}")

            # Append to lists
            mean_2d_list.append(mean_2d)
            cov_2d_list.append(cov_2d)
            center_depth_2d_list.append(center_depth_2d)
            
            # Lift object 2D points to a 3D point cloud
            pts_int = np.clip(np.round(points).astype(int), [0, 0], [width - 1, height - 1])
            d_vals = depth[pts_int[:, 1], pts_int[:, 0]]

            # Only include points that have valid depth values
            valid_mask = d_vals > 0
            valid_pts = points[valid_mask]
            valid_d = d_vals[valid_mask]

            # Append zeros for point cloud if no valid points exist
            if len(valid_pts) == 0:
                point_clouds_list.append(np.zeros((0, 3)))
                continue

            # TODO: Understand what is happening from here downards
            
            # 1. Standard Pinhole Unprojection (Y-down)
            px = (valid_pts[:, 0] - cx) / fx
            py = (valid_pts[:, 1] - cy) / fy
            pz = np.ones_like(px)
            cam_pts = np.stack((px, py, pz), axis=-1) * valid_d[:, None] # (N, 3)
            
            # 2. Convert to Canonical Frame (Y-up)
            cam_pts_canonical = (camera_to_canonical @ cam_pts.T).T # (N, 3)
            
            # 3. Transform to World Space
            world_pts = (camera_rot @ cam_pts_canonical.T).T + camera_pos # (N, 3)
            point_clouds_list.append(world_pts)          
            
        # If no valid 3D object projections, return zeros
        if len(mean_2d_list) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), []
        
        # Convert lists to np arrays for projection
        mean_2d_np = np.array(mean_2d_list)
        cov_2d_np = np.array(cov_2d_list)        
        center_depth_2d_np = np.array(center_depth_2d_list)[:, None]

        # Unproject 2d centroid to 3d mean
        x = (mean_2d_np[:, 0] - cx) / fx
        y = (mean_2d_np[:, 1] - cy) / fy
        z = np.ones_like(x)
        image_optical_center = (np.stack((x, y, z), axis=-1) * center_depth_2d_np)[..., None]
        optical_center = camera_to_canonical[None, ...] @ image_optical_center

        # Transform to world space
        camera_pos_col = np.array(camera_pos).reshape(1, 3, 1)
        means_3d = (camera_rot[None, ...] @ optical_center + camera_pos_col).squeeze(-1)

        # Unproject 2d covariance to 3d covariance
        M = num_objects
        J = np.zeros((M, 2, 3))
        J[:, 0, 0] = fx / center_depth_2d_np[:, 0]
        J[:, 1, 1] = fy / center_depth_2d_np[:, 0]
        J[:, 0, 2] = -x * fx / (center_depth_2d_np[:, 0] ** 2)
        J[:, 1, 2] = -y * fy / (center_depth_2d_np[:, 0] ** 2)

        J_inv = np.linalg.pinv(J)

        covs_3d = J_inv @ cov_2d_np @ J_inv.transpose(0, 2, 1)

        # Depth uncertainty regularization
        covs_3d[:, 2, 2] += (covs_3d[:, 0, 0] + covs_3d[:, 1, 1]) / 2.0

        # Change covariance from image-camera coords to canonical Y-up basis, then rotate into world
        covs_3d = camera_to_canonical[None, ...] @ covs_3d @ camera_to_canonical[None, ...]
        covs_3d = camera_rot[None, ...] @ covs_3d @ camera_rot[None, ...].transpose(0, 2, 1)
        return means_3d, covs_3d, point_clouds_list

    @staticmethod
    def visualize_3d_gaussians_on_image(
        image_input,
        means_3d,
        covs_3d,
        labels,
        focal_length,
        optical_center,
        camera_rot,
        camera_pos,
        output_path,
        triplets=None,
        pred_id_to_name=None,
        
        std_scale=2.0  # k=2 corresponds to ~95% confidence ellipse
    ):
        """
        Projects 3D Gaussians back onto a 2D image plane and saves the visualized image.

        Args:
            image_input: Path to RGB .jpg/.png image, or an already loaded OpenCV uint8 image array (H, W, 3).
            means_3d: (M, 3) Array/Tensor of 3D means in world or camera space.
            covs_3d: (M, 3, 3) Array/Tensor of 3D covariance matrices.
            instances: List or array of surviving object IDs/indices.
            focal_length: Focal length
            output_path: Path string where the annotated .jpg image will be saved.
            camera_rot: Optional (3, 3) camera rotation matrix (if means/covs are in world space).
            camera_pos: Optional (3, 1) or (3,) camera translation vector.
            labels: Optional list/array of string names or class IDs matching instances.
            std_scale: Factor multiplying standard deviations (2.0 = ~95% confidence bounds).
        """
        # Load Image if a path string was passed
        img = image_input
        height, width = img.shape[:2]
        num_objects = len(labels)

        cx, cy = optical_center
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
        if camera_rot is not None and camera_pos is not None:
            R = camera_rot.detach().cpu().numpy() if torch.is_tensor(camera_rot) else np.array(camera_rot)
            T = camera_pos.detach().cpu().numpy() if torch.is_tensor(camera_pos) else np.array(camera_pos)
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

        # Convert canonical (Y-up) back to camera space (Y-down) for projection
        canonical_to_camera = np.diag([1.0, -1.0, 1.0])
        means_cam = (canonical_to_camera[None, ...] @ means_cam[..., None]).squeeze(-1)
        covs_cam = canonical_to_camera[None, ...] @ covs_cam @ canonical_to_camera[None, ...]
    
        # Seed distinct colors for each valid object
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(max(num_objects + 1, 100), 3)).tolist()

        # Initialize centroid container to save object_id: centroid location mapping
        centroids = {}
        # Iterate over each 3D Gaussian
        for idx, (mean_3d, cov_3d) in enumerate(zip(means_cam, covs_cam)):
            X, Y, Z = mean_3d

            # Ignore points behind or too close to the camera lens
            if Z <= 0.1:
                continue

            # 1. Project 3D Mean -> 2D Pixel Centroid
            u = int(np.round((X * fx / Z) + cx))
            v = int(np.round((Y * fy / Z) + cy))
            centroids[idx] = u, v

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

            color = colors[int(idx) % len(colors)]

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
            label_text = f"Obj {idx}"
            class_name = COCO_CLASSES[labels[idx]]
            label_text = f"{class_name} (#{idx})"

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

        # 2. Render Scene Graph Relationships (Arrows & Predicate Labels)
        if triplets is not None and len(triplets) > 0:
            for triplet in triplets:
                sub_id, pred, obj_id = triplet

                if sub_id in centroids and obj_id in centroids:
                    sub_centroid = centroids[sub_id]
                    obj_centroid = centroids[obj_id]

                    # Draw directed arrow from Subject to Object
                    cv2.arrowedLine(img, sub_centroid, obj_centroid, (0, 255, 255), 2, tipLength=0.03, line_type=cv2.LINE_AA)

                    # Compute midpoint for predicate label
                    mid_x = int((sub_centroid[0] + obj_centroid[0]) / 2)
                    mid_y = int((sub_centroid[1] + obj_centroid[1]) / 2)

                    if pred_id_to_name is not None:
                        pred_text = pred_id_to_name[pred]
                    else:
                        pred_text = str(pred)

                    # Draw predicate text with a black background box for legibility
                    (tw, th), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(img, (mid_x - 2, mid_y - th - 2), (mid_x + tw + 2, mid_y + 2), (0, 0, 0), -1)
                    cv2.putText(img, pred_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, img)

    @staticmethod
    def visualize_3d_gaussians_in_3d(
            means_3d,
            covs_3d,
            labels,
            output_path,
            camera_rot=None,
            camera_pos=None,
            triplets=None,
            pred_id_to_name=None,
            std_scale=1.0,
            show_camera=False,
            camera_view_mode="aligned",  # "aligned" (Egocentric), "isometric" (World), "top_down"
            aligned_view_angle=(15, -75),  # (elev, azim) offset for egocentric view
            auto_zoom=True,
            zoom_padding=0.05,
            # Default axis ranges if auto_zoom=False
            x_range=(-1.0, 1.0),  # X: Right / Left (m)
            y_range=(-1.0, 1.0),  # Y: Height Up / Down (m)
            z_range=(0.0, 2.0),  # Z: Depth Forward (m)
        ):
            """Renders 3D Gaussians as 3D ellipsoids and visualizes camera pose."""
            # Convert PyTorch tensors to NumPy if necessary
            if torch.is_tensor(means_3d):
                means_3d = means_3d.detach().cpu().numpy()
            if torch.is_tensor(covs_3d):
                covs_3d = covs_3d.detach().cpu().numpy()
    
            means_3d_orig = means_3d.copy()
            covs_3d_orig = covs_3d.copy()
    
            # Parse Camera Extrinsics
            if camera_pos is None:
                cam_pos_world = np.zeros(3)
            else:
                cam_pos_world = (
                    camera_pos.detach().cpu().numpy()
                    if torch.is_tensor(camera_pos)
                    else np.array(camera_pos)
                )
    
            if camera_rot is None:
                cam_rot_world = np.eye(3)
            else:
                cam_rot_world = (
                    camera_rot.detach().cpu().numpy()
                    if torch.is_tensor(camera_rot)
                    else np.array(camera_rot)
                )
    
            # Transformation Matrix P: Maps Canonical [X, Y, Z] -> Plot Axes [Plot_X = X, Plot_Y = Z, Plot_Z = Y]
            P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    
            # --- 1. COORDINATE FRAME SELECTION (EGOCENTRIC vs WORLD) ---
            if camera_view_mode in ["aligned", "camera", "straight"]:
                # Egocentric mode: Inverse-transform world objects back into camera frame
                R_inv = cam_rot_world.T
                if len(means_3d_orig) > 0:
                    means_local = (R_inv @ (means_3d_orig - cam_pos_world).T).T
                    covs_local = (
                        R_inv[None, ...]
                        @ covs_3d_orig
                        @ cam_rot_world[None, ...]
                    )
                else:
                    means_local = means_3d_orig
                    covs_local = covs_3d_orig
    
                means_plot = (P @ means_local.T).T if len(means_local) > 0 else means_local
                covs_plot = (
                    P[None, ...] @ covs_local @ P[None, ...].transpose(0, 2, 1)
                    if len(covs_local) > 0
                    else covs_local
                )
    
                cam_pos_plot = np.zeros(3)
                cam_rot_plot = np.eye(3)
            else:
                means_plot = (
                    (P @ means_3d_orig.T).T
                    if len(means_3d_orig) > 0
                    else means_3d_orig
                )
                covs_plot = (
                    P[None, ...] @ covs_3d_orig @ P[None, ...].transpose(0, 2, 1)
                    if len(covs_3d_orig) > 0
                    else covs_3d_orig
                )
    
                cam_pos_plot = P @ cam_pos_world
                cam_rot_plot = cam_rot_world
    
            transformed_means = {}
            num_objects = len(labels)
            if len(means_plot) > 0:
                for i, mean in enumerate(means_plot):
                    transformed_means[i] = mean
    
            # Fill entire figure canvas with 3D axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    
            # --- 2. Draw Camera Pose & Wireframe Frustum ---
            if show_camera:
                ax.scatter(
                    cam_pos_plot[0],
                    cam_pos_plot[1],
                    cam_pos_plot[2],
                    color="black",
                    s=80,
                    marker="^",
                    label="Camera Center",
                )
    
                scale = 0.25
                frustum_cam = (
                    np.array(
                        [
                            [-0.5, -0.35, 1.0],
                            [0.5, -0.35, 1.0],
                            [0.5, 0.35, 1.0],
                            [-0.5, 0.35, 1.0],
                        ]
                    ).T
                    * scale
                )
    
                if camera_view_mode in ["aligned", "camera", "straight"]:
                    frustum_target = frustum_cam
                else:
                    frustum_target = (
                        cam_rot_world @ frustum_cam
                    ) + cam_pos_world[:, None]
    
                frustum_plot = P @ frustum_target
    
                for k in range(4):
                    ax.plot(
                        [cam_pos_plot[0], frustum_plot[0, k]],
                        [cam_pos_plot[1], frustum_plot[1, k]],
                        [cam_pos_plot[2], frustum_plot[2, k]],
                        color="black",
                        linewidth=1.2,
                    )
    
                rect = np.append(
                    frustum_plot, frustum_plot[:, :1], axis=1
                )
                ax.plot(rect[0], rect[1], rect[2], color="black", linewidth=1.2)
    
                axes_cam = np.eye(3) * (scale * 1.2)
                if camera_view_mode in ["aligned", "camera", "straight"]:
                    axes_target = axes_cam
                else:
                    axes_target = (cam_rot_world @ axes_cam) + cam_pos_world[:, None]
    
                axes_plot = P @ axes_target
                colors_rgb = ["red", "green", "blue"]
                for k in range(3):
                    ax.quiver(
                        cam_pos_plot[0],
                        cam_pos_plot[1],
                        cam_pos_plot[2],
                        axes_plot[0, k] - cam_pos_plot[0],
                        axes_plot[1, k] - cam_pos_plot[1],
                        axes_plot[2, k] - cam_pos_plot[2],
                        color=colors_rgb[k],
                        linewidth=1.5,
                        arrow_length_ratio=0.15,
                    )
    
            # --- 3. Render Object Ellipsoids ---
            if len(means_plot) > 0:
                np.random.seed(42)
                colors = plt.cm.tab20(np.linspace(0, 1, max(num_objects, 20)))
    
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = np.outer(np.cos(u), np.sin(v))
                y_sphere = np.outer(np.sin(u), np.sin(v))
                z_sphere = np.outer(np.ones_like(u), np.cos(v))
                unit_sphere = np.stack([x_sphere, y_sphere, z_sphere], axis=0)
    
                for idx, (mean_3d, cov_3d) in enumerate(zip(means_plot, covs_plot)):
                    evals, evecs = np.linalg.eigh(cov_3d)
                    evals = np.maximum(evals, 1e-6)
                    radii = std_scale * np.sqrt(evals)
    
                    ellipsoid = np.zeros_like(unit_sphere)
                    for i in range(20):
                        for j in range(20):
                            pt = unit_sphere[:, i, j]
                            ellipsoid[:, i, j] = evecs @ (radii * pt) + mean_3d
    
                    c = colors[int(idx) % len(colors)]
    
                    ax.plot_wireframe(
                        ellipsoid[0],
                        ellipsoid[1],
                        ellipsoid[2],
                        color=c,
                        alpha=0.4,
                        linewidth=0.8,
                        rstride=1,
                        cstride=1,
                    )
    
                    ax.scatter(
                        mean_3d[0],
                        mean_3d[1],
                        mean_3d[2],
                        color=c,
                        s=40,
                        depthshade=False,
                    )
    
                    lbl_id = labels[idx]
                    class_name = COCO_CLASSES[lbl_id]
                    label_text = f"{class_name} (#{idx})"
                    ax.text(
                        mean_3d[0],
                        mean_3d[1],
                        mean_3d[2],
                        label_text,
                        fontsize=8,
                        color="black",
                    )
    
            # --- DEDUPLICATE TRIPLETS ---
            unique_triplets = []
            seen_pairs = set()
            if triplets is not None:
                for triplet in triplets:
                    sub_id, pred, obj_id = triplet[0], triplet[1], triplet[2]
                    pair_key = (min(sub_id, obj_id), max(sub_id, obj_id))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        unique_triplets.append(triplet)
    
            # --- 4. Render Clutter-Free 3D Relation Curves & Labels ---
            if len(unique_triplets) > 0:
                REL_COLORS = {
                    "on": "#2ca02c",  # Green
                    "near": "#ff7f0e",  # Orange
                    "under": "#1f77b4",  # Blue
                    "default": "#d62728",  # Red
                }
    
                for triplet in unique_triplets:
                    sub_id, pred, obj_id = triplet[0], triplet[1], triplet[2]
    
                    if sub_id in transformed_means and obj_id in transformed_means:
                        p_sub = transformed_means[sub_id]
                        p_obj = transformed_means[obj_id]
    
                        d = p_obj - p_sub
                        dist = np.linalg.norm(d)
                        if dist < 1e-5:
                            continue
    
                        d_norm = d / dist
    
                        ref = (
                            np.array([0.0, 1.0, 0.0])
                            if abs(d_norm[1]) < 0.9
                            else np.array([1.0, 0.0, 0.0])
                        )
                        n = np.cross(d_norm, ref)
                        n = n / np.linalg.norm(n)
    
                        curve_amplitude = 0.10 * dist
                        p_mid = (p_sub + p_obj) / 2.0
                        p_control = p_mid + n * curve_amplitude
    
                        t = np.linspace(0, 1, 15)[:, None]
                        curve_pts = (
                            (1 - t) ** 2 * p_sub
                            + 2 * (1 - t) * t * p_control
                            + t**2 * p_obj
                        )
    
                        pred_name = pred_id_to_name[pred]
    
                        color = REL_COLORS.get(
                            pred_name.lower(), REL_COLORS["default"]
                        )
                        linestyle = (
                            "--" if pred_name.lower() in ["near", "next to"] else "-"
                        )
    
                        ax.plot(
                            curve_pts[:, 0],
                            curve_pts[:, 1],
                            curve_pts[:, 2],
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.4,
                            alpha=0.85,
                        )
    
                        label_pos = p_control + n * 0.03
                        ax.text(
                            label_pos[0],
                            label_pos[1],
                            label_pos[2],
                            pred_name,
                            fontsize=7,
                            color=color,
                            weight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                facecolor="white",
                                edgecolor=color,
                                alpha=0.7,
                                linewidth=0.5,
                            ),
                        )
    
            # --- 5. DYNAMIC AUTO-ZOOM BOUNDING BOX COMPUTATION ---
            if auto_zoom and len(means_plot) > 0:
                # Crop tightly around ONLY active objects (ignoring camera origin)
                min_pts = np.min(means_plot, axis=0) - zoom_padding
                max_pts = np.max(means_plot, axis=0) + zoom_padding

                x_range = (min_pts[0], max_pts[0])
                z_range = (min_pts[1], max_pts[1])
                y_range = (min_pts[2], max_pts[2])

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([]) 

            # --- 6. Lock Axis Limits & Maximize Plot Fill ---
            if x_range is not None:
                ax.set_xlim(x_range)
            if z_range is not None:
                ax.set_ylim(z_range)  # Plot Y = Depth Z
            if y_range is not None:
                ax.set_zlim(y_range)  # Plot Z = Height Y

            # DO NOT enforce fixed 1:1:1 box aspect in auto_zoom mode
            # Letting Matplotlib stretch box aspect allows the viewport to maximize canvas fill!
            if not auto_zoom and x_range and y_range and z_range:
                dx = x_range[1] - x_range[0]
                dy = y_range[1] - y_range[0]
                dz = z_range[1] - z_range[0]
                ax.set_box_aspect([dx, dz, dy])

            # --- 7. SET AXIS LABELS & VIEW INIT ---
            if camera_view_mode in ["aligned", "camera", "straight"]:
                ax.set_xlabel("X / Rel Right (m)", fontsize=7.5)
                ax.set_ylabel("Z / Rel Depth (m)", fontsize=7.5)
                ax.set_zlabel("Y / Rel Height (m)", fontsize=7.5)
                ax.set_title("3D Scene Graph (Egocentric Camera View)", fontsize=10)

                elev_offset, azim_offset = aligned_view_angle
                ax.view_init(elev=elev_offset, azim=azim_offset)
                                
                # Zoom Matplotlib's 3D internal camera directly onto the target objects
                ax.dist = 6  # Lower distance (default is 10) zooms in closer to fill canvas

            elif camera_view_mode in ["top_down", "birds_eye"]:
                ax.set_xlabel("X / World Right (m)", fontsize=7.5)
                ax.set_ylabel("Z / World Depth (m)", fontsize=7.5)
                ax.set_zlabel("Y / World Height (m)", fontsize=7.5)
                ax.set_title("3D Scene Graph (Bird's Eye View)", fontsize=10)

                ax.view_init(elev=90, azim=-90)
                ax.set_zticks([])
                ax.set_zlabel("")
                ax.dist = 7
            else:
                ax.set_xlabel("X / World Right (m)", fontsize=7.5)
                ax.set_ylabel("Z / World Depth (m)", fontsize=7.5)
                ax.set_zlabel("Y / World Height (m)", fontsize=7.5)
                ax.set_title("3D Scene Graph (World Frame View)", fontsize=10)

                ax.view_init(elev=20, azim=-60)
                ax.dist = 8

            # Save to JPG with tight padding
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0, format="jpeg")
            plt.close(fig)