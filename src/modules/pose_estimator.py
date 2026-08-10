import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from external.DPVO.dpvo.dpvo import DPVO
from external.DPVO.dpvo.lietorch import SE3
from external.unidepth.unidepth.models.unidepthv2 import UniDepthV2

class PoseEstimator:
    def __init__(self, device, model):
        self.device = device
        self.model = model

        self.trajectory_history = []
        self.minimap_size = 180

    def process_frame(self, frame, frame_idx, focal_length, optical_center):

        intrinsics = torch.tensor([focal_length[0], focal_length[1], optical_center[0], optical_center[1]])

        if isinstance(self.model, DPVO):

            frame = frame.to(self.device)
            intrinsics = intrinsics.to(self.device)
            
            self.model(frame_idx, frame, intrinsics)

            camera_rot, camera_trans = self._extract_latest_dpvo_pose()

            return camera_rot, camera_trans

    def get_metric_scaling(self, t, depth):
        """
        Get the metric scaling based on what the pose model and depth models are
        """
        # Scaling from DPVO metric
        if isinstance(self.model, DPVO):

            # Get the patch for frame t
            frame_patches = self.model.pg.patches_[t].cpu().numpy() # shape: (M, 3, 3, 3)

            # Get the coordinates for the pixels from the depth estimator
            pose_estimator_coords = frame_patches[:, :2, 1, 1].astype(int) # shape: (M, 2, 3, 3), 3x3 pixel patches for every patch m in M, (u,v)
            # Get the depth d of each patch with coords [u,v,d] across 3x3 pixel patches
            pixel_depths_inv = frame_patches[:, 2, 1, 1] # shape: (M, 3, 3), 3x3 pixel patches for every patch m in M
            pose_estimator_depths = 1.0 / (pixel_depths_inv + 1e-8)

            # Get the corresponding depths from the given depth map (from depth estimator)
            depth_np = depth.detach().cpu().numpy()
            depth_estimator_depths = depth_np[pose_estimator_coords[:, 1], pose_estimator_coords[:, 0]]

            # Get the ratios
            ratios = depth_estimator_depths / pose_estimator_depths

            # Get the median-average ratio
            scale = float(np.median(ratios))

            return scale

    def _extract_latest_dpvo_pose(self):
        # 1. Guard against DPVO having no active frames
        if not hasattr(self.model, "n") or self.model.n == 0:
            return np.eye(3), np.zeros(3)

        # 2. Get pose at latest active frame index
        latest_idx = max(0, self.model.n - 1)
        pose_7d = self.model.pg.poses_[latest_idx].detach().cpu().numpy()

        camera_trans = pose_7d[:3]
        quaternion_xyzw = pose_7d[3:]  # [qx, qy, qz, qw]

        # 3. Check for zero-norm quaternion (uninitialized DPVO buffer slot)
        quat_norm = np.linalg.norm(quaternion_xyzw)

        if quat_norm < 1e-5:
            # Fallback to Identity rotation if DPVO has not set the pose yet
            camera_rot = np.eye(3)
            camera_trans = np.zeros(3)
        else:
            # Normalize to avoid numerical drift and convert to 3x3 rotation matrix
            quaternion_normalized = quaternion_xyzw / quat_norm
            camera_rot = Rotation.from_quat(quaternion_normalized).as_matrix()

        return camera_rot, camera_trans

    def visualize(self, frame, camera_rot, camera_trans, output):
        frame_np = frame.detach().cpu().numpy()
        frame_np = np.transpose(frame_np, (1, 2, 0))
        
        # 1. Guard against float32 tensors rendering black
        if frame_np.dtype == np.float32 or frame_np.dtype == np.float64:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
                
        # Ensure memory layout is contiguous for OpenCV
        frame_np = np.ascontiguousarray(frame_np)

        # 2. Convert RGB to BGR for OpenCV rendering and saving 
        vis_img = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        H, W, _ = vis_img.shape

        # 2. Append camera translation position to global history
        self.trajectory_history.append(camera_trans.copy())

        # 3. Render Top-Down Trajectory Inset (X-Z plane view)
        minimap = np.zeros((self.minimap_size, self.minimap_size, 3), dtype=np.uint8)
        traj_arr = np.array(self.trajectory_history)

        if len(traj_arr) > 1:
            xs = traj_arr[:, 0]
            zs = traj_arr[:, 2]  # Z represents depth

            # Normalize coordinates to map onto minimap canvas with 15px padding
            x_min, x_max = xs.min(), xs.max()
            z_min, z_max = zs.min(), zs.max()
            dx = (x_max - x_min) if (x_max - x_min) > 1e-5 else 1.0
            dz = (z_max - z_min) if (z_max - z_min) > 1e-5 else 1.0

            pxs = ((xs - x_min) / dx * (self.minimap_size - 30) + 15).astype(int)
            pzs = ((zs - z_min) / dz * (self.minimap_size - 30) + 15).astype(int)

            # Draw trajectory path line (Cyan)
            for i in range(1, len(pxs)):
                cv2.line(minimap, (pxs[i - 1], pzs[i - 1]), (pxs[i], pzs[i]), (255, 255, 0), 2)

            # Draw current camera position marker (Red)
            cv2.circle(minimap, (pxs[-1], pzs[-1]), 5, (0, 0, 255), -1)

        # Overlay minimap in top-right corner
        vis_img[10:10 + self.minimap_size, W - 10 - self.minimap_size:W - 10] = minimap

        # 4. Extract rotation angles (Euler Yaw, Pitch, Roll in degrees)
        euler_deg = Rotation.from_matrix(camera_rot).as_euler('xyz', degrees=True)

        # 5. Stamp numeric pose info onto top-left corner
        text_pos = f"Pos [X,Y,Z]: [{camera_trans[0]:.2f}, {camera_trans[1]:.2f}, {camera_trans[2]:.2f}] m"
        text_rot = f"Rot [Y,P,R]: [{euler_deg[0]:.1f}deg, {euler_deg[1]:.1f}deg, {euler_deg[2]:.1f}deg]"
        text_uninitialized = f'Not Initialized'

        cv2.putText(vis_img, text_pos, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vis_img, text_rot, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if not self.model.is_initialized:
            cv2.putText(vis_img, text_uninitialized, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 6. Save zero-padded PNG directly to disk (No GUI window required) 
        cv2.imwrite(output, vis_img)