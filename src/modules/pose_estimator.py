import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from external.DPVO.dpvo.dpvo import DPVO

class PoseEstimator:
    def __init__(self, device, model):
        self.device = device
        self.model = model

        self.trajectory_history = []
        self.minimap_size = 180

    def process_frame(self, frame, frame_idx, intrinsics):

        if isinstance(self.model, DPVO):

            # Convert intrinsics to np array
            if not isinstance(intrinsics, torch.Tensor):
                intrinsics = torch.tensor(intrinsics)

            intrinsics = intrinsics.to(self.device)
            
            self.model(frame_idx, frame, intrinsics)

            camera_rot, camera_trans = self._extract_latest_dpvo_pose()

            return camera_rot, camera_trans

    def _extract_latest_dpvo_pose(self):
        """
        Extracts the most recent 6-DoF pose matrix [R | t] from DPVO's internal state.
        """
        latest_pose = None

        # Extract latest pose tensor/array from DPVO internal storage
        if hasattr(self.model, 'poses_') and len(self.model.poses_) > 0:
            latest_pose = self.model.poses_[-1]
            if isinstance(latest_pose, torch.Tensor):
                latest_pose = latest_pose.cpu().numpy()
        elif hasattr(self.model, 'pg') and hasattr(self.model.pg, 'poses_'):
            latest_pose = self.model.pg.poses_[-1].cpu().numpy()

        # Fallback for initial frame before SLAM optimizes pose
        if latest_pose is None:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # Case A: Pose is stored as 7D Lie / Quaternion vector [tx, ty, tz, qx, qy, qz, qw]
        if latest_pose.ndim == 1 and latest_pose.shape[0] == 7:
            trans = latest_pose[:3]
            quat = latest_pose[3:]  # [qx, qy, qz, qw]
            rot = Rotation.from_quat(quat).as_matrix()

        # Case B: Pose is stored directly as a 4x4 homogenous matrix
        elif latest_pose.shape == (4, 4):
            rot = latest_pose[:3, :3]
            trans = latest_pose[:3, 3]

        else:
            rot = np.eye(3, dtype=np.float32)
            trans = np.zeros(3, dtype=np.float32)

        return rot.astype(np.float32), trans.astype(np.float32)

    def visualize_frame(self, frame, camera_trans, camera_rot, output):
        """
        Renders frame + camera pose info (minimap & text) and writes directly to PNG.

        Args:
            frame (np.ndarray): 3xHxW uint8 RGB image array.
            camera_trans (np.ndarray): 3-element translation vector [tx, ty, tz].
            camera_rot (np.ndarray): 3x3 rotation matrix.
        """
        # 1. Convert RGB to BGR for OpenCV rendering and saving 
        vis_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, H, W = vis_img.shape

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

        cv2.putText(vis_img, text_pos, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vis_img, text_rot, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 6. Save zero-padded PNG directly to disk (No GUI window required) 
        cv2.imwrite(output, vis_img)