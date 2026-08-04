import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from external.DPVO.dpvo.dpvo import DPVO
from external.DPVO.dpvo.lietorch import SE3

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

            frame = frame.to(self.device)
            intrinsics = intrinsics.to(self.device)
            
            self.model(frame_idx, frame, intrinsics)

            camera_rot, camera_trans = self._extract_latest_dpvo_pose()

            return camera_rot, camera_trans

    def _extract_latest_dpvo_pose(self):
        dpvo = self.model

        # DPVO has not yet accepted any frame.
        if dpvo.n == 0:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # `n` is the number of valid DPVO keyframes. The newest one is n - 1.
        latest_pose_cw = dpvo.pg.poses_[dpvo.n - 1]

        # DPVO stores its internal pose in the opposite convention to the
        # camera-to-world transform needed by Gaussian3DLift:
        # world_point = R_wc @ camera_point + t_wc
        latest_pose_wc = SE3(latest_pose_cw).inv().data.detach().cpu().numpy()

        translation = latest_pose_wc[:3]
        quaternion_xyzw = latest_pose_wc[3:]

        rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()

        return rotation.astype(np.float32), translation.astype(np.float32)

    def visualize(self, frame, camera_rot, camera_trans, output):
        """
        Renders frame + camera pose info (minimap & text) and writes directly to PNG.

        Args:
            frame (np.ndarray): 3xHxW uint8 RGB image array.
            camera_trans (np.ndarray): 3-element translation vector [tx, ty, tz].
            camera_rot (np.ndarray): 3x3 rotation matrix.
        """
        frame_np = frame.detach().cpu().numpy()
        frame_np = np.transpose(frame_np, (1, 2, 0))
        # 1. Convert RGB to BGR for OpenCV rendering and saving 
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