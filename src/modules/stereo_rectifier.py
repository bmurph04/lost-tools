import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from src.utils import unity_pose_to_cv

class StereoRectifier:
    """
    Rectifies Quest 3 passthrough stereo pairs so that corresponding points
    share an image row, which is what FoundationStereo's horizontal cost
    volume requires.

    The two cameras are rigidly mounted, so the maps are built once.
    """

    def __init__(self, left_focal_length, left_optical_center, right_focal_length, right_optical_center, rel_camera_rot, rel_camera_trans, image_width, image_height):
        
        # Put image with and height into size tuple
        size = (image_width, image_height)
        
        # Normalize the relative extrinsics into the layout stereoRectify wants.
        # OpenCV 5 requires T as an explicit 3x1 column; a flat (3,) array trips a
        # gemm assertion. R arrives as a 3x3 matrix when the coordinate conversion
        # already built one (unity_pose_to_cv), but as a quaternion when no
        # conversion is applied, so accept both.
        rel_camera_rot = np.asarray(rel_camera_rot, dtype=np.float64)
        if rel_camera_rot.shape == (4,):
            rel_camera_rot = Rotation.from_quat(rel_camera_rot).as_matrix()
        elif rel_camera_rot.shape != (3, 3):
            raise ValueError(
                f"rel_camera_rot must be a 3x3 rotation matrix or a 4-element "
                f"quaternion, got shape {rel_camera_rot.shape}")

        rel_camera_trans = np.asarray(rel_camera_trans, dtype=np.float64).reshape(3, 1)


        # Intrinsics matrix for the left camera
        K_left = np.array([
            [left_focal_length[0], 0, left_optical_center[0]],
            [0, left_focal_length[1], left_optical_center[1]],
            [0, 0, 1]
            ],
            dtype=np.float64)
        
        # Intrinsics matrix for the right camera
        K_right = np.array([
            [right_focal_length[0], 0, right_optical_center[0]],
            [0, right_focal_length[1], right_optical_center[1]],
            [0, 0, 1]
            ],
            dtype=np.float64)

        # Distortion coefficients (all 0 for Quest 3 passthrough, rectilinear)
        D = np.zeros(5)

        # stereoRectify returns the following:
        # R1, R2 - rotation matricies from real camera frame to virtual camera frame (R1=left, R2=right),
        # P1, P2 - projection matrices representing rectified intrinsics (exact same between P1 and P2).
            # P2 has extra column with element -f * B in the first row, where B is baseline (sideways offset)   
        # Q - 4x4 reprojection matrix that transforms 3D point into rectified camera1 frame (left camera)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=K_left, 
            distCoeffs1=D, 
            cameraMatrix2=K_right,
            distCoeffs2=D,
            imageSize=size,
            R=rel_camera_rot,
            T=rel_camera_trans,
            flags=cv2.CALIB_ZERO_DISPARITY, 
            alpha=0)

        # Create mapping that will map points in real left camera to rectified virtual left camera
        # Initialized once and used forever since it will stay constant
        self._map_left = cv2.initUndistortRectifyMap(
            cameraMatrix=K_left,
            distCoeffs=D,
            R=R1,
            newCameraMatrix=P1,
            size=size,
            m1type=cv2.CV_16SC2
        )

        # Create mapping that will map points in real left camera to rectified virtual left camera
        # Initialized once and used forever since it will stay constant

        self._map_right = cv2.initUndistortRectifyMap(
            cameraMatrix=K_right,
            distCoeffs=D,
            R=R2,
            newCameraMatrix=P2,
            size=size,
            m1type=cv2.CV_16SC2
        )

        self._R1 = R1
        self.Q = Q
        self.rectified_focal_length = (float(P1[0, 0]), float(P1[1, 1]))
        self.rectified_optical_center = (float(P1[0, 2]), float(P1[1, 2]))
        self.baseline = float(-P2[0, 3] / P2[0, 0])

    def rectify_pair(self, left_frame, right_frame):
        """(3, H, W) uint8 in -> rectified (3, H, W) uint8 out."""
        # Reshape to (H, W, 3) and write frames as contiguous array
        left_frame_transformed = np.ascontiguousarray(left_frame.transpose(1, 2, 0)) 
        right_frame_transformed = np.ascontiguousarray(right_frame.transpose(1, 2, 0)) 

        # Use computed maps to rectify left and right frames
        left_rect = cv2.remap(left_frame_transformed, *self._map_left, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame_transformed, *self._map_right, interpolation=cv2.INTER_LINEAR)

        # Return rectified frames reshaped to (3, H, W)
        return left_rect.transpose(2, 0, 1), right_rect.transpose(2, 0, 1)

    def rectified_left_pose(self, left_pos, left_rot):
        """
        World pose of the *rectified* left camera.

        left_rot must be a (3,3) camera->world rotation in OpenCV convention.
        R1 comes from cv2.stereoRectify and is an OpenCV rotation, so it must be
        composed AFTER any Unity->OpenCV conversion, never before.

        p_world = R_wc @ R1.T @ p_rect, so the rectified rotation is R_wc @ R1.T.
        """
        left_rot = np.asarray(left_rot, dtype=np.float64)
        if left_rot.shape == (4,):
            left_rot = Rotation.from_quat(left_rot).as_matrix()
        return left_pos, left_rot @ self._R1.T