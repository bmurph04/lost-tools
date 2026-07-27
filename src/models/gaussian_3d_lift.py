import numpy as np

class Gaussian3DLift:
    def __init__(self, visualize=False):
        pass

    def lift_points(self, tracker_info, depth, camera_rot, camera_trans, focal_length_px):
        """
        
        points - (N, 2)
        """

        num_objects = tracker_info['num_objects']


        # Infer intrinsics (If no intrinsics, assume principal point is image center)
        height, width = depth.shape
        cx, cy = width / 2.0, height / 2.0
        fx = fy = focal_length_px

        # If no camera intrinsics, assume camera is origin of world coordinate system
        # Only works if analyzing per-frame and not building unified scene graph
        camera_rot = np.eye(3)
        camera_trans = np.zeros(3)

        # Compute 2D mean from tracked points
        mean_2d = np.round(np.mean(points, axis=1)).astype(int)


        