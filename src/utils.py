import torch
import numpy as np
from scipy.spatial.transform import Rotation

import json
import yaml
import cv2
    
def unity_pose_to_cv(unity_pose):
    """
    Convert a Unity camera->world pose (left-handed, Y-up) into OpenCV
    convention (right-handed, Y-down).

    Return cv pos and cv quat.
    """
    unity_pos, unity_quat = unity_pose
    # Convert the position vector by inverting the y axis
    cv_pos = np.array([unity_pos[0], -unity_pos[1], unity_pos[2]])
    
    # Convert the rotation quaternion by inverting the x and z axes
    cv_quat = np.array([-unity_quat[0], unity_quat[1], -unity_quat[2], unity_quat[3]])
    
    cv_rot = Rotation.from_quat(cv_quat).as_matrix()

    return cv_pos, cv_rot
        
def compute_rel_camera_extrinsics(left_geometry, right_geometry, coord_conversion_func = lambda x: x):
    """
    Compute relative camera extrinsics given geometry and sample world poses. 

    Args:
        left_geometry (_type_): _description_
        right_geometry (_type_): _description_
    """
    # Get world positions as numpy arrays
    pos_L = np.array(left_geometry['pos'])
    pos_R = np.array(right_geometry['pos'])
    
    # Get world rotations as rotation objects
    rot_world_L = Rotation.from_quat(left_geometry['rot']) # left camera frame represented in world frame
    rot_world_R = Rotation.from_quat(right_geometry['rot']) # right camera frame represented in world frame

    # Compute relative rotation
    rot_R_world = rot_world_R.inv() # world frame represented in right camera frame
    rot_R_L = rot_R_world * rot_world_L # left camera frame represented in right camera frame  
    rot_R_L_quat = rot_R_L.as_quat()

    # Compute relative translation
    pos_diff = pos_L - pos_R # vector in world frame pointing from right camera to left camera
    trans_in_R = rot_R_world.apply(pos_diff) # above vector represented in right camera frame

    trans_in_R, rot_R_L_quat = coord_conversion_func((trans_in_R, rot_R_L_quat))
    
    return trans_in_R, rot_R_L_quat

def compute_image_scale_factor(current_size, target_size):

    current_width, current_height = current_size
    target_width, target_height = target_size

    if target_width / current_width != target_height / current_height:
        raise ValueError(f"Can't compute scale factor because width and height is scaled differently ({target_width/current_width} for width vs {target_height/current_height} for height)") 

    return target_width / current_width

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_serialized_data(data, load_type=None):
        
    if isinstance(data, str):
        path = data
        ext = path.split('.')[-1]

        if ext == 'yaml':
            with open(path, "r") as f:
                    result = yaml.safe_load(f)

        elif ext == 'json':
            with open(path, "r") as f:
                    result = json.load(f)

        else:
            raise RuntimeError(f"The config extension provided in {path} was not recognized")
        
        return result
    
    if load_type == 'json':
        return json.loads(data)
    elif load_type == 'yaml':
        return yaml.safe_load(data)
    else:
        raise RuntimeError(f"Unknown how to load {data=} with load_type {load_type}")


def load_checkpoint(ckpt: str):
    # 1. Load the checkpoint file into memory first
    if isinstance(ckpt, str):
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    else:
        checkpoint = ckpt

    # 2. Extract state dict if nested inside 'model' or 'state_dict' keys
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    return state_dict

def clamp(value, low, high):
    return min(high, max(value, low))