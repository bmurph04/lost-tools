import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

import json
import yaml

# From https://github.com/facebookresearch/co-tracker/blob/9ed05317b794cd177674e681321780614a65e073/cotracker/models/core/model_utils.py#L20
def get_points_on_a_grid(
    size,
    extent,
    center = None,
    device = torch.device("cpu"),
    margin_div = 64
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    # UPDATE: break size into x size and y size
    size_y, size_x = size
    if size_x == 1 and size_y == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]
    
    # UPDATE: increase margin by 8x and divide into x, y
    margin_y = extent[0] / margin_div
    margin_x = extent[1] / margin_div
    range_y = (margin_y - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin_y)
    range_x = (margin_x - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin_x)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size_y, device=device),
        torch.linspace(*range_x, size_x, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def convert_tracker_tokens_to_spatial_features(raw_features, original_image_shape=(384, 512)):
    """
    Converts a list of [1, 1, N_tokens, 256] transformer tensors 
    into standard REACT-compatible 2D spatial feature maps [1, 256, H', W'].
    """
    spatial_features = []
    
    # Strides corresponding to F1, F2, F3, F4
    strides = [4, 8, 16, 32]
    img_h, img_w = original_image_shape
    
    for feat_tensor, stride in zip(raw_features, strides):
        # 1. Remove Batch and Time dimensions -> Shape: (N_tokens, 256)
        feat_flat = feat_tensor.squeeze(0).squeeze(0)  
        
        # 2. Compute 2D spatial dimensions (H_feat, W_feat) based on stride
        h_feat = img_h // stride
        w_feat = img_w // stride
        
        # 3. Reshape (N_tokens, C) -> (H_feat, W_feat, C)
        feat_2d = feat_flat.view(h_feat, w_feat, 256)
        
        # 4. Permute to (C, H_feat, W_feat) and add Batch dim -> (1, 256, H_feat, W_feat)
        feat_spatial = feat_2d.permute(2, 0, 1).unsqueeze(0)
        
        spatial_features.append(feat_spatial)
        
    return spatial_features

def points_to_bbox(points, padding_ratio=0.00):
    """
    Fits an Axis-Aligned Bounding Box (AABB) using robust quantiles to ignore tracker drift.
    """
    if points.size(0) < 2:
        if points.size(0) == 1:
            pt = points[0]
            pad = 5.0 # absolute pixels
            return torch.tensor([
                [pt[0] - pad, pt[1] - pad],
                [pt[0] + pad, pt[1] - pad],
                [pt[0] + pad, pt[1] + pad],
                [pt[0] - pad, pt[1] + pad]
            ], device=points.device)
        return torch.zeros((4, 2), device=points.device)

    # Use 5th and 95th percentiles to ignore severe tracker drift/outliers
    # Note: Requires float tensors
    points_f = points.float()
    x_min = torch.quantile(points_f[:, 0], 0.05)
    x_max = torch.quantile(points_f[:, 0], 0.95)
    y_min = torch.quantile(points_f[:, 1], 0.05)
    y_max = torch.quantile(points_f[:, 1], 0.95)

    width = torch.clamp(x_max - x_min, min=1.0)
    height = torch.clamp(y_max - y_min, min=1.0)

    # Apply padding
    x_min = x_min - (width * padding_ratio)
    x_max = x_max + (width * padding_ratio)
    y_min = y_min - (height * padding_ratio)
    y_max = y_max + (height * padding_ratio)

    return torch.tensor([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], device=points.device, dtype=points.dtype)
    
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

def load_frame(frame_path):
    """
    Load a frame as a torch tensor.

    Returns nparray representing frame (Height, Width, Channels).

    Args:
        frame_path (str): Path to a frame. 
    """

    frame = Image.open(frame_path).convert("RGB")
    frame_np = np.asarray(frame)
    frame_np_trans = np.transpose(frame_np, axes=(2, 0, 1))
    return frame_np_trans # shape: (3, H, W)

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_serialized_data(path: str):
    ext = path.split('.')[-1]

    if ext == 'yaml':
        with open(path, "r") as f:
                cfg = yaml.safe_load(f)

    elif ext == 'json':
        with open(path, "r") as f:
                cfg = json.load(f)

    else:
        raise RuntimeError(f"The config extension provided in {path} was not recognized")

    return cfg

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

def egoobjects_sort_key(file):
    f = str(file)
    result = f.rsplit('_', 1)[-1]
    result = result.rsplit('.')[0]
    return int(result)

def clamp(value, low, high):
    return min(high, max(value, low))