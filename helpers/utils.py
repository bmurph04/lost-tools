import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

import yaml
from argparse import Namespace

from rfdetr.assets.coco_classes import COCO_CLASSES

# From https://github.com/facebookresearch/co-tracker/blob/9ed05317b794cd177674e681321780614a65e073/cotracker/models/core/model_utils.py#L20
def get_points_on_a_grid(
    size: Tuple[int, ...],
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    margin_div: Optional[int] = 64
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

# def points_to_bbox(points):
#     """
#     FIXME
#     Convert list of points to a representative bbox

#     Args:
#         points - (N, 2) points
    
#     Returns: (4,) tensor of bbox with [x1, y1, x2, y2]
#     """

#     # Simple: Get the min and max (x,y) values from all points
#     x = points[:, 0]
#     y = points[:, 1]

#     x_min, x_max = x.min(), x.max()
#     y_min, y_max = y.min(), y.max()

#     # Add small padding to avoid bbox with zero area
#     width = max(x_max - x_min, 1.0)
#     height = max(y_max - y_min, 1.0)

#     # Create bbox with padding
#     bbox = torch.stack([
#         x_min - (width * 0.1),
#         y_min - (height * 0.1),
#         x_max + (width * 0.1),
#         y_max + (height * 0.1)
#     ])

#     return bbox
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

def build_coco_to_react_mapping(react_obj_classes, coco_classes=COCO_CLASSES, device="cuda"):
    """
    Builds a lookup tensor that maps COCO class IDs (0..max_coco_id) 
    to REACT's PSG object class IDs.
    """
    # Create mapping dictionary from name -> PSG class index
    react_name_to_id = {name.lower(): idx for idx, name in enumerate(react_obj_classes)}
    
    # Handle dict vs list for coco_classes
    if isinstance(coco_classes, dict):
        max_coco_id = max(coco_classes.keys())
        # Initialize mapping array filled with default fallback ID (e.g. 1)
        mapping_arr = [1] * (max_coco_id + 1)
        for coco_id, coco_name in coco_classes.items():
            name_clean = coco_name.lower()
            if name_clean in react_name_to_id:
                mapping_arr[coco_id] = react_name_to_id[name_clean]
    else: # if list
        mapping_arr = []
        for coco_name in coco_classes:
            name_clean = coco_name.lower()
            mapping_arr.append(react_name_to_id.get(name_clean, 1))

    return torch.tensor(mapping_arr, dtype=torch.long, device=device)


def load_frame(frame_path):
    """
    Load a frame as a torch tensor.

    Returns nparray representing frame (Height, Width, Channels).

    Args:
        frame_path (str): Path to a frame. 
    """

    frame = Image.open(frame_path).convert("RGB")
    return np.asarray(frame)

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_args_from_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    args = Namespace(**cfg)
    return args

def clamp(value, low, high):
    return min(high, max(value, low))