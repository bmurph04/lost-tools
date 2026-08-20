
import torch
from typing import Dict, List, Tuple
from src.utils import points_to_bbox
import torch.nn.functional as F
from rfdetr.assets.coco_classes import COCO_CLASSES

class FeatureProcessor:
    """
    FIXME
    Feature processor for tracking data.
    """
    def __init__(self, device, react_obj_classes):
        """
        Args:
            device -
            config -
        """
        self.device = device
        self.coco_to_react = self.build_coco_to_react_mapping(react_obj_classes, device=device)

        self.use_pca = False


    def forward(
            self, 
            tracker_info: Dict[str, torch.Tensor], 
            extent: Tuple[int, int]
        ) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        """
        FIXME
        Convert TrackOn2 data to REACT proposal format.

        Args:
            tracker_info - 
            image_sizes - 

        Returns:

        """



        # Initialize proposals
        proposals = self._build_proposals(tracker_info, extent)

        # Build features
        features = self._build_features(tracker_info['features'])

        projected_features = self._project_features(features)

        # Move feature maps to gpu
        projected_features = [feat.to(self.device, non_blocking=True) for feat in projected_features]        

        return proposals, projected_features

    def _build_proposals(self, tracker_info, extent):
        """
        build proposals
        """
        all_points = tracker_info['points'] # shape: (B, T, N, 2)
        all_class_ids = tracker_info['class_ids'] # shape: (B, N)
        all_class_instances = tracker_info['class_instances'] # shape: (B, N)
        all_confidences = tracker_info['confidences'] # shape: (B, N)
        features = tracker_info['features']

        source_h, source_w = extent
        react_size = 640

        # TrackOn returns point coordinates in the source-frame coordinate system,
        # while its feature maps were extracted from its fixed, non-square input.
        # This project feeds F8/F16/F32 to REACT, so recover that input size from
        # F8 instead of assuming a 640x640 YOLO letterbox.
        if len(features) < 4:
            raise ValueError("Expected TrackOn F4/F8/F16/F32 feature maps")
        track_h = int(features[1].shape[-2] * 8)
        track_w = int(features[1].shape[-1] * 8)
        source_h, source_w = extent
        scale_x = track_w / float(source_w)
        scale_y = track_h / float(source_h)

        num_batches = all_points.size(0)

        # Get the points from the last timestep
        current_points = all_points[:, -1, :, :] # shape: (B, N, 2)

        proposals = []

        # Iterate through each batch
        for b in range(num_batches):

            points = current_points[b, :, :] # shape: (N, 2)
            class_ids = all_class_ids[b, :] # shape: (N,)
            instances = all_class_instances[b, :] # shape: (N,)
            confidences = all_confidences[b, :] # shape: (N,)

            # Combine class_ids and instances into a single key
            combined_keys = (class_ids * 1000) + instances # shape: (N,) assuming < 1000 instances
            
            # CODE FOR NON-CONSECUTIVE CLASSID_INSTANCE PAIRS
            # # Get unique combined keys and index where each key appears at in combined_keys
            # unique_keys, inverse_indices = torch.unique(combined_keys, return_inverse=True)
            # # Get indexing required to put inverse_indices in order, putting all combined_ids contiguous
            # sorted_indices_idxs = torch.argsort(inverse_indices)
            # # Sort the points and index where each key appears at
            # sorted_inverse_indices = inverse_indices[sorted_indices_idxs]
            # sorted_points = points[sorted_indices_idxs]

            # # Count elements per (class, instance) group
            # counts = torch.bincount(sorted_inverse_indices)
            # grouped_points_list = list(torch.split(sorted_points, counts.tolist()))
            
            # Get the unique keys and group sizes
            unique_keys, combined_key_counts = torch.unique_consecutive(combined_keys, return_counts=True)

            # Split into tensors of same (class_id, instance)
            grouped_points_list = list(torch.split(points, combined_key_counts.tolist()))

            # Initialize list of bboxes
            bboxes_list = []

            # Build (D, 4) tensor of bboxes where D is number of object class ids
            for grouped_points in grouped_points_list:
                # Convert points info to bbox info
                bbox = points_to_bbox(grouped_points) # shape: (4,)
                bboxes_list.append(bbox)

            bboxes = torch.stack(bboxes_list, dim=0) # shape: (D, 4)
            
            # Get class_id labels for each box back out of the keys
            box_labels = (unique_keys // 1000).to(self.device) # shape: (D,)

            react_box_labels = self.coco_to_react[box_labels.long()]

            # Get the starting index of each group to get the confidence score
            start_indices = torch.cat([
                torch.tensor([0], device=confidences.device), 
                torch.cumsum(combined_key_counts, dim=0)[:-1]
            ])
            box_confidences = confidences[start_indices] # shape: (D,)

            # `boxes` remain in source-frame space for visualization and spatial
            # features.  `lb_boxes` are in TrackOn's feature-input space and must
            # be used by every visual feature lookup.
            lb_boxes = bboxes.clone()
            lb_boxes[:, [0, 2]] *= react_size / source_w
            lb_boxes[:, [1, 3]] *= react_size / source_h
            lb_boxes[:, [0, 2]].clamp_(0, react_size)
            lb_boxes[:, [1, 3]].clamp_(0, react_size)

            # Compute the flat index against the *actual* F8/F16/F32 grids.
            feat_lookup_idx = self._compute_feat_idx(lb_boxes, (react_size, react_size))

            # Create proposal
            proposal = {
                'boxes': bboxes.to(self.device).float(),
                'lb_boxes': lb_boxes.to(self.device).float(),
                # (H, W), unlike YOLO's historic square integer convention.
                'lb_input_size': react_size,
                'lb_gain': 1.0,
                'lb_pad_w': 0.0,
                'lb_pad_h': 0.0,
                'image_size': (source_w, source_h),
                'pred_labels': react_box_labels.to(self.device).long(),
                'labels': react_box_labels.to(self.device).long(),
                'pred_scores': box_confidences.to(self.device).float(),
                'feat_idx': feat_lookup_idx.to(self.device).long(),
                'mode': 'xyxy'
            }

            # Add to list of proposals
            proposals.append(proposal)
        
        return proposals

    def _compute_feat_idx(self, lb_boxes: torch.Tensor, extent: Tuple[int, int]) -> torch.Tensor:
        """
        Compute DAMP flat FPN indices for boxes in REACT's virtual square space.

        Args:
            lb_boxes: (N, 4), xyxy boxes in 640x640 virtual coordinates.
            extent: (H, W); expected to be (640, 640).

        Returns:
            (N,) flat indices over P3 (80x80), P4 (40x40), P5 (20x20).
        """
        input_h, input_w = extent

        if input_h != input_w:
            raise ValueError(
                "_compute_feat_idx expects virtual square coordinates; "
                f"got extent={extent}"
            )

        input_size = input_h
        stride_p3, stride_p4, stride_p5 = 8, 16, 32

        grid_p3 = input_size // stride_p3  # 80 for 640
        grid_p4 = input_size // stride_p4  # 40 for 640
        grid_p5 = input_size // stride_p5  # 20 for 640

        p3_count = grid_p3 * grid_p3
        p4_count = grid_p4 * grid_p4

        cx = (lb_boxes[:, 0] + lb_boxes[:, 2]) * 0.5
        cy = (lb_boxes[:, 1] + lb_boxes[:, 3]) * 0.5

        width = (lb_boxes[:, 2] - lb_boxes[:, 0]).clamp(min=1.0)
        height = (lb_boxes[:, 3] - lb_boxes[:, 1]).clamp(min=1.0)
        side = torch.sqrt(width * height)

        # Same scale policy as REACT's YOLO/DAMP path.
        use_p3 = side <= input_size / 8
        use_p5 = side > input_size / 4

        col_p3 = (cx / stride_p3).long().clamp(0, grid_p3 - 1)
        row_p3 = (cy / stride_p3).long().clamp(0, grid_p3 - 1)
        idx_p3 = row_p3 * grid_p3 + col_p3

        col_p4 = (cx / stride_p4).long().clamp(0, grid_p4 - 1)
        row_p4 = (cy / stride_p4).long().clamp(0, grid_p4 - 1)
        idx_p4 = p3_count + row_p4 * grid_p4 + col_p4

        col_p5 = (cx / stride_p5).long().clamp(0, grid_p5 - 1)
        row_p5 = (cy / stride_p5).long().clamp(0, grid_p5 - 1)
        idx_p5 = p3_count + p4_count + row_p5 * grid_p5 + col_p5

        return torch.where(use_p5, idx_p5, torch.where(use_p3, idx_p3, idx_p4))

    def _build_features(self, raw_features):
        """
        
        """
        # 2. Slice from 4 levels down to 3 levels matching REACT's FPN pyramid
        lvl0, lvl1, lvl2 = raw_features[1:]
        
        # 3. Project channels zero-shot to match REACT's [128, 256, 512] requirement
        if self.use_pca:
            # PCA Projection via 1x1 convolution (No gradients tracked)
            proj_lvl0 = F.conv2d(lvl0, self.w_lvl0) # [1, 256, H, W] -> [1, 128, H, W]
            proj_lvl1 = lvl1                        # Already [1, 256, H, W] (Matches REACT)
            proj_lvl2 = F.conv2d(lvl2, self.w_lvl2) # [1, 256, H, W] -> [1, 512, H, W]
        else:
            # Fallback deterministic math if PCA file isn't loaded:
            # Target channels for YOLO12m REACT ROI heads: [256, 512, 512]
            proj_lvl0 = lvl0                   # Shape: [1, 256, H0, W0] (matches 256)
            proj_lvl1 = lvl1.repeat(1, 2, 1, 1) # Shape: [1, 512, H1, W1] (256 -> 512)
            proj_lvl2 = lvl2.repeat(1, 2, 1, 1) # Shape: [1, 512, H2, W2] (256 -> 512)

        aligned_features = [proj_lvl0, proj_lvl1, proj_lvl2]
        
        return aligned_features

    def _project_features(self, aligned_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Resize channel-aligned TrackOn F8/F16/F32 features into REACT's
        virtual 640x640 P3/P4/P5 pyramid.

        Input:
            aligned_features:
                F8:  (B, 256, 48, 64)
                F16: (B, 512, 24, 32)
                F32: (B, 512, 12, 16)

        Output:
            P3: (B, 256, 80, 80)
            P4: (B, 512, 40, 40)
            P5: (B, 512, 20, 20)
        """
        expected_channels = (256, 512, 512)
        react_pyramid_sizes = ((80, 80), (40, 40), (20, 20))

        if len(aligned_features) != 3:
            raise ValueError(
                "Expected aligned F8/F16/F32 features, "
                f"but received {len(aligned_features)} levels"
            )

        projected_features = []

        for level, (feature, expected_c, target_size) in enumerate(
            zip(aligned_features, expected_channels, react_pyramid_sizes)
        ):
            if feature.shape[1] != expected_c:
                raise ValueError(
                    f"Level {level} has {feature.shape[1]} channels; "
                    f"expected {expected_c}"
                )

            projected_features.append(
                F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        return projected_features

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