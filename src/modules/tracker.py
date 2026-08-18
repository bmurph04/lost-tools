import torch
import numpy as np
import cv2
from math import sqrt
from pathlib import Path

from external.track_on.model.trackon_predictor import Predictor

from external.track_on.utils.vis_utils import plot_tracks_wo_tail
from src.utils import get_points_on_a_grid, clamp, load_frame


class Tracker:
    """
    Tracker class that handles tracking points (queries) across frames.

    Args:
        device - Device to move torch objects to.
        model - Detector model.
    """
    def __init__(self, name, model, device, max_grid_size=16, min_point_threshold=4, max_points=36, allowed_occlusion_frames=15, vis_ema=0.3):
        self.name = name
        self.device = device
        self.model = model

        # -- General initializations --
        self.stride = 0.015
        self.min_grid_size = 3
        self.max_grid_size = max_grid_size
        self.point_size = 100
        
        self.min_point_threshold = min_point_threshold
        self.max_points = max_points
        self.allowed_occlusion_frames = allowed_occlusion_frames
        self.vis_ema = vis_ema

        # -- Model specific initializations --
        if self.name == 'trackon2':
            self.backbone = self.model.model.backbone
            for attr in ('q_init', 'point_memory', 'temporal_mask', 'N'):
                if not hasattr(self.model, attr):
                    raise AttributeError(
                        f"Predictor is missing '{attr}'. Tracker.prune() depends on "
                        f"TrackOn2's internal buffer layout and needs updating.")
            model.reset()

                
    def process_frame(self, frame, object_query_counts):
            """
            Given a set of queries, process a frame using the initialized tracker
    
            Returns the updated queries.
            """
    
            def trackon_process_frame(frame, object_query_counts):
                """
                Process a frame using TrackOn tracker.
                
                Given a tensor of existing tracked points (N, 2) and a frame, propagate the points
                through the current frame.
    
                Returns points, visibles and frame output.
                """
                
                with torch.inference_mode():
                    # Process frame
                    frame_transformed = frame.unsqueeze(0) # shape (1, 3, H, W)
                    frame_transformed = frame_transformed.to(self.device, non_blocking=True) # Move frame to self.device
    
                    # Model forward pass
                    with torch.autocast(device_type=self.device, dtype=torch.float16):
                        points, visibles = self.model.forward_frame(frame_transformed)
                        
                # FIXME: add comment explaining this
                points_list = list(torch.split(points, object_query_counts, dim=0))
                visibles_list = list(torch.split(visibles, object_query_counts, dim=0))
                
                return (points_list, visibles_list)            
    
            # Run the correct process depending on the tracker model
            if isinstance(self.model, Predictor):
                # Process frame
                points_list, visibles_list = trackon_process_frame(frame, object_query_counts)
                
            return points_list, visibles_list      

    def initialize_queries(self, frame, new_queries_list):
        """
        Initialize new queries according to how the model does it.
        """
        # Don't try initializing anything if initialize_queries was called with new_queries_list = []
        if new_queries_list is None or len(new_queries_list) == 0:
            return

        if self.name == 'trackon2':
            frame_transformed = frame.unsqueeze(0) # shape (1, 3, H, W)
            frame_transformed = frame_transformed.to(self.device, non_blocking=True) # Move frame to self.device

            _, _, height, width = frame_transformed.shape
            _, _, _, _, f_fused_t = self.model.model.extract_frame_features(frame_transformed)

            new_queries = torch.cat(new_queries_list, dim=0).to(self.device, non_blocking=True)
            self.model.init_queries((f_fused_t, self.device), new_queries, height, width)

    def build_detection_grid_points(self, detections_info, frame_extent, margin_div=64):
        """
        Given a dictionary of information about detected objects, build tracker points
        uniformly in each detected object's bounding box.

        Args:
            detections_info - Dictionary of detected object information. Schema:
                {
                    coordinates: (D,4) array
                    class_ids: (D,) array
                    confidences: (D,) array
                }
            frame_extent - the height and width of the frame. Used for normalization
        
        FIXME Returns:
        """
        if detections_info is None:
            return [], []

        total_queries_list = []
        object_query_counts = []

        # Get number of detected objects
        num_objects = detections_info['coordinates'].shape[0]

        # Iterate through each detected object
        for i in range(num_objects):

            # Get coordinate, class_id, and confidence info about the object
            bbox = detections_info['coordinates'][i]
            class_id = detections_info['class_ids'][i]
            confidence = detections_info['class_confidences'][i]

            # Expand bbox tuple
            x_min, y_min, x_max, y_max = bbox
            
            # Calculate the width, height, and center of the bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_center_x = x_min + bbox_width/2
            bbox_center_y = y_min + bbox_height/2

            frame_height, frame_width = frame_extent
            bbox_width_norm = bbox_width / frame_width
            bbox_height_norm = bbox_height / frame_height

            # Compute adaptive grid size based on width and height of bbox
            grid_size_x = int(clamp(bbox_width_norm // self.stride, self.min_grid_size, self.max_grid_size))
            grid_size_y = int(clamp(bbox_height_norm // self.stride, self.min_grid_size, self.max_grid_size))

            # Compute the uniform points within the bbox
            # UPDATE: rely on stride to compute 
            queries = get_points_on_a_grid(size=(grid_size_y, grid_size_x), 
                                 extent=(bbox_height, bbox_width),
                                 center=(bbox_center_y, bbox_center_x), 
                                 margin_div=margin_div,
                                 device=self.device) # shape: (1, grid_size_x*grid_size_y, 2)
            
            queries = queries.squeeze(0) # shape: (grid_size_x*grid_size_y, 2)            
            
            # Concatenate to current tensor of query coordinates
            total_queries_list.append(queries)
            
            # Add query length to current initial_capacity
            object_query_counts.append(queries.size(0))
        
        return total_queries_list, object_query_counts
    
    def update_lifecycle(self, objects_info, points_list, visibles_list):
        """
        Absorb one frame of tracker output. forward_frame returns a BOOL
        visibility (delta_v already applied), so we accumulate an EMA to get a
        continuous score for ranking.
        """
        objects_info['points'] = points_list
        objects_info['visibles'] = visibles_list
        ages = objects_info['occluded_age']
        scores = objects_info['vis_score']

        assert len(ages) == len(scores) == len(points_list), \
            "objects_info lists are out of sync with tracker output"

        for i, vis in enumerate(visibles_list):
            v = vis.float()
            if scores[i] is None or scores[i].shape != v.shape:
                scores[i] = v.clone()          # newly seeded or re-seeded
            else:
                scores[i] = scores[i] * (1 - self.vis_ema) + v * self.vis_ema
            ages[i] = 0 if bool(vis.any()) else ages[i] + 1

    def build_keep_mask(self, objects_info):
        """
        Decimate weak/redundant points and retire fully-occluded objects.

        Returns a flat (N,) bool mask over the tracker's current point ordering
        and rewrites objects_info in place to hold only survivors.

        Retiring here is purely a 2D/VRAM decision -- the object's 3D node
        persists in the scene graph, and GaussianSG.merge re-associates it if
        the object is detected again.
        """
        masks = []
        pts_out, vis_out, counts_out, cls_out, conf_out, age_out, score_out = [], [], [], [], [], [], []

        for i, pts in enumerate(objects_info['points']):
            n = pts.shape[0]
            device = pts.device
            score = objects_info['vis_score'][i]
            age = objects_info['occluded_age'][i]
            vis = objects_info['visibles'][i]

            if age > self.allowed_occlusion_frames:
                masks.append(torch.zeros(n, dtype=torch.bool, device=device))
                continue

            if n <= self.min_point_threshold:
                keep = torch.ones(n, dtype=torch.bool, device=device)
            else:
                keep = score > 0.5                      # smoothed, so one bad frame is tolerated
                n_keep = int(keep.sum())

                if n_keep < self.min_point_threshold:
                    k = min(self.min_point_threshold, n)
                    keep = torch.zeros(n, dtype=torch.bool, device=device)
                    keep[torch.topk(score, k=k).indices] = True
                elif n_keep > self.max_points:
                    ranked = score.masked_fill(~keep, float('-inf'))
                    keep = torch.zeros(n, dtype=torch.bool, device=device)
                    keep[torch.topk(ranked, k=self.max_points).indices] = True

            n_keep = int(keep.sum())
            if n_keep == 0:
                # Never keep a zero-point object: gaussian_lift_points skips
                # empties, which would desync means_3d from class_ids.
                masks.append(torch.zeros(n, dtype=torch.bool, device=device))
                continue

            masks.append(keep)
            pts_out.append(pts[keep])
            vis_out.append(vis[keep])
            counts_out.append(n_keep)
            cls_out.append(objects_info['class_ids'][i])
            conf_out.append(objects_info['confidences'][i])
            age_out.append(age)
            score_out.append(score[keep])

        objects_info['points'] = pts_out
        objects_info['visibles'] = vis_out
        objects_info['object_point_counts'] = counts_out
        objects_info['class_ids'] = cls_out
        objects_info['confidences'] = conf_out
        objects_info['occluded_age'] = age_out
        objects_info['vis_score'] = score_out

        return torch.cat(masks) if masks else None

    def prune(self, keep_mask):
        """
        Drop points from the tracker's active set, preserving survivors' temporal
        memory. Implemented here rather than in external/track_on so that repo
        stays vendored-clean; it compacts Predictor's three parallel buffers,
        which forward_frame slices as [:N].
        """
        if keep_mask is None or not isinstance(self.model, Predictor):
            return

        predictor = self.model
        if predictor.q_init is None or predictor.N == 0:
            return

        n_active = predictor.N
        if keep_mask.shape[0] != n_active:
            raise ValueError(
                f"keep_mask has {keep_mask.shape[0]} entries but the tracker holds "
                f"{n_active} active points -- objects_info and tracker are out of sync.")

        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        n_keep = keep_idx.numel()
        if n_keep == n_active:
            return

        with torch.no_grad():
            # Advanced indexing copies, so there is no aliasing hazard
            predictor.q_init[:n_keep]        = predictor.q_init[:n_active][keep_idx]
            predictor.point_memory[:n_keep]  = predictor.point_memory[:n_active][keep_idx]
            predictor.temporal_mask[:n_keep] = predictor.temporal_mask[:n_active][keep_idx]

            # REQUIRED: init_queries writes only q_init for new queries and assumes
            # point_memory/temporal_mask are still factory-clean. After a prune the
            # vacated rows hold a removed point's memory, which the next object
            # would silently inherit.
            predictor.q_init[n_keep:n_active]        = 0
            predictor.point_memory[n_keep:n_active]  = 0
            predictor.temporal_mask[n_keep:n_active] = True

            predictor.N = n_keep


    def visualize(self, frame, points_list, visibles_list, output):
        """_summary_

        Args:
            points_list (_type_): _description_
            frame (_type_): _description_
            output (_type_): _description_
        """

        if isinstance(self.model, Predictor):
            points = torch.cat(points_list).unsqueeze(0) # shape (T, N, 2) -> (frame, point_index, coordinate)
            visibles = torch.cat(visibles_list).unsqueeze(0) # shape (T, N) -> (frame, point_index)
            points_nt2 = points.detach().cpu().numpy().transpose(1, 0, 2) # shape (N, T, 2)
            occluded_nt = (1 - visibles.detach().cpu().numpy()).transpose(1, 0) # shape (N, T) 
                        
            vis_frame_in = frame.unsqueeze(0).detach().cpu().numpy()
            vis_frame_in = np.transpose(vis_frame_in, axes=(0, 2, 3, 1)) # shape: (1, H, W, 3)

            # Will output a sequence of frames containing only one frame (1, H, W, 3)
            video_track = plot_tracks_wo_tail(
                vis_frame_in.copy(),
                points_nt2,
                occluded_nt,
                point_size=self.point_size
            )

            vis_frame_out = video_track[0]

            vis_frame_bgr = cv2.cvtColor(vis_frame_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output), vis_frame_bgr)