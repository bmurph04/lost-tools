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
    def __init__(self, device, model):

        self.device = device
        self.model = model

        # -- General initializations --
        self.stride = 0.015
        self.min_grid_size = 3
        self.max_grid_size = 16
        self.point_size = 100

        # -- Model specific initializations --
        if isinstance(model, Predictor):
            self.backbone = self.model.model.backbone
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
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
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
        
        if isinstance(self.model, Predictor):
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