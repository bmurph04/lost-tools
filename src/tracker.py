import torch
import cv2
from math import sqrt
from pathlib import Path

from external.track_on.model.trackon_predictor import Predictor

from external.track_on.utils.vis_utils import plot_tracks_wo_tail
from helpers.utils import get_points_on_a_grid, clamp, load_frame


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

        # General initializations
        
        # Save extracted features from feature map during forward passes
        self.extracted_features = {}

        self.stride = 0.015
        self.min_grid_size = 3
        self.max_grid_size = 16
        self.point_size = 100

        # Model specific initializations
        if isinstance(model, Predictor):
            self.backbone = self.model.model.backbone
            model.reset()
                
    
    # def initialize_queries_from_detections(self, detections_info, existing_queries=None, existing_classifications=None):
    def build_detection_grid_points(self, detections_info, frame_extent):
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

        total_queries_list = []
        total_query_classifications_list = []
        total_query_instances_list = []
        total_query_confidences_list = []

        # Get number of detected objects
        num_objects = detections_info['coordinates'].shape[0]

        # # Initialize class ID instances to build query instances
        # class_id_instances = {}

        # Iterate through each detected object
        for i in range(num_objects):

            # Get coordinate, class_id, and confidence info about the object
            bbox = detections_info['coordinates'][i]
            class_id = detections_info['class_ids'][i]
            confidence = detections_info['class_confidences'][i]

            # # Add class_id to instance dict
            # class_id_instances[class_id] = class_id_instances.get(class_id, 0) + 1

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
                                 device='cpu') # shape: (1, grid_size^2, 2)
            
            queries = queries.squeeze(0) # shape: (grid_size^2, 2)            
            
            # Concatenate to current tensor of query coordinates
            total_queries_list.append(queries)

            # Concatenate to current tensor of query classifications
            query_classifications = torch.full((queries.size(0),), class_id)
            total_query_classifications_list.append(query_classifications)

            # Concatenate to current tensor of query instances
            query_instances = torch.full((queries.size(0),), i)
            total_query_instances_list.append(query_instances)

            # Concatenate to current tensor of query confidences
            query_confidences = torch.full((queries.size(0),), confidence)
            total_query_confidences_list.append(query_confidences)

        # Handle empty edge case if no objects detected
        if not total_queries_list:
            total_queries = torch.empty((1, 0, 2))
            total_query_classifications = torch.empty((1, 0))
            total_query_instances = torch.empty((1, 0))
            total_query_confidences = torch.empty((1, 0))
        else:
            total_queries = torch.cat(total_queries_list, dim=0).unsqueeze(0) # shape (1, N, 2)
            total_query_classifications = torch.cat(total_query_classifications_list, dim=0).unsqueeze(0) # shape (1, N)
            total_query_instances = torch.cat(total_query_instances_list, dim=0).unsqueeze(0) # shape (1, N)
            total_query_confidences = torch.cat(total_query_confidences_list, dim=0).unsqueeze(0) # shape (1, N)
        
        return total_queries, total_query_classifications, total_query_instances, total_query_confidences
    

    def process_frame(self, frame_path, new_queries=None, output=None, input_img=None):
        """
        Given a set of queries, process a frame using the initialized tracker

        Returns the updated queries.
        """

        def trackon_process_frame(frame, new_queries, output, input_img):
            """
            Process a frame using TrackOn tracker.
            
            Given a tensor of existing tracked points (N, 2) and a frame, propagate the points
            through the current frame.

            Returns points, visibles and frame output.
            """
            
            with torch.inference_mode():
                # Process frame
                frame_transformed = frame.permute(2, 0, 1) # shape (3, H, W)  
                frame_transformed = frame_transformed.unsqueeze(0) # shape (1, 3, H, W)

                # Move frame and queries to self.device
                frame_transformed = frame_transformed.to(self.device, non_blocking=True)
                if new_queries is not None:
                    new_queries = new_queries.to(self.device, non_blocking=True)

                # Model forward pass
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    points, visibles = self.model.forward_frame(frame_transformed, new_queries=new_queries)

            points = points.unsqueeze(0).unsqueeze(0) # shape (1, T, N, 2) -> (batch, frame, point_index, coordinate)
            visibles = visibles.unsqueeze(0).unsqueeze(0) # shape (1, T, N) -> (batch, frame, point_index)

            points_nt2 = points[0].detach().cpu().numpy().transpose(1, 0, 2) # shape (N, T, 2)
            occluded_nt = (1 - visibles[0].detach().cpu().numpy()).transpose(1, 0) # shape (N, T)
            
            # vis_frame_in = input_img if input_img is not None else frame
            if input_img is not None:
                vis_frame_in = torch.from_numpy(input_img)
            else:
                vis_frame_in = frame

            # Will output a sequence of frames containing only one frame (1, H, W, 3)
            video_track = plot_tracks_wo_tail(
                vis_frame_in.unsqueeze(0),
                points_nt2,
                occluded_nt,
                point_size=self.point_size
            )

            vis_frame_out = video_track[0]

            if output:
                vis_frame_bgr = cv2.cvtColor(vis_frame_out, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output), vis_frame_bgr)
                # print(f'Frame saved to {output}')
                
            return (points, visibles), vis_frame_out
        
        
        # Load frame as image given frame path
        frame = load_frame(frame_path) # shape (H, W, 3)
        frame_tensor = torch.from_numpy(frame)

        # Run the correct process depending on the tracker model
        if isinstance(self.model, Predictor):
            tracker_output, annotated_image = trackon_process_frame(frame_tensor, new_queries, output, input_img)

        
        return tracker_output, annotated_image