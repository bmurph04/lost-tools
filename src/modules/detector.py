import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES
import cv2
import numpy as np
import torch
from pathlib import Path
import tqdm

class Detector:
    """
    Detector class that handles frame-by-frame object detections.

    Args:
        device - Device to move torch objects to.
        model- Detector model. 
        threshold (default = 0.5) - Object detection threshold.
    """
    def __init__(self, device, model, threshold=0.5):
        self.device = device
        self.model = model
        self.threshold = threshold

    def process_frame(self, frame, output=None):
        """
        Processes a single frame using the detector model.
        Logic taken from RF-DETR github: https://github.com/roboflow/rf-detr#detection-1

        Returns the processed frame.

        Args:
            frame - Path to the frame being processed.
            output - Output path for image if specified.
        """
        
        detections = self.model.predict(frame, self.threshold)
        # FIXME: change COCO_CLASSES to detections.data["class_name"]
        labels = [f"{i}: {COCO_CLASSES[class_id]}" for i, class_id in enumerate(detections.class_id)]

        annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
        # annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

        # Get relevant info for detected objects
        detections_info = {
            'coordinates': detections.xyxy, # shape: (D, 4) ndarray
            'class_ids': detections.class_id, # shape: (D,) ndarray
            'class_confidences': detections.confidence # shape: (D,) ndarray 
        }
        
        # If output specified, write to image
        if output:
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output), annotated_image_bgr)

        return detections_info, annotated_image
    
    def filter_detections_info(self, detections_info, objects_info):
        """
        Args:
            detections_info (_type_): _description_
            
        Returns:
        """
        # Get the points tensor from objects info
        objects_points_list = objects_info['points'] # size D list of tensors, shape: (n, 2)
        objects_class_ids = objects_info['class_ids'] # side D list of integers
        detections_coordinates = detections_info['coordinates'] # shape: (D, 4) ndarray
        detections_class_ids = detections_info['class_ids'] # shape: (D,) ndarray

        # Return original detections info if no points
        if len(objects_points_list) == 0 or len(detections_coordinates) == 0:
            return detections_info

        filtered_coordinates_list = []
        filtered_class_ids_list = []
        filtered_class_confidences_list = []

        num_detections = detections_coordinates.shape[0]
        for i in range(num_detections):
            bbox = detections_coordinates[i] # x_min, y_min, x_max, y_max
            x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            detection_class_id = detections_class_ids[i]

            filter_detection = False
            for object_points, object_class_id in zip(objects_points_list, objects_class_ids):

                points_x = object_points[:, 0]
                points_y = object_points[:, 1]

                already_detected_x = (bbox[0] <= points_x) & (points_x <= bbox[2])
                already_detected_y = (bbox[1] <= points_y) & (points_y <= bbox[3])
                already_detected_coordinates = already_detected_x & already_detected_y

                already_detected_class = detection_class_id == object_class_id
            
                # FIXME: Filter out objects if below a specified num_points threshold instead of if any points exist (refresh obj with new points)
                # If there is a truth value, this object is already being detected
                if torch.any(already_detected_coordinates) and already_detected_class:
                    filter_detection = True
                    break
                
            if not filter_detection:
                filtered_coordinates_list.append(detections_info['coordinates'][i])
                filtered_class_ids_list.append(detections_info['class_ids'][i])
                filtered_class_confidences_list.append(detections_info['class_confidences'][i])

        if len(filtered_coordinates_list) == 0:
            filtered_detections = None
        else: 
            filtered_detections = {
                'coordinates': np.stack(filtered_coordinates_list, axis=0), # shape: (D_filtered, 2)
                'class_ids': np.array(filtered_class_ids_list), # shape: (D_filtered,)
                'class_confidences': np.array(filtered_class_confidences_list) # shape: (D_filtered)
             }
        
        return filtered_detections
        
    # def process_frame_seq(self, frames_dir, output=None):
    #         """
    #         Processes a sequence of frames using the detector model.
    #         Should only be used for individually testing/debugging detector model.

    #         Returns the processed frame sequence along with detection info from each frame

    #         Args:
    #             frames_dir - Director containing sequence of frames.
    #             output - Output path for video if specified.
    #         """
            
    #         frames_dir_path = Path(frames_dir)
            
    #         detections_info_seq = []
    #         annotated_image_seq = []

    #         print(f'Processing frames from {frames_dir}')

    #         for frame in tqdm(frames_dir_path.iterdir()):
    #             # Process the frame
    #             detections_info, annotated_image = self.process_frame(frame)
                
    #             # Add to detections and frame sequence
    #             detections_info_seq.append(detections_info)
    #             annotated_image_seq.append(annotated_image)

    #         # If output specified, write to video
    #         if output:
    #             h, w, c = annotated_image_seq[0].shape
    #             video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    #             for frame in annotated_image_seq:
    #                 video_writer.write(frame)

    #             video_writer.release()
    #             print(f'Video saved to {output}')

    #         return detections_info_seq, annotated_image_seq