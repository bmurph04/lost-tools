import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES
import cv2
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
            'coordinates': detections.xyxy, # shape: (D, 4)
            'class_ids': detections.class_id, # shape: (D,)
            'class_confidences': detections.confidence # shape: (D,)
        }
        
        # If output specified, write to image
        if output:
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output), annotated_image_bgr)
            # print(f'Frame saved to {output}')

        return detections_info, annotated_image

    def process_frame_seq(self, frames_dir, output=None):
        """
        Processes a sequence of frames using the detector model.
        Should only be used for individually testing/debugging detector model.

        Returns the processed frame sequence along with detection info from each frame

        Args:
            frames_dir - Director containing sequence of frames.
            output - Output path for video if specified.
        """
        
        frames_dir_path = Path(frames_dir)
        
        detections_info_seq = []
        annotated_image_seq = []

        print(f'Processing frames from {frames_dir}')

        for frame in tqdm(frames_dir_path.iterdir()):
            # Process the frame
            detections_info, annotated_image = self.process_frame(frame)
            
            # Add to detections and frame sequence
            detections_info_seq.append(detections_info)
            annotated_image_seq.append(annotated_image)

        # If output specified, write to video
        if output:
            h, w, c = annotated_image_seq[0].shape
            video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

            for frame in annotated_image_seq:
                video_writer.write(frame)

            video_writer.release()
            print(f'Video saved to {output}')

        return detections_info_seq, annotated_image_seq
