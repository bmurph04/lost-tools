import torch
import numpy as np
from pathlib import Path
import argparse
import warnings
from tqdm import tqdm

# -- external model imports --
from rfdetr import RFDETRMedium, RFDETRSegMedium # type: ignore
from external.track_on.model.trackon_predictor import Predictor
from unidepth.models import UniDepthV2  # type: ignore

# -- lost-tools modules --
from src.modules.detector import Detector
from src.modules.tracker import Tracker
from src.modules.depth_estimator import DepthEstimator
# from src.sgg2d import SceneGraphGenerator2D
from src.modules.point_lifter import PointLifter
from src.modules.system_eval import SystemEvaluator

# -- lost-tools models and methods --
from src.models.gaussian_3d_lift import Gaussian3DLift

# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from helpers.utils import pick_device, load_args_from_yaml

# global vars
WARMUP_FRAMES = 3
DETECTOR_FREQ = 5
TRACKER_EXTRACTED_FEATURES = {}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lost Tools pipeline")
    p.add_argument("--input", required=True, type=str, help="Path to directory with streamed frames")
    p.add_argument("--output", default="./outputs", type=str, help="Folder path to output visualizations to")
    # Tracker args
    p.add_argument("--tracker-config", default="./config/trackon2.yaml", type=str, help="Path to tracker model config .yaml")
    p.add_argument("--tracker-ckpt", default="./checkpoints/trackon2_dinov3_checkpoint.pt", type=str, help="Path to tracker model checkpoint")
    # 2D scene graph generator args
    p.add_argument("--sgg2d-config", default="./config/react_yolo12m_psg.yaml", type=str, help="Path to 2D scene graph generator config .yaml")
    p.add_argument("--sgg2d-ckpt", default="./checkpoints/react_yolo12m_psg.pth", type=str, help="Path to 2D scene graph generator checkpoint")
    # Depth estimator args
    p.add_argument("--depth-ckpt", default="./checkpoints/unidepth.safetensors", type=str, help="Path to depth estimator checkpoint")
    # Miscellaneous args
    p.add_argument("--generate-depth", action='store_true', help="Boolean to generate depth information for input frames, necessary for runtime")

    return p.parse_args()

def egoobjects_sort_key(file):
    f = str(file)
    result = f.rsplit('_', 1)[-1]
    result = result.removesuffix(".jpg")
    return int(result)


def main() -> None:

    # ----- Initialization setup -----

    # Parse user args
    args = parse_args()
    # Initialize parsed user args
    tracker_config_args = load_args_from_yaml(args.tracker_config) # Load tracker config
    tracker_ckpt = args.tracker_ckpt
    depth_ckpt = args.depth_ckpt
    frames_dir = sorted([f for f in Path(args.input).iterdir()], key=egoobjects_sort_key) # Sort input frame seq
    output_folder = args.output # Initialize output folder
    generate_depth = args.generate_depth

    # Choose device
    device = pick_device()

    # Initialize detector model and module
    detector_model = RFDETRMedium()
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector_model.optimize_for_inference(dtype=torch.float16) # detector_model.eval()
    detector = Detector(device, detector_model)

    # Initialize tracker model and module
    tracker_model = Predictor(model_args=tracker_config_args, checkpoint_path=tracker_ckpt, support_grid_size=0)
    tracker_model.eval()
    tracker_model.to(device)
    tracker = Tracker(device, tracker_model)

    # Initialize depth estimator model and module
    if generate_depth:
        # depth_model, depth_preprocessing_transform = depth_pro.create_model_and_transforms() 
        depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-vitb14")
        depth_model.eval()
        depth_model.to(device)
    depth_estimator = DepthEstimator(device, depth_model) if generate_depth else None

    # Initialize point lifting method and module
    point_lifting_method = Gaussian3DLift()
    point_lifter = PointLifter(point_lifting_method)
 
    # Initialize system evaluator module for metrics
    sys_evaluator = SystemEvaluator(device=device)

    # Initialize variables and containers before frame loop
    test_speed = False # Set to True on the frame that speed tests should begin
    image_height, image_width = None, None
    
    objects_info = {
        'points': [], # size D list, shape (n, 2) tensors
        'object_point_counts': [], # size D list of integers
        'class_ids': [], # size D list of integers
        'confidences': [], # side D list of integers
    } # Object info container that is updated with each frame

    # Initialize output strings
    detector_output_prefix = f'outputs/{output_folder}/output_detector'
    tracker_output_prefix = f'outputs/{output_folder}/output_tracker'
    point_lifter_output_prefix = f'outputs/{output_folder}/output_point_lifter'
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t, frame_path in tqdm(enumerate(frames_dir)):     
            frame_str = str(frame_path)
            new_queries = None
            
            if t >= WARMUP_FRAMES:
                test_speed = True

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # Process new objects at DETECTOR_FREQ hz
            if t % DETECTOR_FREQ == 0:
                # ----- Detector -----
                # Process frames with detector at DETECTOR_FREQ hz
                sys_evaluator.start_speed_test('detector') if test_speed else None
                detections_info, detector_image = detector.process_frame(
                    frame_str, 
                    output=f'{detector_output_prefix}_{t:06d}.jpg'# detector_image shape: (H, W, 3)
                )
                sys_evaluator.end_speed_test('detector') if test_speed else None

                # FIXME: comment description (Filter detections)
                detections_info = detector.filter_detections_info(detections_info, objects_info)
                image_height, image_width, _ = detector_image.shape
                
                # ----- Tracker -----
                # Using the detector bbox info, create grid of queries for each object
                new_queries, new_object_point_counts = tracker.build_detection_grid_points(detections_info, frame_extent=(image_height, image_width), margin_div=8)

                # FIXME: comment description
                if detections_info is not None:
                    objects_info['object_point_counts'].extend(new_object_point_counts)
                    objects_info['class_ids'].extend(detections_info['class_ids'])
                    objects_info['confidences'].extend(detections_info['class_confidences'])
          
            # ----- Tracker -----
            # Process frame using tracker
            sys_evaluator.start_speed_test('tracker') if test_speed else None
            # Set the initial capacity of the tracker model to the number of queries
            tracker.model.initial_capacity = sum(objects_info['object_point_counts']) # FIXME: make general for any tracker
            (points_list, visibles_list) = tracker.process_frame(
                frame_str, 
                objects_info['object_point_counts'],
                new_queries=new_queries, 
                # output=f'{tracker_output_prefix}_{t:06d}.jpg'
                )
            sys_evaluator.end_speed_test('tracker') if test_speed else None

            objects_info['points'] = points_list

            # ----- Depth Estimator -----
            # Generate depth information if necessary
            if generate_depth:
                assert depth_estimator is not None
                sys_evaluator.start_speed_test('depth_estimator') if test_speed else None
                depth, focal_length = depth_estimator.process_frame(frame_str)
                sys_evaluator.end_speed_test('depth_estimator') if test_speed else None

            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            means_3d, covs_3d, valid_object_instances = point_lifter.lift_points(
                objects_info=objects_info, 
                depth=depth, 
                focal_length=focal_length,
                output=f'{point_lifter_output_prefix}_in3d_{t:06d}.jpg', 
                input_img=frame_str
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None

            if t % 200 == 10:
                sys_evaluator.print_latency_metrics()
            
    # ----- Cleanup and evaluation -----
    # Remove hook handles
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()