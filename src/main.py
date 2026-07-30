import torch
import numpy as np
from pathlib import Path
import argparse
import warnings
from tqdm import tqdm

# -- external model imports --
from external.rfdetr.src.rfdetr import RFDETRMedium
from external.track_on.model.trackon_predictor import Predictor
from unidepth.models import UniDepthV2  # type: ignore

# -- lost-tools modules --
from src.modules.detector import Detector
from src.modules.tracker import Tracker
from src.modules.depth_estimator import DepthEstimator
# from src.sgg2d import SceneGraphGenerator2D
from src.modules.point_lifter import PointLifter
from src.modules.scene_graph_generator_3d import SceneGraphGenerator3D
from src.modules.system_eval import SystemEvaluator

# -- lost-tools models and methods --
from src.models.gaussian_3d_lift import Gaussian3DLift
from src.models.geometric_3dsg_build import Geometric3DSGBuilder
# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from helpers.utils import pick_device, load_args_from_yaml, load_frame

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
    p.add_argument("--visualize", action='store_true', help="Boolean to visualize each step of pipeline")

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
    visualize = args.visualize

    # Choose device
    device = pick_device()

    # Initialize detector model and module
    detector_model = RFDETRMedium()
    # with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    detector_model.inference()
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
        depth_model.resolution_level = 4
        depth_model.eval()
        depth_model.to(device)
    depth_estimator = DepthEstimator(device, depth_model) if generate_depth else None

    # Initialize point lifting method and module
    # FIXME: pass in args to configure
    point_lifting_method = Gaussian3DLift()
    point_lifter = PointLifter(point_lifting_method)

    # Initialize 3D scene graph generator method and module
    # FIXME: pass in args to configure
    scene_graph_gen_3d_method = Geometric3DSGBuilder()
    scene_graph_generator_3d = SceneGraphGenerator3D(scene_graph_gen_3d_method)
 
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
    depth_estimator_output_prefix = f'outputs/{output_folder}/output_depth_estimator'
    point_lifter_output_prefix = f'outputs/{output_folder}/output_point_lifter'
    sgg3d_output_prefix = f'outputs/{output_folder}/output_sgg3d'
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t, frame_path in tqdm(enumerate(frames_dir)):     
            frame_str = str(frame_path)
            frame = torch.from_numpy(load_frame(frame_str))
            image_height, image_width, _ = frame.shape
            
            if t >= WARMUP_FRAMES:
                test_speed = True

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # ----- Tracker -----
            num_total_points = sum(objects_info['object_point_counts'])
            has_active_points = num_total_points > 0

            # Process frame if there are active points 
            if has_active_points:
                # Set tracker initial capacity based on object point count
                tracker.model.initial_capacity = num_total_points # FIXME: make general for any tracker
                # Process frame using tracker
                sys_evaluator.start_speed_test('tracker') if test_speed else None
                points_list, visibles_list = tracker.process_frame(frame, objects_info['object_point_counts'])
                sys_evaluator.end_speed_test('tracker') if test_speed else None 
                tracker.visualize(frame, points_list, visibles_list, output=f'{tracker_output_prefix}_{t:06d}.jpg') if visualize else None

                # Store points list at frame t
                objects_info['points'] = points_list

            if t % DETECTOR_FREQ == 0:
                # ----- Detector -----
                # Process frame using detector
                sys_evaluator.start_speed_test('detector') if test_speed else None
                detections_info = detector.process_frame(frame)
                sys_evaluator.end_speed_test('detector') if test_speed else None
                detector.visualize(frame, detections_info, output=f'{detector_output_prefix}_{t:06d}.jpg') if visualize else None
                
                # Filter detections againt updated tracker point positions
                detections_info = detector.filter_detections_info(detections_info, objects_info)

                # FIXME: comment description
                if detections_info is not None:
                    # Using the detector bbox info, greate grid of queries for each object
                    new_points_list, new_object_point_counts = tracker.build_detection_grid_points(
                        detections_info, 
                        frame_extent=(image_height, image_width), 
                        margin_div=8
                    )

                    objects_info['points'].extend(new_points_list)
                    objects_info['object_point_counts'].extend(new_object_point_counts)
                    objects_info['class_ids'].extend(detections_info['class_ids'])
                    objects_info['confidences'].extend(detections_info['class_confidences'])
                    tracker.initialize_queries(frame_path, new_points_list)
                    
                    if visualize:
                        new_visibles_list = [torch.ones(points.shape[0], dtype=torch.bool, device=points.device) 
                                                                for points in new_points_list]
                        visibles_list.extend(new_visibles_list)
                        tracker.visualize(frame, objects_info['points'], visibles_list, output=f'{tracker_output_prefix}_{t:06d}.jpg')

            # ----- Depth Estimator -----
            # Generate depth information if necessary
            if generate_depth:
                assert depth_estimator is not None
                sys_evaluator.start_speed_test('depth_estimator') if test_speed else None
                depth, focal_length = depth_estimator.process_frame(frame)
                sys_evaluator.end_speed_test('depth_estimator') if test_speed else None
                depth_estimator.visualize(depth, output=f'{depth_estimator_output_prefix}_{t:06d}.jpg') if visualize else None
            else:
                # TODO: Assign depth and focal length directly from image and camera intrinsics
                pass
            
            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            point3d_representation, object_instances = point_lifter.lift_points(
                objects_point_list=objects_info['points'], 
                depth=depth, 
                focal_length=focal_length,
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            point_lifter.visualize(
                frame,
                focal_length=focal_length,
                point_lifter_output=point3d_representation, 
                object_instances=object_instances,
                object_labels=objects_info['class_ids'],
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
            ) if visualize else None
                            
            # ----- 3D Scene Graph Generator -----
            sys_evaluator.start_speed_test('3d_sgg') if test_speed else None
            scene_graph_3d = scene_graph_generator_3d.generate_graph(point3d_representation, object_instances)
            sys_evaluator.end_speed_test('3d_sgg') if test_speed else None
            scene_graph_generator_3d.visualize(
                frame=frame,
                focal_length=focal_length,
                scene_graph=scene_graph_3d,
                points_representation=point3d_representation,
                object_instances=object_instances,
                object_labels=objects_info['class_ids'],
                output=f'{sgg3d_output_prefix}_{t:06d}.jpg'
            ) if visualize else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None
            
    # ----- Cleanup and evaluation -----
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()