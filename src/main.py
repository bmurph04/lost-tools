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
from helpers.utils import pick_device, load_args_from_yaml, convert_tracker_tokens_to_spatial_features
from src.models.geometric_sg import build_2d_scene_graph, save_scene_graph_frame

# global vars
WARMUP_FRAMES = 5
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

def tracker_hook_fn(module, input, output):
    if isinstance(output, (list, tuple)):
        # Storing all 4 feature levels
        TRACKER_EXTRACTED_FEATURES['all_levels'] = [f.detach().cpu() for f in output]
        # Or if you just want the main high-level semantic feature map (F3/F4):
        TRACKER_EXTRACTED_FEATURES['main_feature_map'] = output[-1].detach().cpu()
    else:
        TRACKER_EXTRACTED_FEATURES['main_feature_map'] = output.detach().cpu()

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
    tracker_hook_handle = tracker.backbone.register_forward_hook(tracker_hook_fn)
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
                detections_info, detector_image = detector.process_frame(frame_str, output=f'{detector_output_prefix}_{t:06d}.jpg') # detector_image shape: (H, W, 3)
                sys_evaluator.end_speed_test('detector') if test_speed else None

                # FIXME: comment description (Filter detections)
                detections_info = detector.filter_detections_info(detections_info, objects_info)
                image_height, image_width, _ = detector_image.shape
                
                # ----- Tracker -----
                # Using the detector bbox info, create grid of queries for each object
                new_queries, new_object_point_counts = tracker.build_detection_grid_points(detections_info, frame_extent=(image_height, image_width))
                # Set the initial capacity of the tracker model to the number of queries
                tracker.model.initial_capacity = sum(new_object_point_counts) # FIXME: make general for any tracker

                # FIXME: comment description
                if detections_info is not None:
                    objects_info['object_point_counts'].extend(new_object_point_counts)
                    objects_info['class_ids'].extend(detections_info['class_ids'])
                    objects_info['confidences'].extend(detections_info['class_confidences'])
                
            # # ----- Detector -----
            # if t % DETECTOR_FREQ == 0:
            #     sys_evaluator.start_speed_test('detector') if test_speed else None
            #     detections_info, detector_image = detector.process_frame(frame_str, output=f'{detector_output_prefix}_{t:06d}.jpg') # detector_image shape: (H, W, 3)
            #     sys_evaluator.end_speed_test('detector') if test_speed else None
            
            # ----- Tracker -----
            # Process frame using tracker
            sys_evaluator.start_speed_test('tracker') if test_speed else None
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
                depth, focal_length = depth_estimator.process_frame(frame_str)

            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter')
            means_3d, covs_3d, valid_object_instances = point_lifter.lift_points(
                objects_info=objects_info, 
                depth=depth, 
                focal_length=focal_length,
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
                input_img=frame_str
            )
            sys_evaluator.end_speed_test('point_lifter')
            
            sys_evaluator.end_speed_test('frame') if test_speed else None
            
    # ----- Cleanup and evaluation -----
    # Remove hook handles
    tracker_hook_handle.remove()
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()



# avg_det = sys_evaluator.get_avg_latency('detector')
    # # avg_det = 0
    # avg_track = sys_evaluator.get_avg_latency('tracker')
    # # avg_track = 0
    # avg_total = avg_det + avg_track
    # fps_inference_only = 1000.0 / avg_total
    # print("\n" + "="*50)
    # print("           LATENCY BENCHMARK REPORT          ")
    # print("="*50)
    # print(f" Frames Evaluated:      {len(frames_dir)} (Skipped {WARMUP_FRAMES} warmup frames)")
    # print("-" * 50)
    # print(f" Detector (RF-DETR):    {avg_det:6.2f} ms  ({(avg_det/avg_total)*100:4.1f}%)")
    # print(f" Tracker (Track-On2):   {avg_track:6.2f} ms  ({(avg_track/avg_total)*100:4.1f}%)")
    # print("-" * 50)
    # print(f" Total Frame Latency:   {avg_total:6.2f} ms")
    # print(f" Pure Model FPS:        {fps_inference_only:6.2f} FPS (Detector + Tracker)")
    # print("="*50 + "\n")
    
    # frame_path = '/home/mrw4/workspaces/EgoObjects-Dataset/categories/3E6796957F4287E3094D10885F27F806/01/3E6796957F4287E3094D10885F27F806_01_53.jpg'

    # detections_info, detector_image = detector.process_frame(frame_path, output='output/output_detector.jpg')
    # queries, query_classifications = tracker.build_detection_grid_points(detections_info)

    # (points, visibles), tracker_image = tracker.process_frame(frame_path, new_queries=queries, output='output/output_tracker.jpg', input_img=detector_image)
