import torch
from pathlib import Path
import argparse
import warnings
from tqdm import tqdm
# model imports
from rfdetr import RFDETRMedium, RFDETRSegMedium # type: ignore
from external.track_on.model.trackon_predictor import Predictor
import depth_pro # type: ignore
# lost-tools imports
from src.detector import Detector
from src.tracker import Tracker
from src.sgg2d import SceneGraphGenerator2D
from src.system_eval import SystemEvaluator
from src.custom_react_model import CustomReactModel
from helpers.utils import pick_device, load_args_from_yaml, convert_tracker_tokens_to_spatial_features
from src.geometric_sg import build_2d_scene_graph, save_scene_graph_frame

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
    p.add_argument("--tracker-ckpt", default="./checkpoints/trackon2_dinov3_checkpoint.pt", type=str, help="Path to tracker model checkpoint .pt")
    # 2D scene graph generator args
    p.add_argument("--sgg2d-config", default="./config/react_yolo12m_psg.yaml", type=str, help="Path to 2D scene graph generator config .yaml")
    p.add_argument("--sgg2d-ckpt", default="./checkpoints/react_yolo12m_psg.pth", type=str, help="Path to 2D scene graph generator checkpoint .pth")
    # Miscellaneous args
    p.add_argument("--generate-depth", default=False, type=bool, help="Boolean to generate depth information for input frames, necessary for runtime")

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
    sgg2d_config = args.sgg2d_config
    sgg2d_ckpt = args.sgg2d_ckpt
    frames_dir = sorted([f for f in Path(args.input).iterdir()], key=egoobjects_sort_key) # Sort input frame seq
    output_folder = args.output # Initialize output folder
    generate_depth = args.generate_depth

    # Choose device
    device = pick_device()

    # Initialize system evaluator for metrics
    sys_evaluator = SystemEvaluator(device=device)

    # For each module, initialize the model being used
    detector_model = RFDETRMedium()
    tracker_model = Predictor(model_args=tracker_config_args, checkpoint_path=tracker_ckpt, support_grid_size=0)
    depth_model = depth_pro.create_model_and_transforms() if generate_depth else None

    # Set models to evaluation mode
    tracker_model.eval()

    # Move models to correct device
    tracker_model.to(device)
    
    # Initialize the modules
    detector = Detector(device, detector_model)
    tracker = Tracker(device, tracker_model)

    # Optimize models for inference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector_model.optimize_for_inference(dtype=torch.float16)
        depth_model.eval()

    # Generate depth information if necessary
    if generate_depth:
        generate_depth_info()

    # Initialize variables and containers before frame loop
    tracker_hook_handle = tracker.backbone.register_forward_hook(tracker_hook_fn)
    test_speed = False # Set to True on the frame that speed tests should begin
    image_height, image_width = None, None
    
    # ----- Main loop -----
    for t, frame_path in tqdm(enumerate(frames_dir)):     
        frame_str = str(frame_path)
        new_queries = None
        
        if t >= WARMUP_FRAMES:
            test_speed = True

        sys_evaluator.start_speed_test('frame') if test_speed else None 

        # Initial frame query initializations
        if t == 0:
            # ----- Detector -----
            # Process initial frame using detector
            detections_info, detector_image = detector.process_frame(frame_str) # detector_image shape: (H, W, 3)        
            image_height, image_width, _ = detector_image.shape
            
            # ----- Tracker -----
            # Using the detector bbox info, create grid of queries for each object
            new_queries, query_classifications, query_instances, query_confidences = tracker.build_detection_grid_points(detections_info, frame_extent=(image_height, image_width))
            new_queries = new_queries.squeeze(0) # shape (1, N, 2) --> (N, 2)
            # Set the initial capacity of the tracker model to the number of queries
            tracker.model.initial_capacity = new_queries.shape[0]
    
        # ----- Detector -----
        # Process frames with detector at DETECTOR_FREQ hz
        if t % DETECTOR_FREQ == 0:
            sys_evaluator.start_speed_test('detector') if test_speed else None
            detections_info, detector_image = detector.process_frame(frame_str, output=f'outputs/{output_folder}/output_detector_{t:06d}.jpg') # detector_image shape: (H, W, 3)
            sys_evaluator.end_speed_test('detector') if test_speed else None
        
        # ----- Tracker -----
        # Process frame using tracker
        sys_evaluator.start_speed_test('tracker') if test_speed else None
        (points, visibles), tracker_image = tracker.process_frame(frame_str, new_queries=new_queries, output=f'outputs/{output_folder}/output_tracker_{t:06d}.jpg')
        sys_evaluator.end_speed_test('tracker') if test_speed else None

        tracker_feat_map = TRACKER_EXTRACTED_FEATURES['all_levels']
        standard_feat_map = convert_tracker_tokens_to_spatial_features(tracker_feat_map, tracker.model.model.input_size)
        # print("Feature map shapes:")
        # [print(f"{f.shape}") for f in standard_feat_map]
        # Combine relevant tracking info into a dictionary
        tracker_info = {
            'points': points, # Updated every frame
            'class_ids': query_classifications, # Initialized after detector
            'class_instances': query_instances,
            'class_confidences': query_confidences, # Initialized after detector
            'features': standard_feat_map # Initialized after detector
        }

        # ----- 2D Scene Graph Generator -----
        triplets = build_2d_scene_graph(tracker_info, device=device, image=tracker_image, output=f'outputs/{output_folder}/output_intermed_bboes_{t:06d}.jpg')
        save_scene_graph_frame(triplets, output=f'outputs/{output_folder}/output_geometricsg_{t:06d}.jpg')

        # nodes, rels = sgg2d.process_tracking_data(tracker_info, extent=(image_height, image_width), output=f'outputs/{output_folder}/output_sgg2d_{t:06d}.jpg')
        
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
