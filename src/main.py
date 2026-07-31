import torch
import numpy as np
from pathlib import Path
import argparse
import warnings
from tqdm import tqdm

# -- external model imports --
from external.rfdetr.src.rfdetr import RFDETRMedium
from external.track_on.model.trackon_predictor import Predictor
from external.unidepth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
from external.DPVO.dpvo.dpvo import DPVO

# -- lost-tools modules --
from src.modules.detector import Detector
from src.modules.tracker import Tracker
from src.modules.depth_estimator import DepthEstimator
from src.modules.pose_estimator import PoseEstimator
# from src.sgg2d import SceneGraphGenerator2D
from src.modules.point_lifter import PointLifter
from src.modules.scene_graph_generator_3d import SceneGraphGenerator3D
from src.modules.dynamic_scene_graph_3d import DynamicSceneGraph3D
from src.modules.system_eval import SystemEvaluator

# -- lost-tools models and methods --
from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.build_geometric_3dsg import Geometric3DSGBuilder
# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from src.utils import pick_device, load_args_from_yaml, load_args_from_json, load_frame, load_checkpoint

# global vars
WARMUP_FRAMES = 3
DETECTOR_FREQ = 5
PRED_NAMES = ['near', 'on'] # NOTE: predicate names are static for current implementation, should change if preds are generated

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lost Tools pipeline")
    p.add_argument("--input", required=True, type=str, help="Path to directory with streamed frames")
    p.add_argument("--output", default="./outputs", type=str, help="Folder path to output visualizations to")
    # Tracker args
    p.add_argument("--tracker-config", default="./configs/trackon2.yaml", type=str, help="Path to tracker model config .yaml")
    p.add_argument("--tracker-ckpt", default="./checkpoints/trackon2_dinov3_checkpoint.pt", type=str, help="Path to tracker model checkpoint")
    # 2D scene graph generator args
    p.add_argument("--sgg2d-config", default="./configs/react_yolo12m_psg.yaml", type=str, help="Path to 2D scene graph generator model config .yaml")
    p.add_argument("--sgg2d-ckpt", default="./checkpoints/react_yolo12m_psg.pth", type=str, help="Path to 2D scene graph generator model checkpoint")
    # Depth estimator args
    p.add_argument("--depth-config", default="./configs/unidepth.json", type=str, help="Path to depth estimator model config")
    p.add_argument("--depth-ckpt", default="./checkpoints/unidepth_model.bin", type=str, help="Path to depth estimator model checkpoint")
    # Pose estimator args
    p.add_argument("--pose-config", default="./configs/dpvo.yaml", type=str, help="Path to pose estimator model config .yaml")
    p.add_argument("--pose-ckpt", default="./checkpoints/dpvo.pth", type=str, help="Path to pose estimator model checkpoint")
    # Miscellaneous args
    p.add_argument("--generate-intrinsics", action='store_true', help="Boolean to generate intrinsics for input frames, necessary for runtime")
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
    depth_config = load_args_from_json(args.depth_config)
    depth_ckpt = args.depth_ckpt
    pose_config = load_args_from_yaml(args.pose_config)
    pose_ckpt = args.pose_ckpt
    frames_dir = sorted([f for f in Path(args.input).iterdir()], key=egoobjects_sort_key) # Sort input frame seq
    output_folder = args.output # Initialize output folder
    generate_intrinsics = args.generate_intrinsics
    visualize = args.visualize

    # Choose device
    device = pick_device()

    # Initialize detector model and module
    detector_model = RFDETRMedium() # pretrained weights are downloaded within init
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
    if generate_intrinsics:
        # depth_model, depth_preprocessing_transform = depth_pro.create_model_and_transforms() 
        depth_model = UniDepthV2(depth_config)
        depth_model.load_state_dict(load_checkpoint(depth_ckpt), strict=False)
        depth_model.resolution_level = 4
        depth_model.eval()
        depth_model.to(device)
        depth_estimator = DepthEstimator(device, depth_model)

        pose_model = DPVO(pose_config, pose_ckpt) # Set H and W params later
        pose_model.network.float()
        pose_estimator = PoseEstimator(device, pose_model)

    # Initialize point lifting method and module
    # FIXME: pass in args to configure
    point_lifting_method = Gaussian3DLift()
    point_lifter = PointLifter(point_lifting_method)

    # Initialize 3D scene graph generator method and module
    # FIXME: pass in args to configure
    scene_graph_gen_3d_method = Geometric3DSGBuilder()
    scene_graph_generator_3d = SceneGraphGenerator3D(sgg_method=scene_graph_gen_3d_method, point_lifting_method=point_lifting_method)
    
    # Initialize dynamic 3D scene graph class
    dynamic_scene_graph = DynamicSceneGraph3D(point_lifting_method=point_lifting_method)
    
    # Initialize system evaluator module for metrics
    sys_evaluator = SystemEvaluator(device=device)

    # Initialize variables and containers before frame loop
    test_speed = False # Set to True on the frame that speed tests should begin
    image_height, image_width = None, None
    # NOTE: These maps are static for current implementation, dynamic if preds are generated
    pred_name_to_id = {name: id for id, name in enumerate(PRED_NAMES)}
    pred_id_to_name = PRED_NAMES
    
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
    dynamic_sg_output_prefix = f'outputs/{output_folder}/output_dynamic_sg'
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t, frame_path in tqdm(enumerate(frames_dir)):     
            frame_str = str(frame_path)
            frame = torch.from_numpy(load_frame(frame_str)) # shape: (3, H, W)
            _, image_height, image_width = frame.shape
            
            if t >= WARMUP_FRAMES:
                test_speed = True

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # ----- Tracker -----
            num_total_points = sum(objects_info['object_point_counts'])
            has_active_points = num_total_points > 0
            points_list, visibles_list = [], []
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
                    tracker.initialize_queries(frame, new_points_list)
                    
                    if visualize:
                        new_visibles_list = [torch.ones(points.shape[0], dtype=torch.bool, device=points.device) 
                                                                for points in new_points_list]
                        visibles_list.extend(new_visibles_list)
                        tracker.visualize(frame, objects_info['points'], visibles_list, output=f'{tracker_output_prefix}_{t:06d}.jpg')

            # Generate depth information if necessary
            if generate_intrinsics:
                # ----- Depth Estimator -----
                assert depth_estimator is not None
                sys_evaluator.start_speed_test('depth_estimator') if test_speed else None
                depth, focal_length, camera_coords = depth_estimator.process_frame(frame)
                sys_evaluator.end_speed_test('depth_estimator') if test_speed else None
                depth_estimator.visualize(depth, output=f'{depth_estimator_output_prefix}_{t:06d}.jpg') if visualize else None

                intrinsics = [focal_length[0], focal_length[1], camera_coords[0], camera_coords[1]]

                # ----- Pose Estimator -----
                assert pose_estimator is not None
                sys_evaluator.start_speed_test('pose_estimator') if test_speed else None
                camera_rot, camera_trans = pose_estimator.process_frame(frame, t, intrinsics)
            else:
                # TODO: Assign depth, focal length, and camera coords directly from image and camera intrinsics
                pass
            
            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            points3d_representation, object_instances = point_lifter.lift_points(
                objects_point_list=objects_info['points'], 
                depth=depth, 
                focal_length=focal_length,
                camera_rot=camera_rot,
                camera_trans=camera_trans
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            point_lifter.visualize(
                frame,
                focal_length=focal_length,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                point_lifter_output=points3d_representation, 
                object_instances=object_instances,
                object_labels=objects_info['class_ids'],
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
            ) if visualize else None
                            
            # ----- 3D Scene Graph Generator -----
            sys_evaluator.start_speed_test('3dsg_gen') if test_speed else None
            scene_graph_3d = scene_graph_generator_3d.generate_triplets(points3d_representation, object_instances, pred_name_to_id)
            sys_evaluator.end_speed_test('3dsg_gen') if test_speed else None
            scene_graph_generator_3d.visualize(
                frame=frame,
                focal_length=focal_length,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                scene_graph=scene_graph_3d,
                pred_id_to_name=pred_id_to_name,
                points_representation=points3d_representation,
                object_instances=object_instances,
                object_labels=objects_info['class_ids'],
                output=f'{sgg3d_output_prefix}_{t:06d}.jpg'
            ) if visualize else None
            
            # ----- 3D Scene Graph Merging -----
            sys_evaluator.start_speed_test('3dsg_merge')
            update_idx = dynamic_scene_graph.add(
                object_labels=objects_info['class_ids'], 
                points_representation=points3d_representation, 
                triplets=scene_graph_3d
            )
            dynamic_scene_graph.merge(update_idx)
            sys_evaluator.end_speed_test('3dsg_merge')
            dynamic_scene_graph.visualize(
                frame=frame,
                focal_length=focal_length,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                pred_id_to_name=pred_id_to_name,
                output=f'{dynamic_sg_output_prefix}_{t:06d}.jpg'
            ) if visualize else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None
            
    # ----- Cleanup and evaluation -----
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()