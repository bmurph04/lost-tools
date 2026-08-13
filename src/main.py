import torch
import numpy as np
from pathlib import Path
from argparse import Namespace
import warnings
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation

# Add external repo root to sys.path
ffs_path = Path(__file__).resolve().parent.parent / "external" / "fast_foundationstereo"
if str(ffs_path) not in sys.path:
    sys.path.insert(0, str(ffs_path))

# -- external model imports --
from external.rfdetr.src.rfdetr import RFDETRMedium
from external.track_on.model.trackon_predictor import Predictor
from external.unidepth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
from core.foundation_stereo import FastFoundationStereo
from external.DPVO.dpvo.dpvo import DPVO

# -- lost-tools modules --
from src.modules.detector import Detector
from src.modules.tracker import Tracker
from src.modules.stereo_rectifier import StereoRectifier
from src.modules.depth_provider import DepthProvider
from src.modules.pose_provider import PoseProvider
# from src.sgg2d import SceneGraphGenerator2D
from src.modules.point_lifter import PointLifter
from src.modules.scene_graph_generator_3d import SceneGraphGenerator3D
from src.modules.dynamic_scene_graph_3d import DynamicSceneGraph3D
from src.modules.system_eval import SystemEvaluator

# -- lost-tools models and methods --
from src.models.pose_metadata import PoseMetadata
from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.build_geometric_3dsg import Geometric3DSGBuilder
# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from src.args import parse_and_validate_args
from src.utils import pick_device, load_serialized_data, load_frame, load_checkpoint, egoobjects_sort_key, compute_rel_camera_extrinsics

# global vars
WARMUP_FRAMES = 8
DETECTOR_FREQ = 5
PRED_NAMES = ['near', 'on'] # NOTE: predicate names are static for current implementation, should change if preds are generated

def main() -> None:

    # ----- Initialization setup -----
    config_dict = parse_and_validate_args()
    device = pick_device()

    frames_dir = sorted([f for f in Path(config_dict['input']).iterdir()], key=egoobjects_sort_key) # Sort input frame seq
    _, image_height, image_width = load_frame(frames_dir[0]).shape # Grab the first frame to get image width and height

    if config_dict['input_right'] is not None:
        right_frames_dir = sorted([f for f in Path(config_dict['input_right']).iterdir()], key=egoobjects_sort_key)
        assert len(right_frames_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but right_frames_dir has {len(frames_dir)} frames"

    if config_dict['input_metadata'] is not None:
        # TODO: Validate metadata schema
        metadata_dir = sorted([f for f in Path(config_dict['input_metadata']).iterdir()], key=egoobjects_sort_key)
        assert len(metadata_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but metadata_dir has {len(metadata_dir)} frames"

        geometry_data = load_serialized_data(str(metadata_dir[0]))
        
    if config_dict['input_geometry'] is not None:
        # TODO: Behavior for supplying intrinsics and relative extrinsics directly from config
        geometry_data = load_serialized_data(config_dict['input_geometry'])
        
    # Initialize geometry
    if config_dict['geometry_source'] == 'metadata' or config_dict['geometry_source'] == 'external':
        # Initialize geometry with intrinsic data
        left_geometry, right_geometry = geometry_data.get('leftCamera'), geometry_data.get('rightCamera')
        if left_geometry is None or right_geometry is None:
            raise KeyError(f"Tried accessing left and right geometry using keys leftCamera and rightCamera, \
                           but got {left_geometry} for leftCamera and {right_geometry} for rightCamera")
            
        
        # Check if we're missing relative rotation and translation info
        if geometry_data.get('relative_rot') is None or geometry_data.get('relative_trans') is None:
            # If we have access to a sample world position and rotation from geometry_data, compute relative extrinsics directly
            has_world_trans = 'pos' in left_geometry.keys() and 'pos' in right_geometry.keys()
            has_world_rot = 'rot' in left_geometry.keys() and 'rot' in right_geometry.keys()
            if has_world_trans and has_world_rot:
                geometry_data['relative_trans'], geometry_data['relative_rot'] = compute_rel_camera_extrinsics(left_geometry, right_geometry)
            else:
                raise RuntimeError(f"There's no way to retrieve relative camera translation and rotation, so stereo images can't be rectified. \
                                   Please either provide relative_rot and relative_trans in source of geometry, or provide a sample \
                                    position and rotation in geometry data.")
        
        # Assign camera geometry
        focal_length = left_geometry.get('fx'), left_geometry.get('fy')
        right_focal_length = right_geometry.get('fx'), right_geometry.get('fy')
        optical_center = left_geometry.get('cx'), left_geometry.get('cy')
        right_optical_center = right_geometry.get('cx'), right_geometry.get('cy')      
        rel_camera_rot = geometry_data.get('relative_rot')
        rel_camera_trans = geometry_data.get('relative_trans')
        baseline = rel_camera_rot[0]
    else:
        # Geometry will be estimated. Initialize geometry with no intrinsic data
        focal_length, right_focal_length = None, None
        optical_center, right_optical_center = None, None
        rel_camera_rot, rel_camera_trans = None, None
        baseline = None
        
    # Initialize rectifier and depth provider module
    if config_dict['depth_source'] == 'stereo':
        # Initialize a rectifier, necessary for correct mapping of pixels across camera frames
        # Already validated geometry source is not estimation
        rectifier = StereoRectifier(focal_length, optical_center, right_focal_length, right_optical_center, image_width, image_height, rel_camera_rot, rel_camera_trans) # TODO: fix this line
        
        # Initialize a stereo depth estimator model
        depth_model = torch.load(config_dict['depth_ckpt'], map_location=device, weights_only=False) # FastFoundationStereo
        depth_model.eval()
        depth_model.to(device)
    else:
        rectifier = None
        # Initialize a mono depth estimator model
        depth_model = UniDepthV2(load_serialized_data(config_dict['depth_config']))
        depth_model.load_state_dict(load_checkpoint(config_dict['depth_ckpt']), strict=False)
        depth_model.resolution_level = 4
    
    depth_provider = DepthProvider(device, depth_model)
    
    # Initialize pose provider module
    if config_dict['pose_source'] == 'metadata':
        # Initialize a pose metadata provider
        pose_model = PoseMetadata(rectifier)
    else:
        # Initialize a pose estimator model
        pose_model = DPVO(load_serialized_data(config_dict['pose_config']), config_dict['pose_ckpt'], ht=image_height, wd=image_width) # FIXME: Set H and W params later
    
    pose_provider = PoseProvider(device, pose_model)
        
    # Initialize detector model and module
    detector_model = RFDETRMedium() # pretrained weights are downloaded within init
    # with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    detector_model.inference()
    detector = Detector(device, detector_model)

    # Initialize tracker model and module
    tracker_model = Predictor(model_args=Namespace(**load_serialized_data(config_dict['tracker_config'])), checkpoint_path=config_dict['tracker_ckpt'], support_grid_size=0)
    tracker_model.eval()
    tracker_model.to(device)
    tracker = Tracker(device, tracker_model)

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

    # Initialize maps for relation predicate names to predicate ids
    # NOTE: Static for current implementation, dynamic if preds are generated
    pred_name_to_id = {name: id for id, name in enumerate(PRED_NAMES)} 
    pred_id_to_name = PRED_NAMES
    
    objects_info = {
        'points': [], # size D list, shape (n, 2) tensors
        'object_point_counts': [], # size D list of integers
        'class_ids': [], # size D list of integers
        'confidences': [], # side D list of integers
    } # Object info container that is updated with each frame    

    # Initialize output strings
    detector_output_prefix = f'outputs/{config_dict['output']}/output_detector'
    tracker_output_prefix = f'outputs/{config_dict['output']}/output_tracker'
    depth_estimator_output_prefix = f'outputs/{config_dict['output']}/output_depth_estimator'
    pose_estimator_output_prefix = f'outputs/{config_dict['output']}/output_pose_estimator'
    point_lifter_output_prefix = f'outputs/{config_dict['output']}/output_point_lifter'
    sgg3d_output_prefix = f'outputs/{config_dict['output']}/output_sgg3d'
    dynamic_sg_output_prefix = f'outputs/{config_dict['output']}/output_dynamic_sg'

    # Initialize other miscellaneous variables before frame loop
    intrinsics_buffer = [] # Buffer to estimate intrinsics after warmup
    pose_to_depth_scale = None
    visualize = config_dict['visualize']
    test_speed = False # Set to True on the frame that speed tests should begin
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t in tqdm(range(len(frames_dir))):
            test_speed = True if t == WARMUP_FRAMES else None

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # Load all necessary data from input directories
            frame = torch.from_numpy(load_frame(str(frames_dir[t]))) # shape: (3, H, W)            
            right_frame = torch.from_numpy(load_frame(str(right_frames_dir[t]))) if right_frames_dir else None
            frame_metadata = load_serialized_data(str(metadata_dir[t])) if metadata_dir else None
            
            # Rectify the images if in stereo
            sys_evaluator.start_speed_test('rectifier') if test_speed else None
            frame, right_frame = rectifier.rectify_pair(frame, right_frame) if rectifier else frame, right_frame
            sys_evaluator.end_speed_test('rectifier') if test_speed else None
            
            # Depth provider
            sys_evaluator.start_speed_test('depth_provider') if test_speed else None
            depth, frame_focal_length, frame_optical_center = depth_provider.process_frame(frame, right_frame=right_frame, focal_length=focal_length, optical_center=optical_center, baseline=baseline)
            sys_evaluator.end_speed_test('depth_provider') if test_speed else None
            
            if focal_length is None or optical_center is None:
                # If still in warmup frames, add results to intrinsics buffer for later averaging
                if t < WARMUP_FRAMES:
                    frame_intrinsics = [frame_focal_length[0], frame_focal_length[1], frame_optical_center[0], frame_optical_center[1]]
                    intrinsics_buffer.append(frame_intrinsics)
                elif t == WARMUP_FRAMES:
                    # Average the intrinsics buffer to get fixed camera intrinsics for the rest of the sequence
                    intrinsics_est = np.median(intrinsics_buffer, axis=0)
                    focal_length = intrinsics_est[0], intrinsics_est[1]
                    optical_center = intrinsics_est[2], intrinsics_est[3]
        
            # Pose provider
            sys_evaluator.start_speed_test('pose_provider') if test_speed else None
            camera_rot, camera_trans = pose_provider.process_frame(
                frame=frame, 
                frame_idx=t, 
                metadata=frame_metadata, 
                focal_length=focal_length if focal_length else frame_focal_length, 
                optical_center=optical_center if optical_center else frame_optical_center
            )
            sys_evaluator.end_speed_test('pose_provider') if test_speed else None

            # # ----- Depth Estimator -----
            # # Process frame using depth estimator
            # sys_evaluator.start_speed_test('depth_estimator') if test_speed else None
            # depth, frame_focal_length_est, frame_optical_center_est = depth_estimator.process_frame(depth_est_input, focal_length=focal_length, baseline=camera_baseline)
            # sys_evaluator.end_speed_test('depth_estimator') if test_speed else None
            # depth_estimator.visualize(depth, output=f'{depth_estimator_output_prefix}_{t:06d}.jpg') if visualize else None

            # if estimate_intrinsics: 
            #     # If still in warmup frames, add results to intrinsics buffer for later averaging
            #     if t < WARMUP_FRAMES:
            #         frame_intrinsics = [frame_focal_length_est[0], frame_focal_length_est[1], frame_optical_center_est[0], frame_optical_center_est[1]]
            #         intrinsics_buffer.append(frame_intrinsics)
            #         # Allow intrinsics to vary frame to frame for now before we freeze it frame number WARMUP_FRAMES
            #         focal_length, optical_center = frame_focal_length_est, frame_optical_center_est
            #     elif t == WARMUP_FRAMES:
            #         # Average the intrinsics buffer to get fixed camera intrinsics for the rest of the sequence
            #         intrinsics_est = np.median(intrinsics_buffer, axis=0)
            #         focal_length = intrinsics_est[0], intrinsics_est[1]
            #         optical_center = intrinsics_est[2], intrinsics_est[3]
            
            # if estimate_pose:
            #     # ----- Pose Estimator -----
            #     # Run pose estimation
            #     sys_evaluator.start_speed_test('pose_estimator')
            #     # Get camera pose from pose estimator
            #     camera_rot, camera_trans = pose_estimator.process_frame(frame, t, focal_length, optical_center)
            #     sys_evaluator.end_speed_test('pose_estimator')
                
            #     if t == WARMUP_FRAMES:
            #          # Get pose to depth scale
            #         pose_to_depth_scale = pose_estimator.get_metric_scaling(t, depth)
                    
            #     # We need to know how to convert between the units these modules use to have accurate 3D camera translation tracking
            #     # Apply scaling if it's been found
            #     if pose_to_depth_scale is not None:
            #         camera_trans = pose_to_depth_scale * camera_trans

            #     pose_estimator.visualize(frame, camera_rot, camera_trans, output=f'{pose_estimator_output_prefix}_{t:06d}.jpg') if visualize else None
                                
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

            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            points3d_representation = point_lifter.lift_points(
                objects_point_list=objects_info['points'], 
                depth=depth, 
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            point_lifter.visualize(
                frame,
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                point_lifter_output=points3d_representation, 
                object_labels=objects_info['class_ids'],
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
            ) if visualize else None
                            
            # ----- 3D Scene Graph Generator -----
            sys_evaluator.start_speed_test('3dsg_gen') if test_speed else None
            scene_graph_3d = scene_graph_generator_3d.generate_triplets(points3d_representation, pred_name_to_id)
            sys_evaluator.end_speed_test('3dsg_gen') if test_speed else None
            scene_graph_generator_3d.visualize(
                frame=frame,
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                scene_graph=scene_graph_3d,
                pred_id_to_name=pred_id_to_name,
                points_representation=points3d_representation,
                object_labels=objects_info['class_ids'],
                output=f'{sgg3d_output_prefix}_{t:06d}.jpg',
                camera_view_mode="aligned",
                show_camera=False,
                auto_zoom=True,
                x_range=(-0.3,0.5),
                y_range=(-0.1,0.4),
                z_range=(1.0, 1.8),
                std_scale=1.0
            ) if visualize else None
            
            if t >= WARMUP_FRAMES:
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
                    optical_center=optical_center,
                    camera_rot=camera_rot,
                    camera_trans=camera_trans,
                    pred_id_to_name=pred_id_to_name,
                    output=f'{dynamic_sg_output_prefix}_{t:06d}.jpg'
                ) if visualize else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None

            # Print metrics
            if t % 50 == 0:
                sys_evaluator.print_latency_metrics()
            
    # ----- Cleanup and evaluation -----
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()