import torch
import numpy as np
from pathlib import Path
from argparse import Namespace
import warnings
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message=r'`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated',
)

# Add external repo roots to sys.path
_repo_root = Path(__file__).resolve().parent.parent
ffs_path = _repo_root / "external" / "fast_foundationstereo"
if str(ffs_path) not in sys.path:
    sys.path.insert(0, str(ffs_path))

# track_on resolves its own modules absolutely (`from model.trackon import ...`)
trackon_path = _repo_root / "external" / "track_on"
if str(trackon_path) not in sys.path:
    sys.path.insert(0, str(trackon_path))

# Pure-PyTorch stand-in for mmcv.ops.MultiScaleDeformableAttention. Appended, not
# inserted, so a real mmcv install wins if one is present. See src/compat/mmcv/.
compat_path = _repo_root / "src" / "compat"
if str(compat_path) not in sys.path:
    sys.path.append(str(compat_path))

# Restore the DINOv3 attribute layout track_on expects (see src/compat/trackon_compat.py).
# Must run before the Track-On Predictor is constructed.
from src.compat.trackon_compat import apply as apply_trackon_compat
apply_trackon_compat()

# -- external model imports --
from external.rfdetr.src.rfdetr import RFDETRMedium
from external.track_on.model.trackon_predictor import Predictor
# from core.foundation_stereo import FastFoundationStereo

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
from src.dataclasses.tracked_objects import TrackedObjectSet
from src.dataclasses.config_dataclasses import MergeConfig
from src.models.merge_gaussian_sg import GaussianSGMerge
# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from src.args import parse_and_validate_args
from src.utils import pick_device, load_serialized_data, load_frame, load_checkpoint, frames_sort_key, \
    compute_rel_camera_extrinsics, unity_pose_to_cv, compute_image_scale_factor

def main() -> None:

    # ----- Initialization setup -----
    config_dict = parse_and_validate_args()
    device = pick_device()

    frames_dir = sorted([f for f in Path(config_dict['input']).iterdir()], key=frames_sort_key) # Sort input frame seq
    _, orig_image_height, orig_image_width = load_frame(frames_dir[0]).shape # Grab the first frame to get image width and height
    image_width, image_height = config_dict['target_image_size'] 
    image_scale_factor = compute_image_scale_factor((orig_image_width, orig_image_height), (image_width, image_height))

    if config_dict['input_right'] is not None:
        right_frames_dir = sorted([f for f in Path(config_dict['input_right']).iterdir()], key=frames_sort_key)
        assert len(right_frames_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but right_frames_dir has {len(frames_dir)} frames"

    if config_dict['input_metadata'] is not None:
        # TODO: Validate metadata schema
        metadata_dir = sorted([f for f in Path(config_dict['input_metadata']).iterdir()], key=frames_sort_key)
        assert len(metadata_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but metadata_dir has {len(metadata_dir)} frames"

        geometry_data = load_serialized_data(str(metadata_dir[0]))
        
    if config_dict['input_geometry'] is not None:
        # TODO: Behavior for supplying intrinsics and relative extrinsics directly from config
        geometry_data = load_serialized_data(config_dict['input_geometry'])
    
    # TODO: move this to a utility function
    if config_dict['input_camera_coords'] == 'unity':
        coord_conversion_func = unity_pose_to_cv
    else:
        coord_conversion_func = lambda x: x
        
    # Initialize geometry
    if config_dict['geometry_source'] in ('metadata', 'input'):
        # Initialize geometry with intrinsic data
        left_geometry, right_geometry = geometry_data.get('leftCamera'), geometry_data.get('rightCamera')
        if left_geometry is None or right_geometry is None:
            raise KeyError(f"Tried accessing left and right geometry using keys leftCamera and rightCamera, \
                           but got {left_geometry} for leftCamera and {right_geometry} for rightCamera")

        # Check if we're missing relative rotation and translation info
        if geometry_data.get('relative_trans') is None or geometry_data.get('relative_rot') is None:
            # If we have access to a sample world position and rotation from geometry_data, compute relative extrinsics directly
            has_world_trans = 'pos' in left_geometry.keys() and 'pos' in right_geometry.keys()
            has_world_rot = 'rot' in left_geometry.keys() and 'rot' in right_geometry.keys()
            
            if has_world_trans and has_world_rot:
                geometry_data['relative_trans'], geometry_data['relative_rot'] = compute_rel_camera_extrinsics(left_geometry, right_geometry, coord_conversion_func)
            else:
                raise RuntimeError(f"There's no way to retrieve relative camera translation and rotation, so stereo images can't be rectified. \
                                   Please either provide relative_rot and relative_trans in source of geometry, or provide a sample \
                                    position and rotation in geometry data.")
        
        # Assign camera geometry
        focal_length = left_geometry.get('fx') * image_scale_factor, left_geometry.get('fy') * image_scale_factor
        right_focal_length = right_geometry.get('fx') * image_scale_factor, right_geometry.get('fy') * image_scale_factor
        optical_center = left_geometry.get('cx') * image_scale_factor, left_geometry.get('cy') * image_scale_factor
        right_optical_center = right_geometry.get('cx') * image_scale_factor, right_geometry.get('cy') * image_scale_factor  
        rel_camera_rot = geometry_data.get('relative_rot')
        rel_camera_trans = geometry_data.get('relative_trans')
    else:
        # Geometry will be estimated. Initialize geometry with no intrinsic data
        focal_length, right_focal_length = None, None
        optical_center, right_optical_center = None, None
        rel_camera_rot, rel_camera_trans = None, None
        
    # Initialize rectifier and depth provider module
    if config_dict['depth_source'] == 'stereo':
        # Initialize a rectifier, necessary for correct mapping of pixels across camera frames
        # Already validated geometry source is not estimation
        rectifier = StereoRectifier(focal_length, optical_center, right_focal_length, right_optical_center, rel_camera_rot, rel_camera_trans, image_width, image_height) # TODO: fix this line
        focal_length = rectifier.rectified_focal_length
        optical_center = rectifier.rectified_optical_center
        baseline = rectifier.baseline
        
        # Initialize a stereo depth estimator model
        depth_model = torch.load(config_dict['depth_provider']['ckpt'], map_location=device, weights_only=False) # FastFoundationStereo
        depth_config = load_serialized_data(config_dict['depth_provider']['config'])
        depth_model.args.max_disp = depth_config['max_disp']
        depth_model.args.mixed_precision = depth_config['mixed_precision']
        depth_model.args.valid_iters = depth_config['valid_iters']
        depth_model.eval()
        depth_model.to(device)

    else:
        rectifier = None
        baseline = None

        # Initialize a mono depth estimator model
        from external.unidepth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
        depth_model = UniDepthV2(load_serialized_data(config_dict['depth_provider']['config']))
        depth_model.load_state_dict(load_checkpoint(config_dict['depth_provider']['ckpt']), strict=False)
        depth_model.resolution_level = 4
    
    depth_model.eval()
    depth_model.to(device)
    depth_provider = DepthProvider(config_dict['depth_provider']['model_name'], depth_model, device)
    
    # Initialize pose provider module
    if config_dict['pose_source'] == 'metadata':
        # Initialize a pose metadata provider
        pose_model = PoseMetadata(rectifier, coord_conversion_func=coord_conversion_func)
    else:
        # Initialize a pose estimator model
        from external.DPVO.dpvo.dpvo import DPVO
        pose_model = DPVO(load_serialized_data(config_dict['pose_provider']['config']), config_dict['pose_provider']['ckpt'], ht=image_height, wd=image_width) # FIXME: Set H and W params later
    
    pose_provider = PoseProvider(config_dict['pose_provider']['model_name'], pose_model, device)
        
    # Initialize detector model and module
    detector_model = RFDETRMedium() # pretrained weights are downloaded within init
    detector_model.inference()
    detector = Detector(config_dict['detector']['model_name'], detector_model, device)

    # Initialize tracker model and module
    tracker_model = Predictor(model_args=Namespace(**load_serialized_data(config_dict['tracker']['config'])), checkpoint_path=config_dict['tracker']['ckpt'], support_grid_size=0)
    tracker_model.eval()
    tracker_model.to(device)
    tracker = Tracker(
        name=config_dict['tracker']['model_name'], 
        model=tracker_model, 
        device=device,
        max_grid_size=config_dict['tracker']['max_grid_size'],
    )

    # Initialize point lifting method and module
    # FIXME: pass in args to configure
    point_lifting_method = Gaussian3DLift()
    point_lifter = PointLifter(config_dict['point_lifter']['model_name'], point_lifting_method)

    # Initialize 3D scene graph generator method and module
    # FIXME: pass in args to configure
    scene_graph_gen_3d_method = Geometric3DSGBuilder()
    scene_graph_generator_3d = SceneGraphGenerator3D(config_dict['3dsgg']['model_name'], sgg_method=scene_graph_gen_3d_method, point_lifting_method_name=config_dict['point_lifter']['model_name'])
    
    # Initialize dynamic 3D scene graph class
    dynamic_scene_graph_method = GaussianSGMerge(config=MergeConfig.from_dict(load_serialized_data(config_dict['3dsg_merging']['config'])), num_rel_class=len(config_dict['pred_names']))
    dynamic_scene_graph = DynamicSceneGraph3D(config_dict['3dsg_merging']['model_name'], dynamic_scene_graph_method)
    
    # Initialize system evaluator module for metrics
    sys_evaluator = SystemEvaluator(device=device)

    # Initialize maps for relation predicate names to predicate ids
    # NOTE: Static for current implementation, dynamic if preds are generated
    pred_name_to_id = {name: id for id, name in enumerate(config_dict['pred_names'])} 
    pred_id_to_name = config_dict['pred_names']
    
    objects = TrackedObjectSet() # Object info container that is updated with each frame    

    # TODO: comment description
    torch.backends.cudnn.benchmark = True

    # Initialize output strings
    detector_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["detector"]["output_suffix"]}/output_detector'
    tracker_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["tracker"]["output_suffix"]}/output_tracker'
    depth_provider_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["depth_provider"]["output_suffix"]}/output_depth_provider'
    pose_provider_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["pose_provider"]["output_suffix"]}/output_pose_provider'
    point_lifter_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["point_lifter"]["output_suffix"]}/output_point_lifter'
    sgg3d_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["3dsgg"]["output_suffix"]}/output_sgg3d'
    dynamic_sg_output_prefix = f'{config_dict["output_prefix"]}/{config_dict["3dsg_merging"]["output_suffix"]}/output_dynamic_sg'

    # Initialize other miscellaneous variables before frame loop
    intrinsics_buffer = [] # Buffer to estimate intrinsics after warmup
    pose_to_depth_scale = None
    visualize = config_dict['visualize']
    warmup_frames = config_dict['warmup_frames']
    detector_interval = config_dict['detector_interval']
    global_merge_interval = config_dict['global_merge_interval']
    is_stereo = config_dict['depth_source'] == 'stereo'
    has_metadata = config_dict['geometry_source'] == 'metadata' or config_dict['pose_source'] == 'metadata'
    test_speed = False # Set to True on the frame that speed tests should begin
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t in tqdm(range(len(frames_dir))):
            test_speed = True if t >= warmup_frames else False
            run_detector = True if t % detector_interval == 0 else False
            run_global_merge = True if global_merge_interval and t % global_merge_interval == 0 else False
                

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # Load all necessary data from input directories
            frame = load_frame(str(frames_dir[t]), extent=(image_height, image_width)) # shape: (3, H, W)            
            right_frame = load_frame(str(right_frames_dir[t]), extent=(image_height, image_width)) if is_stereo else None
            frame_metadata = load_serialized_data(str(metadata_dir[t])) if has_metadata else None
            
            # ----- Stereo Rectifier -----
            # Rectify the images if in stereo
            sys_evaluator.start_speed_test('rectifier') if test_speed else None
            frame, right_frame = rectifier.rectify_pair(frame, right_frame) if rectifier else (frame, right_frame)
            sys_evaluator.end_speed_test('rectifier') if test_speed else None
            
            # Convert frames to torch tensor
            frame = torch.from_numpy(np.ascontiguousarray(frame))
            right_frame = torch.from_numpy(np.ascontiguousarray(right_frame)) if right_frame is not None else None
            
            # ----- Depth Provider -----
            sys_evaluator.start_speed_test('depth_provider') if test_speed else None
            depth, frame_focal_length, frame_optical_center = depth_provider.process_frame(
                frame, 
                right_frame=right_frame, 
                focal_length=focal_length, 
                optical_center=optical_center, 
                baseline=baseline
            )
            sys_evaluator.end_speed_test('depth_provider') if test_speed else None
            depth_provider.visualize(depth, output=f'{depth_provider_output_prefix}_{t:06d}.jpg') if visualize else None
            
            # Assign the active focal length and optical center (frame estimate vs constant)
            active_focal_length = focal_length if focal_length is not None else frame_focal_length
            active_optical_center = optical_center if optical_center is not None else frame_optical_center
                    
            # ----- Pose provider -----
            sys_evaluator.start_speed_test('pose_provider') if test_speed else None
            camera_pos, camera_rot = pose_provider.process_frame(
                frame=frame, 
                frame_idx=t, 
                metadata=frame_metadata, 
                focal_length=active_focal_length, 
                optical_center=active_optical_center
            )
            pose_provider.visualize(frame, camera_pos, camera_rot, output=f'{pose_provider_output_prefix}/{t:06d}.jpg') if visualize else None
            sys_evaluator.end_speed_test('pose_provider') if test_speed else None
            
            # Estimate focal length if not already given/rectified/estimated
            if focal_length is None or optical_center is None:
                # If still in warmup frames, add results to intrinsics buffer for later averaging
                if t < warmup_frames:
                    frame_intrinsics = [frame_focal_length[0], frame_focal_length[1], frame_optical_center[0], frame_optical_center[1]]
                    intrinsics_buffer.append(frame_intrinsics)
                elif t == warmup_frames:
                    pose_to_depth_scale = pose_provider.get_metric_scaling(t, depth)
                    # Average the intrinsics buffer to get fixed camera intrinsics for the rest of the sequence
                    intrinsics_est = np.median(intrinsics_buffer, axis=0)
                    focal_length = intrinsics_est[0], intrinsics_est[1]
                    optical_center = intrinsics_est[2], intrinsics_est[3]
            
            # We need to know how to convert between the units these modules use to have accurate 3D camera translation tracking
            # Apply scaling if it's been found
            if pose_to_depth_scale is not None:
                camera_pos = pose_to_depth_scale * camera_pos
                                
            # ----- Tracker -----
            
            # Process frame if there are active points 
            if objects.total_points > 0:
                # Set tracker initial capacity based on object point count
                tracker.model.initial_capacity = objects.total_points # TODO: make general for any tracker
                # Process frame using tracker
                sys_evaluator.start_speed_test('tracker') if test_speed else None
                points_list, visibles_list = tracker.process_frame(frame, objects.point_counts)
                sys_evaluator.end_speed_test('tracker') if test_speed else None 

                objects.update_from_tracker(points_list, visibles_list)

            if run_detector:
                # ----- Detector -----
                # Process frame using detector
                sys_evaluator.start_speed_test('detector') if test_speed else None
                detections_info = detector.process_frame(frame)
                sys_evaluator.end_speed_test('detector') if test_speed else None
                detector.visualize(frame, detections_info, output=f'{detector_output_prefix}_{t:06d}.jpg') if visualize else None
                
                # Filter detections againt updated tracker point positions
                detections_info = detector.filter_detections_info(detections_info, objects.points, objects.class_ids)

                # TODO: comment description
                if detections_info is not None:
                    # Using the detector bbox info, greate grid of queries for each object
                    new_points_list, new_object_point_counts = tracker.build_detection_grid_points(
                        detections_info, 
                        frame_extent=(image_height, image_width), 
                        margin_div=16
                    )

                    new_visibles_list = [
                        torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device) 
                        for pts in new_points_list
                    ]

                    objects.extend(
                        class_ids=detections_info['class_ids'],
                        confidences=detections_info['class_confidences'],
                        points_list=new_points_list,
                        visibles_list=new_visibles_list
                    )
                    tracker.initialize_queries(frame, new_points_list)
                    
            tracker.visualize(frame, objects.points, objects.visibles, output=f'{tracker_output_prefix}_{t:06d}.jpg') if visualize else None

            # ----- Point Lifting to 3D -----
            observed_objects = objects.observed(min_visible_frac=0.5)
            # TODO: comment description
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            observations = point_lifter.lift_points(
                tracked_objects=observed_objects, 
                depth=depth, 
                focal_length=active_focal_length,
                optical_center=active_optical_center,
                camera_pos=camera_pos,
                camera_rot=camera_rot,
                baseline=baseline
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            point_lifter.visualize(
                frame,
                focal_length=active_focal_length,
                optical_center=active_optical_center,
                camera_pos=camera_pos,
                camera_rot=camera_rot,
                observations=observations, 
                object_labels=observations.class_ids,
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
            ) if visualize else None
                            
            # ----- 3D Scene Graph Generator -----
            # TODO: comment description
            sys_evaluator.start_speed_test('3dsg_gen') if test_speed else None
            scene_graph_3d = scene_graph_generator_3d.generate_triplets(observations, pred_name_to_id)
            sys_evaluator.end_speed_test('3dsg_gen') if test_speed else None
            scene_graph_generator_3d.visualize(
                frame=frame,
                focal_length=active_focal_length,
                optical_center=active_optical_center,
                camera_rot=camera_rot,
                camera_pos=camera_pos,
                scene_graph=scene_graph_3d,
                pred_id_to_name=pred_id_to_name,
                observations=observations,
                object_labels=observations.class_ids,
                output=f'{sgg3d_output_prefix}_{t:06d}.jpg',
                camera_view_mode="aligned",
                show_camera=False,
                auto_zoom=True,
                x_range=(-0.3,0.5),
                y_range=(-0.1,0.4),
                z_range=(1.0, 1.8),
                std_scale=1.0
            ) if visualize else None
            
            # TODO: comment description
            if t >= warmup_frames:
                # ----- 3D Scene Graph Merging -----
                sys_evaluator.start_speed_test('3dsg_merge')
                update_idx = dynamic_scene_graph.add(
                    observations=observations, 
                    triplets=scene_graph_3d,
                    frame_num=t
                )
                dynamic_scene_graph.merge(update_idx, t, run_global_merge)
                sys_evaluator.end_speed_test('3dsg_merge')
                dynamic_scene_graph.visualize(
                    frame=frame,
                    focal_length=active_focal_length,
                    optical_center=active_optical_center,
                    camera_rot=camera_rot,
                    camera_pos=camera_pos,
                    pred_id_to_name=pred_id_to_name,
                    output=f'{dynamic_sg_output_prefix}_{t:06d}.jpg'
                ) if visualize else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None

            # Print metrics
            if "frame" in sys_evaluator.eval_dict and t % 10 == 0:
                sys_evaluator.print_latency_metrics()
            
    # ----- Cleanup and evaluation -----
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()