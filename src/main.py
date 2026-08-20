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
from src.dataclasses.tracked_objects import TrackedObjectSet
from src.dataclasses.config_dataclasses import MergeConfig, PointDecayConfig, ReassociationConfig
from src.dataclasses.frame_source import make_frame_source
from src.models.pose_metadata import PoseMetadata
from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.build_geometric_3dsg import Geometric3DSGBuilder
from src.models.merge_gaussian_sg import GaussianSGMerge

# -- lost-tools misc --
from src.args import parse_and_validate_args
from src.utils import pick_device, load_serialized_data, load_checkpoint, \
    compute_rel_camera_extrinsics, unity_pose_to_cv, compute_image_scale_factor

def main() -> None:

    # ----- Initialization setup -----
    config_dict = parse_and_validate_args()
    device = pick_device()
    
    input_source = config_dict['input_source']
    image_width, image_height = config_dict['target_image_size']
    
    # Frames come either from a captured directory or from live headset stream. Both yield InputFrame
    source = make_frame_source(config_dict, target_extent=(image_height, image_width))
    
    # Geometry setup needs frame 0 metadata and resolution, and a live source can't report until either arrives
    # Block here for the first frame, which is held by bootstrap() and re-delivered as the first iteration
    try:
        first_frame = source.bootstrap()
    except BaseException:
        # A live source holds a bound socket and receiver thread, so release them rather than leaving them claimed
        # by a failed start
        source.close()
        raise
    
    orig_image_width, orig_image_height = source.source_size
    image_scale_factor = compute_image_scale_factor((orig_image_width, orig_image_height), (image_width, image_height))

    if first_frame.metadata is not None:
        # TODO: Validate metadata schema
        geometry_data = first_frame.metadata
        
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
        rectifier = StereoRectifier(focal_length, optical_center, right_focal_length, right_optical_center, rel_camera_rot, rel_camera_trans, image_width, image_height)
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
    detector = Detector(
        config_dict['detector']['model_name'], 
        detector_model, device, 
        threshold=config_dict['detector']['threshold'], 
        filter_detection_fraction=config_dict['detector']['filter_detection_fraction'])

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
    merge_config = MergeConfig.from_dict(load_serialized_data(config_dict['3dsg_merging']['config']))
    dynamic_scene_graph_method = GaussianSGMerge(config=merge_config, num_rel_class=len(config_dict['pred_names']))
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
    sg_interval = config_dict['sg_interval']
    decay_config = PointDecayConfig.from_dict(config_dict['point_decay'])
    reassoc_config = ReassociationConfig.from_dict(config_dict['reassociation'])
    # TODO: Move this check to parse_and_validate_args
    if reassoc_config.enabled and reassoc_config.max_depth_diff >= merge_config.broken_track_dist:
        raise ValueError(
            f"reassociation.max_depth_diff ({reassoc_config.max_depth_diff}) must stay below "
            f"broken_track_dist ({merge_config.broken_track_dist})")

    is_stereo = config_dict['depth_source'] == 'stereo'
    has_metadata = config_dict['geometry_source'] == 'metadata' or config_dict['pose_source'] == 'metadata'
    test_speed = False # Set to True on the frame that speed tests should begin
    total_frames = len(source) if hasattr(source, '__len__') else None
    dropped_frames = 0 # Frames the source discarded because the pipeline was behind
    
    # ----- Main loop -----
    # Two clocks:
    #    `t` advances once per frame the source delivers and drives the tracker, which runs
    #      on every frame so object identity never sees a gap. A broken track is what
    #      makes the detector mint a fresh object_id, and a fresh object_id is what
    #      becomes a duplicate scene-graph node.
    #   `sg_step` advances once per frame the expensive 3D stack runs, and drives everything
    #      measured in 3D steps (detector cadence, warmup, merge eviction age, DPVO frame idx, etc.)
    # These two are identical when sg_interval == 1.
    sg_step = 0
    try:
        with torch.inference_mode():
            for frame_data in tqdm(source, total=total_frames):
                t = frame_data.index
                dropped_frames += frame_data.dropped
                run_3d = t % sg_interval == 0
                test_speed = True if t >= warmup_frames else False
                run_detector = True if run_3d and sg_step % detector_interval == 0 else False
                run_global_merge = True if run_3d and global_merge_interval and sg_step % global_merge_interval == 0 else False

                sys_evaluator.start_speed_test('frame') if test_speed else None 

                # Take all necessary data from the frame the source delivered
                # Right frame is only consumed by depth model, so it is left alone on frames skipping 3D stack.
                frame = frame_data.left
                right_frame = frame_data.right if is_stereo and run_3d else None
                frame_metadata = frame_data.metadata if has_metadata else None
                
                # ----- Stereo Rectifier -----
                # Rectify the images if in stereo
                sys_evaluator.start_speed_test('rectifier') if test_speed else None
                frame, right_frame = rectifier.rectify_pair(frame, right_frame) if rectifier else (frame, right_frame)
                sys_evaluator.end_speed_test('rectifier') if test_speed else None
                
                # Convert frames to torch tensor
                frame = torch.from_numpy(np.ascontiguousarray(frame))
                right_frame = torch.from_numpy(np.ascontiguousarray(right_frame)) if right_frame is not None else None
                
                if run_3d:
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
                        frame_idx=sg_step, 
                        metadata=frame_metadata, 
                        focal_length=active_focal_length, 
                        optical_center=active_optical_center
                    )
                    pose_provider.visualize(frame, camera_pos, camera_rot, output=f'{pose_provider_output_prefix}_{t:06d}.jpg') if visualize else None
                    sys_evaluator.end_speed_test('pose_provider') if test_speed else None
                    
                    # Estimate focal length if not already given/rectified/estimated
                    if focal_length is None or optical_center is None:
                        # If still in warmup frames, add results to intrinsics buffer for later averaging
                        if sg_step < warmup_frames:
                            frame_intrinsics = [frame_focal_length[0], frame_focal_length[1], frame_optical_center[0], frame_optical_center[1]]
                            intrinsics_buffer.append(frame_intrinsics)
                        elif sg_step == warmup_frames:
                            pose_to_depth_scale = pose_provider.get_metric_scaling(sg_step, depth)
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

                        # ----- Re-association -----
                        # Before minting ids, ask the scene graph whether any of these
                        # detections is an object it already holds.
                        matched_ids = dynamic_scene_graph.match_detections(
                            detections_info,
                            tracked_objects=objects,
                            depth=depth,
                            focal_length=active_focal_length,
                            optical_center=active_optical_center,
                            camera_rot=camera_rot,
                            camera_pos=camera_pos,
                            config=reassoc_config,
                        )

                        # Retire the stale track holding each matched id so the fresh
                        # seed is that identity's only owner. Order-preserving, so the
                        # same mask compacts the tracker's buffers.
                        retire_mask = objects.retire(matched_ids)
                        if retire_mask is not None:
                            tracker.prune(retire_mask)

                        objects.extend(
                            class_ids=detections_info['class_ids'],
                            confidences=detections_info['class_confidences'],
                            points_list=new_points_list,
                            visibles_list=new_visibles_list,
                            object_ids=matched_ids,
                        )
                        tracker.initialize_queries(frame, new_points_list)
                        check_tracker_sync(tracker, objects, 'detector seeding', t)

                        reassociated = sum(1 for i in matched_ids if i is not None)
                        if reassociated:
                            print(f"[reassoc] frame {t}: {reassociated}/{len(matched_ids)} "
                                  f"detections rejoined existing nodes")
                        
                tracker.visualize(frame, objects.points, objects.visibles, output=f'{tracker_output_prefix}_{t:06d}.jpg') if visualize else None

                if run_3d:
                    # ----- Point Lifting to 3D -----
                    # Filter objects to only those that were observed, with a (# of points observed / total_points) >= min_visible_frac
                    observed_objects = objects.observed(min_visible_frac=0.5)
                    sys_evaluator.start_speed_test('point_lifter') if test_speed else None
                    # Lift these objects into 3D and call them `observations`, which are class Observation3D 
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
                    if sg_step >= warmup_frames:
                        # ----- 3D Scene Graph Merging -----
                        sys_evaluator.start_speed_test('3dsg_merge')
                        update_idx = dynamic_scene_graph.add(
                            observations=observations, 
                            triplets=scene_graph_3d,
                            frame_num=sg_step
                        )
                        dynamic_scene_graph.merge(update_idx, sg_step, run_global_merge)
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
                        
                    sg_step += 1
                            
                # ----- Tracker point decay -----
                # Trim the tracker's working set before the next frame, so its
                # cost stops growing with session length. Points are dropped from
                # within objects, never whole objects, so object_id -- and with it
                # the scene graph's identity association -- survives.
                if decay_config.enabled:
                    sys_evaluator.start_speed_test('point_decay') if test_speed else None
                    keep_mask = objects.plan_decay(decay_config)
                    if keep_mask is not None:
                        tracker.prune(keep_mask)
                        objects.apply_decay(keep_mask)
                    sys_evaluator.end_speed_test('point_decay') if test_speed else None
                
                sys_evaluator.end_speed_test('frame') if test_speed else None

                # Print metrics
                if "frame" in sys_evaluator.eval_dict and t % 10 == 0:
                    sys_evaluator.print_latency_metrics()
    except KeyboardInterrupt:
        print("\n[main] Stopped by KeyboardInterrupt")
    finally:
        source.close()
        
    # ----- Cleanup and evaluation -----
    # Print metrics
    if dropped_frames:
        print(f"[main] Source dropped {dropped_frames} frames to keep the stream fresh")
    
    # A live session stopped before warmup has no frame timings to report, and
    # the report requires them.
    if 'frame' in sys_evaluator.eval_dict:
        sys_evaluator.print_latency_metrics()
    else:
        print(f"[main] Stopped before frame {warmup_frames} (warmup_frames), no latency metrics collected")
        
def check_tracker_sync(tracker, objects, stage, t) -> None:
    """
    Fail where the tracker's query set and the object registry diverge, rather
    than several frames later inside torch.split.

    The tracker returns one row per active query in query order, so every
    mutation of one side has to be mirrored on the other: extend pairs with
    initialize_queries, apply_decay pairs with prune. This names which pairing
    broke and on which frame.
    """
    model_n = getattr(tracker.model, 'N', None)
    if model_n is None or model_n == objects.total_points:
        return
    raise RuntimeError(
        f"[frame {t}] tracker/registry desync after {stage}: tracker holds {model_n} points, "
        f"registry accounts for {objects.total_points} across {len(objects)} objects. "
        f"point_counts={objects.point_counts}")


if __name__ == "__main__":
    main()