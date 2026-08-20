import numpy as np

from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.merge_gaussian_sg import GaussianSGMerge
class DynamicSceneGraph3D:
    
    def __init__(self, name, dynamic_sg):
        self.name = name
        self.dynamic_sg = dynamic_sg
            
    def add(self, observations, triplets, frame_num):
        if self.name == '3d_gaussian_merging':
            rels = np.array([[s, o] for s, p, o in triplets], dtype=np.int64)
            rel_classes = np.array([p for s, p, o in triplets], dtype=np.int64)
            
            return self.dynamic_sg.add(
                new_classes=observations.class_ids,
                new_means=observations.means,
                new_covs=observations.covs,
                new_rels=rels,
                new_rel_classes=rel_classes,
                new_pcds=observations.point_clouds,
                object_ids=observations.object_ids,
                frame_num=frame_num
            )
            
    def merge(self, update_idx, frame_num, global_merge=False):
        if self.name == '3d_gaussian_merging':
            self.dynamic_sg.merge(update_idx, frame_num, global_merge)
    
    def match_detections(self, detections_info, tracked_objects, depth,
                         focal_length, optical_center, camera_rot, camera_pos, config):
        """
        Decide which detections are objects the graph already knows.

        Returns a list with one entry per detection: an existing object_id to
        reuse, or None to mint a fresh one.

        Matching is done in 2D at detection time because a same-class detection
        overlapping a node's projection is far stronger evidence of identity than
        3D Hellinger distance, and is not subject to depth noise. Doing it here
        also means the pair never reaches the geometric merge pass at all.
        """
        num_detections = len(detections_info['class_ids'])
        matches = [None] * num_detections

        if self.name != '3d_gaussian_merging' or not config.enabled or num_detections == 0:
            return matches
        if camera_rot is None or camera_pos is None or focal_length is None or optical_center is None:
            return matches

        nodes, owners, means, classes = self.dynamic_sg.candidate_nodes()
        if nodes.size == 0:
            return matches

        # Skip nodes whose track is still healthy: re-seeding one would throw away
        # a working track in order to "repair" it.
        visible_frac = {o.object_id: o.visible_fraction for o in tracked_objects}
        stale = np.array(
            [visible_frac.get(int(tid), 0.0) < config.max_visible_frac for tid in owners],
            dtype=bool)
        if not stale.any():
            return matches
        nodes, owners, means, classes = nodes[stale], owners[stale], means[stale], classes[stale]

        uv, z = Gaussian3DLift.project_to_image(
            means, focal_length, optical_center, camera_rot, camera_pos)

        depth_np = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
        depth_np = depth_np.squeeze()
        height, width = depth_np.shape

        coordinates = detections_info['coordinates']
        detection_classes = detections_info['class_ids']

        # Score every (detection, node) pair that clears the gates, then resolve
        # greedily best-first so one node cannot claim two detections.
        candidates = []
        for d in range(num_detections):
            x_min, y_min, x_max, y_max = (float(value) for value in coordinates[d])
            box_cx, box_cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

            # Depth at the box centre separates same-class objects that overlap in
            # the image, one occluding the other. Image overlap alone cannot.
            px = int(np.clip(round(box_cx), 0, width - 1))
            py = int(np.clip(round(box_cy), 0, height - 1))
            detection_depth = float(depth_np[py, px])

            for k in range(nodes.shape[0]):
                if int(classes[k]) != int(detection_classes[d]):
                    continue
                if z[k] <= 0:                                   # behind the camera
                    continue

                u, v = uv[k]
                margin = config.bbox_margin
                if not (x_min - margin <= u <= x_max + margin and
                        y_min - margin <= v <= y_max + margin):
                    continue

                if detection_depth > 0:
                    depth_diff = abs(detection_depth - float(z[k]))
                    if depth_diff > config.max_depth_diff:
                        continue
                else:
                    depth_diff = config.max_depth_diff          # no depth: rank last

                centre_dist = float(np.hypot(u - box_cx, v - box_cy))
                candidates.append((depth_diff, centre_dist, d, int(owners[k])))

        used_detections, used_owners = set(), set()
        for _, _, d, object_id in sorted(candidates):
            if d in used_detections or object_id in used_owners:
                continue
            matches[d] = object_id
            used_detections.add(d)
            used_owners.add(object_id)

        return matches

    def visualize(self, frame, pred_id_to_name, output, focal_length=None, optical_center=None, camera_rot=None, camera_pos=None, camera_view_mode="aligned"):

        if self.name == '3d_gaussian_merging':
            valid_indices = np.flatnonzero(self.dynamic_sg._valid_mask)
            means = self.dynamic_sg._means[valid_indices]
            covs = self.dynamic_sg._covs[valid_indices]
            labels = self.dynamic_sg._classes[valid_indices]

            relation_indices = np.argwhere(self.dynamic_sg._rels > 0) # Each row is [subject_id, object_id, predicate_id]
            triplets = [
                (int(subject_id), int(predicate_id), int(object_id))
                for subject_id, object_id, predicate_id in relation_indices
                if self.dynamic_sg._valid_mask[subject_id]
                and self.dynamic_sg._valid_mask[object_id]
            ]            

            Gaussian3DLift.visualize_3d_gaussians_in_3d(
                means_3d=means,
                covs_3d=covs,
                labels=labels,
                triplets=triplets,
                pred_id_to_name=pred_id_to_name,
                output_path=output,
                camera_rot=camera_rot,
                camera_pos=camera_pos,
                camera_view_mode=camera_view_mode
            )