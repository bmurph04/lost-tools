import numpy as np

from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.merge_gaussian_sg import GaussianSGMerge
class DynamicSceneGraph3D:
    
    def __init__(self, name, point_lifting_method_name, num_rel_class=2, merge_threshold=0.5):
        self.name = name
        if point_lifting_method_name == 'gaussian_3d_lift':
            self.dynamic_sg = GaussianSGMerge(num_rel_class, merge_threshold)
            
    def add(self, observations, triplets):
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
            )
            
    def merge(self, update_idx):
        if self.name == '3d_gaussian_merging':
            self.dynamic_sg.merge(update_idx)

    def visualize(self, frame, pred_id_to_name, output, focal_length=None, optical_center=None, camera_rot=None, camera_pos=None, camera_view_mode="isometric"):

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