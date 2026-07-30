import numpy as np

from src.models.lift_gaussian_3d import Gaussian3DLift
from external.FROSS.Merging.utils import GaussianSG

class DynamicSceneGraph3D:
    
    def __init__(self, point_lifting_method, num_rel_class=2, merge_threshold=0.85):
        if isinstance(point_lifting_method, Gaussian3DLift):
            self.dynamic_sg = GaussianSG(num_rel_class, merge_threshold)
            
    def add(self, object_labels, points_representation, triplets):
        
        if isinstance(self.dynamic_sg, GaussianSG):
            means, covs, pcds = points_representation
            
            rels = [[subj, obj] for subj, pred, obj in triplets]
            rels_np = np.array(rels, dtype=np.int64)
            rel_classes = [pred for subj, pred, obj in triplets]
            rel_classes_np = np.array(rel_classes, dtype=np.int64)
            
            update_idx = self.dynamic_sg.add(
                new_classes=object_labels,
                new_means=means,
                new_covs=covs,
                new_rels=rels_np,
                new_rel_classes=rel_classes_np,
                new_pcds=pcds
            )
            
    def merge(self, update_idx):
        if isinstance(self.dynamic_sg, GaussianSG):
            self.dynamic_sg.merge(update_idx)