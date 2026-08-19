from dataclasses import dataclass
from typing import Any, List
import numpy as np

@dataclass
class Observations3D:
    """
    3D observations for one frame.
    """
    means: Any                  # (M, 3) tensor
    covs: Any                   # (M, 3, 3) tensor
    point_clouds: List[Any]     # M arrays of (n, 3) tensors
    object_ids: List[int]       # M persistent ids, aligned by construction
    class_ids: List[int]        # M class ids
    
    def __post_init__(self):
        m = len(self.means)
        assert len(self.covs) == m and len(self.point_clouds) == m and len(self.object_ids) == m and len(self.class_ids) == m, \
            f"Observations3D fields must be aligned, but got: {len(self.means)=}, \
            {len(self.covs)=}, {len(self.point_clouds)=}, {len(self.object_ids)=}, {len(self.class_ids)=}"
            
    def __len__(self): return len(self.means)
    
    @property
    def extents(self):
        """
        Property defining how to compute the extents for lifted 3D observations

        Returns:
            _type_: _description_
        """
        return np.sqrt(np.diagonal(self.covs, axis1=1, axis2=2))