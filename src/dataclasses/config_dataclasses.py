from dataclasses import dataclass, fields

@dataclass(frozen=True)
class MergeConfig:
    merge_threshold: float = 0.7
    max_merge_dist: float = 0.30
    broken_track_dist: float = 0.50
    pcd_cap: int = 20000
    evict_age: int = 150
    evict_min_obs: int = 3
    global_threshold: float = 0.85
    global_max_dist: float = 0.60
    global_disjoint: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})