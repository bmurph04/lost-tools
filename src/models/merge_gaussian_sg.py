"""GaussianSG with a corrected merge step.

Subclasses the vendored FROSS class rather than monkeypatching it, so the merge
policy lives with the rest of our scene-graph code.

Changes vs upstream:
  * Non-finite distances are masked before argmin. Upstream took argmin first,
    so a single degenerate candidate returned nan and `nan < threshold` vetoed a
    merge that a different candidate qualified for.
  * The survivor keeps absorbing until nothing else qualifies, re-evaluating
    after each merge, instead of merging only the single argmin candidate.
  * A hard separation gate, so covariance shape mismatch cannot authorise
    dragging a centroid across the room.
  * Covariance is a running weighted average. Upstream added a between-group
    term that grew cov on every merge (~12%/frame measured), silently loosening
    the merge gate until distinct objects were swallowed.
  * Point clouds are capped, with merge weights tracked separately so the
    statistics are unaffected by capping.
  * Unconfirmed nodes are evicted, so a phantom from one bad observation cannot
    persist forever.
"""

import os

import numpy as np

from external.FROSS.Merging.utils import GaussianSG
from src.dataclasses.config_dataclasses import MergeConfig

DEBUG = os.environ.get('LOST_TOOLS_SG_DEBUG', '') == '1'

class GaussianSGMerge(GaussianSG):
    def __init__(self, num_rel_class, config: MergeConfig):
        super().__init__(num_rel_class, config.merge_threshold)
        self.config = config
        self._point_counts = np.zeros(self._max_size, dtype=np.int64) # Stores number of points that have contributed to each object
        self._observation_counts = np.zeros(self._max_size, dtype=np.int64) # Stores number of times each object has been observed
        self._frame_last_seen = np.zeros(self._max_size, dtype=np.int64) # Stores frame on which each object was last seen
        self._track_node = {}       # persistent object_id -> node slot

    # -- bookkeeping --------------------------------------------------------

    def _sync_tables(self):
        """Grow the side tables when _expand_if_needed grows the slot arrays."""
        n = self._valid_mask.shape[0]
        if self._point_counts.shape[0] == n:
            return
        for name in ('_point_counts', '_observation_counts', '_frame_last_seen'):
            old = getattr(self, name)
            grown = np.zeros(n, dtype=np.int64)
            m = min(n, old.shape[0])
            grown[:m] = old[:m]
            setattr(self, name, grown)

    def _point_count_of(self, idx):
        if self._point_counts[idx] > 0:
            return int(self._point_counts[idx])
        pcd = self._pcd[idx]
        return 0 if pcd is None else len(pcd)

    # -- overrides ----------------------------------------------------------

    def add(self, new_classes, new_means, new_covs, new_rels, new_rel_classes,
            new_pcds, frame_num, object_ids=None):
        idxs = super().add(new_classes, new_means, new_covs, new_rels,
                           new_rel_classes, new_pcds)
        self._sync_tables()
        self._observation_counts[idxs] = 1
        self._frame_last_seen[idxs] = frame_num
        self._point_counts[idxs] = 0

        if object_ids is None:
            return idxs

        assert len(object_ids) == len(idxs), (
            f'object_ids ({len(object_ids)}) must align with observations '
            f'({len(idxs)})')

        # Identity association: a track that already owns a node folds straight
        # into it, no distance test. The tracker asserting continuity is stronger
        # evidence than 3D proximity, and is immune to depth noise.
        survivors = []
        for k, tid in enumerate(object_ids):
            new_idx = idxs[k]
            node = self._track_node.get(tid)
            broken = (node is not None and self._valid_mask[node]
                      and np.linalg.norm(self._means[new_idx] - self._means[node])
                      > self.config.broken_track_dist)
            if broken:
                # Track jumped implausibly far -- assume it lost lock rather than
                # dragging a good node across the room. Start a fresh node.
                if DEBUG:
                    print(f'[sg] BROKEN TRACK tid={tid}: node={node} dropped')
                del self._track_node[tid]
                node = None
            if (node is not None and node != new_idx and self._valid_mask[node]
                    and self._classes[node] == self._classes[new_idx]):
                self._merge_gaussians(new_idx, node)
                survivors.append(node)
            else:
                self._track_node[tid] = new_idx
                survivors.append(new_idx)

        # Survivors still go through the geometric pass, which now has only one
        # job left: consolidating separate tracks of the same physical object.
        return np.unique(np.array(survivors, dtype=np.int64))

    def _merge_gaussians(self, idx1, idx2):
        """Merge idx1 into idx2, keeping a running-average covariance."""
        mean1, cov1 = self._means[idx1], self._covs[idx1]
        mean2, cov2 = self._means[idx2], self._covs[idx2]

        n1 = self._point_count_of(idx1)
        n2 = self._point_count_of(idx2)
        if n1 == 0 and n2 == 0:
            n1 = n2 = 1
        total = n1 + n2

        self._means[idx2] = (n1 * mean1 + n2 * mean2) / total
        # Weighted average only. Upstream also added
        #   n1*n2*outer(mean1-mean2) / total**2
        # which compounds every merge and relaxes the gate over time.
        self._covs[idx2] = (n1 * cov1 + n2 * cov2) / total

        self._rels[idx2, self._valid_mask] += self._rels[idx1, self._valid_mask]
        self._rels[self._valid_mask, idx2] += self._rels[self._valid_mask, idx1]

        pcd1 = self._pcd[idx1] if self._pcd[idx1] is not None else np.empty((0, 3))
        pcd2 = self._pcd[idx2] if self._pcd[idx2] is not None else np.empty((0, 3))
        merged = np.concatenate((pcd1, pcd2))
        if self.config.pcd_cap > 0 and len(merged) > self.config.pcd_cap:
            merged = merged[np.random.choice(len(merged), self.config.pcd_cap, replace=False)]
        self._pcd[idx2] = merged

        self._point_counts[idx2] = total          # true weight, ignores the cap
        self._observation_counts[idx2] = max(1, self._observation_counts[idx2]) + max(1, self._observation_counts[idx1])
        self._frame_last_seen[idx2] = max(self._frame_last_seen[idx1], self._frame_last_seen[idx2])
        for tid, node in list(self._track_node.items()):
            if node == idx1:
                self._track_node[tid] = idx2
        self._retire(idx1)

    def _retire(self, idx):
        self._classes[idx] = -9999999
        self._means[idx] = np.nan
        self._covs[idx] = np.nan
        self._rels[idx, self._valid_mask] = 0
        self._rels[self._valid_mask, idx] = 0
        self._pcd[idx] = None
        self._valid_mask[idx] = False
        self._point_counts[idx] = 0
        self._observation_counts[idx] = 0
        self._frame_last_seen[idx] = 0
        
        for tid in [t for t, n in self._track_node.items() if n == idx]:
            del self._track_node[tid]

    def merge(self, update_idx, frame_num, global_merge=False):
        """
        TODO: summary

        Args:
            update_idx (_type_): _description_
            frame_num (_type_): _description_
            global_merge (bool, optional): _description_. Defaults to False.
        """
        self._sync_tables()
        self._merge_pass(update_idx, frame_num)

        # Periodic global sweep: reconsider every node, not just the ones this
        # frame touched. Catches pairs never co-examined, and pairs that only
        # drifted into range after several merges refined them.
        if global_merge:
            before = int(self._valid_mask.sum())
            self._merge_pass(
                np.nonzero(self._valid_mask)[0],
                frame_num,
                threshold=self.config.global_threshold,
                max_dist=self.config.global_max_dist,
                disjoint_only=self.config.global_disjoint)
            if DEBUG:
                print(f'[sg] GLOBAL REMERGE frame={frame_num} '
                      f'(threshold={self.config.global_threshold} max_dist={self.config.global_max_dist}): '
                      f'{before} -> {int(self._valid_mask.sum())} nodes')

        self._evict(frame_num)


    def _merge_pass(self, update_idx, frame_num, threshold=None, max_dist=None, disjoint_only=False):
        """
        TODO: summary

        Args:
            update_idx (_type_): _description_
            frame_num (_type_): _description_
            threshold (_type_, optional): _description_. Defaults to None.
            max_dist (_type_, optional): _description_. Defaults to None.
            disjoint_only (bool, optional): _description_. Defaults to False.
        """
        threshold = self.merge_threshold if threshold is None else threshold
        max_dist = self.config.max_merge_dist if max_dist is None else max_dist
        update_idx = np.asarray(update_idx).tolist()
        while update_idx:
            idx = update_idx.pop()
            if not self._valid_mask[idx]:        # absorbed earlier in this call
                continue

            while True:
                cand = (self._classes == self._classes[idx]) & self._valid_mask
                cand[idx] = False
                cand = np.nonzero(cand)[0]
                if cand.size == 0:
                    break

                with np.errstate(divide='ignore', invalid='ignore'):
                    dist = self._batched_hellinger_distance(
                        self._means[idx], self._covs[idx],
                        self._means[cand], self._covs[cand])
                dist = np.where(np.isfinite(dist), dist, np.inf)

                sep = np.linalg.norm(self._means[cand] - self._means[idx], axis=1)
                gated = np.where(sep <= max_dist, dist, np.inf)

                if disjoint_only and self._frame_last_seen[idx] == frame_num:
                    # One physical object yields at most ONE observation per frame.
                    # If both nodes were observed on this frame they are provably
                    # distinct objects, however close they sit.
                    gated = np.where(self._frame_last_seen[cand] == frame_num, np.inf, gated)


                best = int(np.argmin(gated))
                if DEBUG:
                    print(f'[sg] idx={idx} class={self._classes[idx]} '
                          f'cand={cand.size} obs={self._observation_counts[idx]} '
                          f'best={dist[best]:.3f} sep={sep[best] * 100:.1f}cm '
                          f'-> {"MERGE" if gated[best] < threshold else "no merge"}')
                if gated[best] >= threshold:
                    break

                target = cand[best]
                self._merge_gaussians(idx, target)
                idx = target                     # absorb from the survivor

    def _evict(self, frame_num):
        """Retire nodes never confirmed by a later observation."""
        if self.config.evict_age <= 0:
            return
        age = frame_num - self._frame_last_seen
        stale = self._valid_mask & (age > self.config.evict_age) & (self._observation_counts < self.config.evict_min_obs)
        for idx in np.nonzero(stale)[0]:
            if DEBUG:
                print(f'[sg] EVICT idx={idx} class={self._classes[idx]} '
                      f'obs={self._observation_counts[idx]} age={age[idx]}')
            self._retire(idx)