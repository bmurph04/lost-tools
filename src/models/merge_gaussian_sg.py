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

DEBUG = os.environ.get('LOST_TOOLS_SG_DEBUG', '') == '1'
MAX_MERGE_DIST = float(os.environ.get('LOST_TOOLS_SG_MAX_DIST', '0.30'))
PCD_CAP = int(os.environ.get('LOST_TOOLS_SG_PCD_CAP', '20000'))
EVICT_AGE = int(os.environ.get('LOST_TOOLS_SG_EVICT_AGE', '150'))
EVICT_MIN_OBS = int(os.environ.get('LOST_TOOLS_SG_EVICT_MIN_OBS', '3'))


class GaussianSGMerge(GaussianSG):

    def __init__(self, num_rel_class, merge_threshold):
        super().__init__(num_rel_class, merge_threshold)
        self._frame = 0
        self._weights = np.zeros(self._max_size, dtype=np.int64)
        self._obs = np.zeros(self._max_size, dtype=np.int64)
        self._seen = np.zeros(self._max_size, dtype=np.int64)

    # -- bookkeeping --------------------------------------------------------

    def _sync_tables(self):
        """Grow the side tables when _expand_if_needed grows the slot arrays."""
        n = self._valid_mask.shape[0]
        if self._weights.shape[0] == n:
            return
        for name in ('_weights', '_obs', '_seen'):
            old = getattr(self, name)
            grown = np.zeros(n, dtype=np.int64)
            m = min(n, old.shape[0])
            grown[:m] = old[:m]
            setattr(self, name, grown)

    def _weight_of(self, idx):
        if self._weights[idx] > 0:
            return int(self._weights[idx])
        pcd = self._pcd[idx]
        return 0 if pcd is None else len(pcd)

    # -- overrides ----------------------------------------------------------

    def add(self, *args, **kwargs):
        idxs = super().add(*args, **kwargs)
        self._sync_tables()
        self._obs[idxs] = 1
        self._seen[idxs] = self._frame
        self._weights[idxs] = 0        # 0 => fall back to len(pcd)
        return idxs

    def _merge_gaussians(self, idx1, idx2):
        """Merge idx1 into idx2, keeping a running-average covariance."""
        mean1, cov1 = self._means[idx1], self._covs[idx1]
        mean2, cov2 = self._means[idx2], self._covs[idx2]

        n1 = self._weight_of(idx1)
        n2 = self._weight_of(idx2)
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
        if PCD_CAP > 0 and len(merged) > PCD_CAP:
            merged = merged[np.random.choice(len(merged), PCD_CAP, replace=False)]
        self._pcd[idx2] = merged

        self._weights[idx2] = total          # true weight, ignores the cap
        self._obs[idx2] = max(1, self._obs[idx2]) + max(1, self._obs[idx1])
        self._seen[idx2] = self._frame
        self._retire(idx1)

    def _retire(self, idx):
        self._classes[idx] = -9999999
        self._means[idx] = np.nan
        self._covs[idx] = np.nan
        self._rels[idx, self._valid_mask] = 0
        self._rels[self._valid_mask, idx] = 0
        self._pcd[idx] = None
        self._valid_mask[idx] = False
        self._weights[idx] = 0
        self._obs[idx] = 0
        self._seen[idx] = 0

    def merge(self, update_idx):
        self._sync_tables()
        self._frame += 1

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
                gated = np.where(sep <= MAX_MERGE_DIST, dist, np.inf)

                best = int(np.argmin(gated))
                if DEBUG:
                    print(f'[sg] idx={idx} class={self._classes[idx]} '
                          f'cand={cand.size} obs={self._obs[idx]} '
                          f'best={dist[best]:.3f} sep={sep[best] * 100:.1f}cm '
                          f'-> {"MERGE" if gated[best] < self.merge_threshold else "no merge"}')
                if gated[best] >= self.merge_threshold:
                    break

                target = cand[best]
                self._merge_gaussians(idx, target)
                idx = target                     # absorb from the survivor

        self._evict()

    def _evict(self):
        """Retire nodes never confirmed by a later observation."""
        if EVICT_AGE <= 0:
            return
        age = self._frame - self._seen
        stale = self._valid_mask & (age > EVICT_AGE) & (self._obs < EVICT_MIN_OBS)
        for idx in np.nonzero(stale)[0]:
            if DEBUG:
                print(f'[sg] EVICT idx={idx} class={self._classes[idx]} '
                      f'obs={self._obs[idx]} age={age[idx]}')
            self._retire(idx)