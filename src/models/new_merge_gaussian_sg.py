import numpy as np

class MergedGaussianSG:
    def __init__(self):
        
    
    def update(self, observations, triplets, frame_num, global_merge=False):
        """
        Fold one frame of observations into the graph and consolidate.

        Args:
            observations (_type_): _description_
            triplets (_type_): _description_
            frame_num (_type_): _description_
            global_merge (bool, optional): _description_. Defaults to False.
        """
        if len(observations) == 0:
            return
        
        # Assign slots to the observations
        observation_to_slot = self._resolve(observations, frame_num)
        # Add relations
        self._add_relations(triplets, observation_to_slot)
        
        touched = np.unique(np.asarray(observation_to_slot, dtype=np.int64))
        self._merge_pass(touched, frame_num)
        
        if global_merge:
            self._merge_pass(
                np.flatnonzero(self._valid_mask), frame_num,
                threshold=self.config.global_threshold,
                max_dist=self.config.global_max_dist,
                disjoint_only=self.config.global_disjoing
            )
            
        self._evict(frame_num)
    
    def _resolve(self, observations, frame_num):
        """
        Map each observation to the node it belongs in, opening nodes as needed.
        
        A track that already owns a node folds straight into it with no distance test,
        because the tracker asserting continuity is stronger evidence than 3D proximity
        and is immune to depth noise.

        Args:
            observations (_type_): _description_
            frame_num (_type_): _description_
        """
        observation_to_slot = []
        
        # Iterate through each object id
        for i, object_id in enumerate(observations.object_ids):
            # Get the node slot the object id is assigned to, if it is assigned
            slot = self._object_id_to_node_slot(object_id)
            
            # If the slot is assigned and valid, the object is currently a part of the scene graph
            if slot is not None and self._valid_mask[slot]:
                # The object jumped if the current observation of the object is further than 
                # broken_track_dist away from the stored location of the object
                object_jumped = (np.linalg.norm(observations.means[i] - self._means[slot]) > self.config.broken_track_dist)

                # Check if the object jumped or if the class_id of the observation is different than the class_id of the slot