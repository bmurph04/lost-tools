import numpy as np

class Geometric3DSGBuilder:
    """
    FIXME
    """
    def __init__(self, near_distance=0.5, on_horizontal_tolerance=0.00, on_vertical_tolerance=0.05):
        """
        FIXME
        Args:
        """
        self.near_distance = near_distance
        self.on_horizontal_tolerance = on_horizontal_tolerance
        self.on_vertical_tolerance = on_vertical_tolerance         

    def build_3d_scene_graph(self, means, extents, pred_name_to_id):
        """
        
        """
        # Using on and near predicates
        on_pred_id = pred_name_to_id['on']
        best_on_relations = {}
        near_pred_id = pred_name_to_id['near']
        near_relations = []

        num_objects = len(means)

        for i in range(num_objects):
            for j in range(i+1, num_objects):

                # Get the means and extents
                a_mean, b_mean = means[i, :], means[j, :]
                a_extents, b_extents = extents[i, :], extents[j, :]

                # Compute the distance between the means
                distance = np.linalg.norm(a_mean - b_mean)

                # Get the horizontal (x and z) coordinates and compute the distance 
                horizontal_distance = [abs(a_mean[0] - b_mean[0]), abs(a_mean[2] - b_mean[2])]
                horizontal_contact_limit = [(a_extents[0]/2.0 + b_extents[0]/2.0), (a_extents[2]/2.0 + b_extents[2]/2.0)]
                
                # Get the vertical (y) difference
                if a_mean[1] >= b_mean[1]:
                    above_idx = i
                    below_idx = j
                    above_surface = a_mean[1] - a_extents[1]/2.0
                    below_surface = b_mean[1] + b_extents[1]/2.0
                else:
                    above_idx = j
                    below_idx = i
                    above_surface = b_mean[1] - b_extents[1]/2.0
                    below_surface = a_mean[1] + a_extents[1]/2.0 
                    
                vertical_gap = abs(above_surface - below_surface)

                # Objects are aligned if distance is within range of contact limit
                horizontally_aligned = horizontal_distance[0] <= (horizontal_contact_limit[0] + self.on_horizontal_tolerance) and \
                                        horizontal_distance[1] <= (horizontal_contact_limit[1] + self.on_horizontal_tolerance)
                vertically_aligned = vertical_gap <= self.on_vertical_tolerance

                # print(f'\nvertical distance between {i} and {j} is {vertical_distance} and the contact limit is {vertical_contact_limit}, adjusted its {vertical_contact_limit + self.on_vertical_tolerance}')
                # print(f'horizontal distance between {i} and {j} is {horizontal_distance} and the contact limit is {horizontal_contact_limit}\n')
                # print(f'horizontally aligned: {horizontally_aligned} and vertically_aligned: {vertically_aligned}\n')

                # print('\n')
                # print(f'object a: {i}, object b: {j}')
                # print(f'obj a mean: {a_mean}, obj b mean: {b_mean}')
                # print(f'distance: {distance}, horizontal distance: {horizontal_distance}, vertical distance: {vertical_distance}')
                # print(f'vertically aligned: {vertical_contact_limit}, horizontally aligned: {horizontally_aligned}, near: {distance <= self.near_distance}')
                # print('\n')

                # On relation if horizontally and vertically aligned
                if horizontally_aligned and vertically_aligned:

                    if above_idx in best_on_relations:
                        existing_below_idx, existing_vertical_gap = best_on_relations[above_idx]

                        if vertical_gap < existing_vertical_gap:
                            best_on_relations[above_idx] = (below_idx, vertical_gap)

                    else:
                        best_on_relations[above_idx] = (below_idx, vertical_gap)

                    # Continue to suppress near relation
                    continue

                # Near relation if distance is close enough
                if distance <= self.near_distance:
                    near_relations.extend([(i, near_pred_id, j), (j, near_pred_id, i)])

        scene_graph = []

        # Track active 'ON' pairs to suppress reciprocal 'NEAR' edges
        on_pairs = set()
        for above_idx, (below_idx, _) in best_on_relations.items():
            scene_graph.append((above_idx, on_pred_id, below_idx))
            on_pairs.add((above_idx, below_idx))
            on_pairs.add((below_idx, above_idx))

        # Add 'NEAR' relations (skipping pairs that already have an 'ON' relation)
        for sub_id, pred_id, obj_id in near_relations:
            if (sub_id, obj_id) not in on_pairs:
                scene_graph.append((sub_id, pred_id, obj_id))
                
        return scene_graph