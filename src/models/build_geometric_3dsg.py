import numpy as np

class Geometric3DSGBuilder:
    """
    FIXME
    """
    def __init__(self, near_distance=0.2, on_horizontal_tolerance=0.0, on_vertical_tolerance=0.0):
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
        near_pred_id = pred_name_to_id['near']

        num_objects = len(means)
        scene_graph = []

        for i in range(num_objects):
            for j in range(i+1, num_objects):

                # Get the means and extents
                a_mean, b_mean = means[i, :], means[j, :]
                a_extents, b_extents = extents[i, :], extents[j, :]

                # Compute the distance between the means
                distance = np.linalg.norm(a_mean - b_mean)

                # Get the horizontal (x and z) coordinates and compute the distance 
                horizontal_distance = float(np.linalg.norm(a_mean[[0, 2]] - b_mean[[0, 2]]))
                # Horizontal ?
                horizontal_contact_limit = float(np.linalg.norm(a_extents[[0, 2]])/2.0 + np.linalg.norm(b_extents[[0, 2]])/2.0)
                
                # Get the vertical (y) difference
                vertical_distance = abs(a_mean[1] - b_mean[1])
                # Vertical ?
                vertical_contact_limit = a_extents[1]/2.0 + b_extents[1]/2.0
                # Add a bit of slack based on the heights of objects a and b
                vertical_contact_limit_adj = vertical_contact_limit + a_extents[1]/10.0 + b_extents[1]/10.0

                # print(f'\nvertical distance between {i} and {j} is {vertical_distance} and the contact limit is {vertical_contact_limit} and adjusted its {vertical_contact_limit_adj}\n')

                # Objects are aligned if distance is within range of contact limit
                horizontally_aligned = horizontal_distance <= horizontal_contact_limit + self.on_horizontal_tolerance
                vertically_aligned = vertical_distance <= vertical_contact_limit + self.on_vertical_tolerance

                # print('\n')
                # print(f'object a: {i}, object b: {j}')
                # print(f'obj a mean: {a_mean}, obj b mean: {b_mean}')
                # print(f'distance: {distance}, horizontal distance: {horizontal_distance}, vertical distance: {vertical_distance}')
                # print(f'vertically aligned: {vertical_contact_limit}, horizontally aligned: {horizontally_aligned}, near: {distance <= self.near_distance}')
                # print('\n')

                # On relation if horizontally and vertically aligned
                if horizontally_aligned and vertically_aligned:
                    on_relation = (i, on_pred_id, j) if a_mean[1] > b_mean[1] else (j, on_pred_id, i)
                    scene_graph.append(on_relation)
                    # Continue to suppress near relation
                    # print(f'Object {i} is on {j}\n')
                    continue

                if distance <= self.near_distance:
                    near_relations = [(i, near_pred_id, j), (j, near_pred_id, i)]
                    scene_graph.extend(near_relations)

        return scene_graph