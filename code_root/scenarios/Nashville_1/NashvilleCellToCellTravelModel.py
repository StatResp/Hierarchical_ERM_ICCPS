from math import hypot


class NashvilleCellToCellTravelModel:

    def __init__(self,
                 cell_to_xy_dict,
                 travel_rate_cells_per_second):  # 60.0 / 3600.0 -> 60 mph for 1 mile cells
        self.travel_rate_cells_per_second = travel_rate_cells_per_second
        self.cell_to_xy_dict = cell_to_xy_dict

    def get_distance_cell_coords(self, curr_cell_coords, dest_cell_coords):

        if curr_cell_coords is None:
            raise ValueError('cell1 is none')
        elif dest_cell_coords is None:
            raise ValueError('cell2 is none')
        return hypot(curr_cell_coords[0] - dest_cell_coords[0], curr_cell_coords[1] - dest_cell_coords[1])


    def interpolate_distance(self, curr_cell, dest_cell, journey_fraction):

        curr_cell_coords = self.cell_to_xy_dict[curr_cell]
        dest_cell_coords = self.cell_to_xy_dict[dest_cell]

        new_x = int(((1.0 - journey_fraction) * float(curr_cell_coords[0])) + (journey_fraction * float(dest_cell_coords[0])))
        new_y = int(((1.0 - journey_fraction) * float(curr_cell_coords[1])) + (journey_fraction * float(dest_cell_coords[1])))

        # want closest cell to interpolated one

        # TODO | @AM - only calculate below if cell is invalid (save computation time)
        nearest_cell = min(self.cell_to_xy_dict.items(), key=lambda _: self.get_distance_cell_coords(_[1], (new_x, new_y)))[0]

        return nearest_cell


    def get_travel_time(self, curr_cell, dest_cell, curr_time):

        curr_cell_coords = self.cell_to_xy_dict[curr_cell]
        dest_cell_coords = self.cell_to_xy_dict[dest_cell]

        distance = self.get_distance_cell_coords(curr_cell_coords, dest_cell_coords)

        return distance / self.travel_rate_cells_per_second

    def get_distance_cell_ids(self, curr_cell, dest_cell):

        curr_cell_coords = self.cell_to_xy_dict[curr_cell]
        dest_cell_coords = self.cell_to_xy_dict[dest_cell]

        if curr_cell_coords is None:
            raise ValueError('cell1 is none')
        elif dest_cell_coords is None:
            raise ValueError('cell2 is none')
        return hypot(curr_cell_coords[0] - dest_cell_coords[0], curr_cell_coords[1] - dest_cell_coords[1])

