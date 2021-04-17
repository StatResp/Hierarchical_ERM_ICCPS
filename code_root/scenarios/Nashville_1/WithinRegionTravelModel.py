

class NashvilleWithinRegionTravelModel:

    def __init__(self,
                 mean_dist_for_region_dict,
                 travel_cells_per_second):
        self.travel_cells_per_second = travel_cells_per_second
        self.mean_dist_for_region_dict = mean_dist_for_region_dict


    # TODO | eventually make it include the num_resp
    def  get_expected_travel_time(self, region_id, curr_time, num_responders):
        return self.mean_dist_for_region_dict[region_id] / self.travel_cells_per_second
