

class BetweenRegionTravelModel:

    def __init__(self, region_centers, cell_travel_model):
        self.cell_travel_model = cell_travel_model
        self.region_centers = region_centers

    def get_travel_time(self, region_1, region_2, curr_time):
        # ignore time for now

        return self.cell_travel_model.get_travel_time(self.region_centers[region_1],
                                                      self.region_centers[region_2],
                                                      curr_time)

