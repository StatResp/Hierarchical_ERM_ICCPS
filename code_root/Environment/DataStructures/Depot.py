

class Depot:

    def __init__(self,
                 cell_loc,
                 assigned_responders,
                 capacity,
                 my_id,
                 lat=None,
                 lon=None):
        self.lon = lon
        self.lat = lat
        self.capacity = capacity
        self.assigned_responders = assigned_responders
        self.cell_loc = cell_loc
        self.my_id = my_id
