
import itertools


class Incident:

    __last_id = 0

    def __init__(self,
                 cell_loc,
                 time,
                 clearance_time,
                 my_id=None):

        self.cell_loc = cell_loc
        self.time = time
        self.clearance_time = clearance_time

        if my_id is None:
            self.my_id = Incident.__last_id
            Incident.__last_id += 1
        else:
            self.my_id = my_id

