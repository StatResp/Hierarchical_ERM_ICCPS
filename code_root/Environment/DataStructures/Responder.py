

class Responder:

    def __init__(self,
                 my_id,
                 cell_loc,
                 status,
                 available,
                 dest,
                 t_state_change,
                 # delta_t_state_change,
                 region_assignment,
                 assigned_depot_id,
                 incident,
                 available_time,
                 total_distance_moved,
                 total_incident_distance,
                 total_depot_movement_distance):

        self.total_incident_distance = total_incident_distance
        self.total_depot_movement_distance = total_depot_movement_distance
        self.total_distance_moved = total_distance_moved  # in whatever units
        self.available_time = available_time
        self.incident = incident
        self.my_id = my_id
        self.status = status
        self.available = available
        self.dest = dest
        self.t_state_change = t_state_change
        # self.delta_t_state_change = delta_t_state_change
        self.region_assignment = region_assignment
        self.assigned_depot_id = assigned_depot_id
        self.cell_loc = cell_loc
