
from Environment.enums import EventType
from Environment.ResponderDynamics import ResponderDynamics

class EnvironmentModel:

    '''
    Update state based on passage of time. This includes...
    - responder updates with time
    - adding incidents
    - incident response and reward
    - depot use

    *event agnostic*
    '''

    def __init__(self,
                 travel_model):

        self.resp_dynamics = ResponderDynamics(travel_model=travel_model)
        self.travel_model = travel_model


    def update(self, state, new_time):
        '''
        Updates the state to the given time. This is mostly updating the responders
        :param state:
        :param new_time:
        :return:
        '''

        assert new_time >= state.time

        # update the state of each responder
        for resp_id, resp_obj in state.responders.items():
            self.resp_dynamics.update_responder(resp_id=resp_id,
                                                _new_time=new_time,
                                                full_state=state)

        state.time = new_time

    def add_incident(self, state, incident):
        '''
        Adds pending incident to the given state. Call to simulate incident occurance
        :param state:
        :param incident:
        :return:
        '''

        state.active_incidents.append(incident)


    def get_pending_incidents(self, state):
        return state.active_incidents


    def respond_to_incident(self, state, resp_id, incident_id):
        '''
        updates the state for the given responder to respond to the given incident. Returns the response time.
        Removes incident from active incident list
        :param state:
        :param resp_id:
        :param incident:
        :return:
        '''

        incident = next(_ for _ in state.active_incidents if _.my_id == incident_id)

        response_time = self.resp_dynamics.responder_assign_to_incident(resp_id=resp_id,
                                                                        incident=incident,
                                                                        full_state=state)

        state.active_incidents.remove(incident)

        return response_time

    def assign_responder_to_depot_and_move_there(self, state, resp_id, depot_id):
        '''
        assign the given responder to the given depot. Then direct the responder to travel to the assigned depot.
        :param state:
        :param resp_id:
        :param depot_id:
        :return:
        '''
        travel_time =  self.resp_dynamics.assign_responder_to_depot_and_move_there(full_state=state,
                                                                                   resp_id = resp_id,
                                                                                   depot_id= depot_id)

        return travel_time


    def assign_responder_to_depot(self, state, resp_id, depot_id):
        '''
        Assign the given responder to the given depot WITHOUT assigning the responder to move to the depot.
        Useful if the responder is currently busy (for example, currently servicing an incident)
        :param state:
        :param resp_id:
        :param depot_id:
        :return:
        '''
        self.resp_dynamics.assign_responder_to_depot(full_state=state,
                                                     resp_id = resp_id,
                                                     depot_id= depot_id)

    def move_resp_to_assigned_depot(self, state, resp_id):
        '''
        direct the given responder to move to its assigned depot
        :param state:
        :param resp_id:
        :param depot_id:
        :return:
        '''
        return self.resp_dynamics.move_responder_to_assigned_depot(resp_id=resp_id,
                                                     full_state=state)


    def assign_resp_to_region(self, state, resp_id, region_id):
        '''
        Assign the given responder to the given region.
        :param state:
        :param resp_id:
        :param region_id:
        :return:
        '''
        self.resp_dynamics.assign_responder_to_region(full_state=state,
                                                      resp_id=resp_id,
                                                      region_id=region_id)


    def get_depot_ids_in_region(self, state, region_id):
        '''
        return a list of all the depot ids 'within' the given region
        :param state:
        :param region_id:
        :return:
        '''
        region_cells = state.regions[region_id]
        return [_[0] for _ in state.depots.items() if _[1].cell_loc in region_cells]


    def get_closest_depot_to_cell(self, cell_id, state):
        '''
        return the depot that is closest to the given cell
        :param cell_id:
        :param state:
        :return:
        '''

        closest_dist = float('inf')
        closest_depot = None

        for depot_id, depot_obj in state.depots.items():
            dist = self.travel_model.get_distance_cell_ids(depot_obj.cell_loc, cell_id)
            if dist < closest_dist:
                closest_dist = dist
                closest_depot = depot_id

        return closest_depot




