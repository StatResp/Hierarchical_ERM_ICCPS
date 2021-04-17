
from Environment.enums import RespStatus
import copy

class ResponderDynamics:

    def __init__(self,
                 travel_model):
        '''
        Represents the environmental dynamics of individual responder agents
        :param travel_model: Travel model which governs the time taken for agents to move from cell to cell
        '''
        self.travel_model = travel_model


    def update_responder(self, resp_id, _new_time, full_state):
        '''
        In this function, a responder's state will be updated to the given time. The responder can pass through
        multiple status changes during this update
        :param resp_obj:
        :param resp_id:
        :param _new_time:
        :param full_state:
        :return:
        '''

        curr_responder_time = copy.copy(full_state.time)

        # update the responder until it catches up with new time
        while curr_responder_time < _new_time:

            resp_status = full_state.responders[resp_id].status

            new_resp_time = None

            # handle status cases
            if resp_status == RespStatus.WAITING:
                new_resp_time, update_status = self.waiting_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)
            elif resp_status == RespStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)

            elif resp_status == RespStatus.RESPONDING:
                new_resp_time, update_status = self.response_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)

            elif resp_status == RespStatus.SERVICING:
                new_resp_time, update_status = self.service_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)


            curr_responder_time = new_resp_time

        if curr_responder_time == _new_time:
            resp_status = full_state.responders[resp_id].status

            new_resp_time = None

            # handle status cases
            if resp_status == RespStatus.WAITING:
                new_resp_time, update_status = self.waiting_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)
            elif resp_status == RespStatus.IN_TRANSIT:
                new_resp_time, update_status = self.transit_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)

            elif resp_status == RespStatus.RESPONDING:
                new_resp_time, update_status = self.response_update(resp_id,
                                                                    curr_responder_time,
                                                                    _new_time,
                                                                    full_state)

            elif resp_status == RespStatus.SERVICING:
                new_resp_time, update_status = self.service_update(resp_id,
                                                                   curr_responder_time,
                                                                   _new_time,
                                                                   full_state)

            curr_responder_time = new_resp_time


    def waiting_update(self, resp_id, curr_responder_time, _new_time, full_state):
        '''
        In this case, there is nothing to update for the responder's state, since it is simply waiting.
        :param resp_obj:
        :param resp_id:
        :param _new_time:
        :return:
        '''

        # assert
        return _new_time, True

    def transit_update(self, resp_id, curr_responder_time, _new_time, full_state):
        '''
        In this case, we update the position of the responder
        :param resp_obj:
        :param resp_id:
        :param v:
        :param _new_time:
        :return: resp_current_time
        '''

        if _new_time >= full_state.responders[resp_id].t_state_change:
            # responder has arrived
            # assume will now be waiting

            time_of_arrival = full_state.responders[resp_id].t_state_change

            full_state.responders[resp_id].cell_loc = full_state.responders[resp_id].dest
            full_state.responders[resp_id].dest = None
            full_state.responders[resp_id].status = RespStatus.WAITING
            full_state.responders[resp_id].t_state_change = None

            return time_of_arrival, True

        else:
            # responder has not arrived. Must interpolate
            journey_fraction = (_new_time - curr_responder_time) / (full_state.responders[resp_id].t_state_change - curr_responder_time)

            # TODO | should use beginning cell loc
            full_state.responders[resp_id].cell_loc = self.interpolate_distance(full_state.responders[resp_id].cell_loc, full_state.responders[resp_id].dest, journey_fraction)

            return _new_time, False

    def response_update(self, resp_id, curr_responder_time, _new_time, full_state):
        '''
        This will be very similar to the transit update. Can just use it?
        :param resp_obj:
        :param resp_id:
        :param state_curr_time:
        :param _new_time:
        :return:
        '''

        _resp_curr_time, resp_arrived = self.transit_update(resp_id, curr_responder_time, _new_time, full_state)

        if resp_arrived:
            # will now switch to servicing
            full_state.responders[resp_id].status = RespStatus.SERVICING
            full_state.responders[resp_id].t_state_change = full_state.responders[resp_id].incident.clearance_time + _resp_curr_time

        return _resp_curr_time, resp_arrived

    def service_update(self, resp_id, curr_responder_time, _new_time, full_state):
        '''
        Service status update. If still servicing, do nothing. Otherwise, send back to depot and now available
        :param resp_obj:
        :param resp_id:
        :param state_curr_time:
        :param _new_time:
        :return:
        '''

        if _new_time >= full_state.responders[resp_id].t_state_change:
            # completed servicing incident.
            time_of_completion = full_state.responders[resp_id].t_state_change

            full_state.responders[resp_id].dest = full_state.depots[full_state.responders[resp_id].assigned_depot_id].cell_loc
            full_state.responders[resp_id].status = RespStatus.IN_TRANSIT

            travel_time = self.travel_model.get_travel_time(full_state.responders[resp_id].cell_loc, full_state.responders[resp_id].dest, full_state.time)
            full_state.responders[resp_id].t_state_change = time_of_completion + travel_time

            full_state.responders[resp_id].available = True
            full_state.responders[resp_id].incident = None

            return time_of_completion, True

        else:
            # still servicing
            return _new_time, False

    def responder_assign_to_incident(self, resp_id, incident, full_state):
        '''
        update the given responder to respond to the given incident (obj).
        Return the response time
        What needs to happen:
        - determine t_state_change (and return response time)
        :param resp_obj:
        :param resp_id:
        :param incident:
        :param full_state:
        :return:
        '''

        # make sure that it isn't sending an unavailable responder
        assert full_state.responders[resp_id].available

        full_state.responders[resp_id].incident = incident
        full_state.responders[resp_id].status = RespStatus.RESPONDING
        full_state.responders[resp_id].available = False
        full_state.responders[resp_id].dest = incident.cell_loc

        travel_time = self.travel_model.get_travel_time(full_state.responders[resp_id].cell_loc, full_state.responders[resp_id].dest, full_state.time)
        full_state.responders[resp_id].t_state_change = full_state.time + travel_time
        full_state.responders[resp_id].available_time = full_state.time + travel_time + incident.clearance_time

        distance_traveled = self.travel_model.get_distance_cell_ids(full_state.responders[resp_id].cell_loc, full_state.responders[resp_id].dest)
        full_state.responders[resp_id].total_distance_moved += distance_traveled
        full_state.responders[resp_id].total_incident_distance += distance_traveled

        resp_time = full_state.responders[resp_id].t_state_change - incident.time

        return resp_time

    def assign_responder_to_depot_and_move_there(self, full_state, resp_id, depot_id):
        '''
        assign a responder to a different depot
        :param resp_obj:
        :param full_state:
        :param resp_id:
        :param depot_id:
        :return:
        '''

        self.assign_responder_to_depot(full_state, resp_id, depot_id)
        travel_time = self.move_responder_to_assigned_depot(resp_id, full_state)

        # print(full_state.responders[resp_id].__dict__)
        return travel_time

    def assign_responder_to_depot(self, full_state, resp_id, depot_id):
        # depot = full_state.depots[depot_id]
        # print(full_state.responders[resp_id].__dict__)
        full_state.responders[resp_id].assigned_depot_id = depot_id
        # print(full_state.responders[resp_id].__dict__)

    def move_responder_to_assigned_depot(self, resp_id, full_state):
        # print(full_state.responders[resp_id].__dict__)

        depot = full_state.depots[full_state.responders[resp_id].assigned_depot_id]

        full_state.responders[resp_id].status = RespStatus.IN_TRANSIT
        full_state.responders[resp_id].dest = depot.cell_loc

        travel_time = self.travel_model.get_travel_time(full_state.responders[resp_id].cell_loc,
                                                        full_state.responders[resp_id].dest,
                                                        full_state.time)
        full_state.responders[resp_id].t_state_change = full_state.time + travel_time

        distance_traveled = self.travel_model.get_distance_cell_ids(full_state.responders[resp_id].cell_loc,
                                                                    full_state.responders[resp_id].dest)
        full_state.responders[resp_id].total_distance_moved += distance_traveled
        full_state.responders[resp_id].total_depot_movement_distance += distance_traveled

        return travel_time

    def assign_responder_to_region(self, full_state, resp_id, region_id):
        full_state.responders[resp_id].region_assignment = region_id

    def interpolate_distance(self, curr_cell, dest_cell, journey_fraction):
        # new_x = ((1.0 - journey_fraction) * float(curr_cell[0])) + (journey_fraction * float(dest_cell[0]))
        # new_y = ((1.0 - journey_fraction) * float(curr_cell[1])) + (journey_fraction * float(dest_cell[1]))
        #
        # return (int(new_x), int(new_y))
        return self.travel_model.interpolate_distance(curr_cell, dest_cell, journey_fraction)

