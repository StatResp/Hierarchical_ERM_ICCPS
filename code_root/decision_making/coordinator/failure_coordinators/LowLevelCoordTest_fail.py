import time

import numpy as np
from Environment.Simulator import Simulator
from Environment.enums import EventType
from Environment.DataStructures.Event import Event



import random
import copy

class LowLevelCoord_Fail:

    def __init__(self,
                 environment_model,
                 travel_model,
                 dispatch_policy,
                 low_level_policy,
                 min_time_between_allocation):
        # self.dispatch_decision_maker = dispatch_decision_maker
        self.min_time_between_allocation = min_time_between_allocation
        self.low_level_policy = low_level_policy
        self.dispatch_policy = dispatch_policy
        self.travel_model = travel_model
        self.environment_model = environment_model
        self.metrics = dict()
        self.metrics['resp_times'] = dict()
        self.metrics['region_violations'] = dict()
        self.metrics['computation_times'] = dict()


    def event_processing_callback_funct(self, state, curr_event, next_event):
        '''
        function that is called when each new event occurs in the underlying simulation.
        :param state:
        :param curr_event:
        :return:
        '''

        if curr_event.event_type == EventType.INCIDENT:
            '''
            If the event is an incident, do the following: 
            - add the incident to the incident queue
            - send a responder to the incident
            - record the response times
            - perform low level allocation 
            - 
            '''

            # TODO | can low level allocation be limited to just the regions affected by the event?

            start_compu_time = time.time()

            self.add_incident(state, curr_event)
            new_events = self.dispatch_to_active_incidents(state)

            allocation_action = self.low_level_policy.process(state)

            self.process_low_level_action(allocation_action, state)

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

            end_compu_time = time.time()
            self.metrics['computation_times'][curr_event] = (end_compu_time - start_compu_time)

            return new_events

        elif curr_event.event_type == EventType.RESPONDER_AVAILABLE:
            '''
            In this case, we know that a responder is available (or might be), so try dispatch
            '''
            new_events = self.dispatch_to_active_incidents(state)

            return new_events

        elif curr_event.event_type == EventType.ALLOCATION:
            '''
            Only need to allocate
            '''

            start_compu_time = time.time()

            new_events = []
            allocation_action = self.low_level_policy.process(state)

            self.process_low_level_action(allocation_action, state)

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

            end_compu_time = time.time()
            self.metrics['computation_times'][curr_event] = (end_compu_time - start_compu_time)

            return new_events

        elif curr_event.event_type == EventType.FAILURE:

            '''
            process the failure event, and then run the low level planner as normal
            '''

            start_compu_time = time.time()
            self.process_failure_event(state, event=curr_event)

            new_events = []
            allocation_action = self.low_level_policy.process(state)

            self.process_low_level_action(allocation_action, state)

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

            end_compu_time = time.time()
            self.metrics['computation_times'][curr_event] = (end_compu_time - start_compu_time)

            return new_events



    def process_low_level_action(self, action, state):
        for resp_id, dep_id in action.items():
            if state.responders[resp_id].available:
                # if available, assign and move
                self.environment_model.assign_responder_to_depot_and_move_there(state, resp_id, dep_id)
            else:
                # only assign
                self.environment_model.assign_responder_to_depot(state=state,
                                                             resp_id=resp_id,
                                                             depot_id=dep_id)

    def add_incident(self, state, incident_event):
        incident = incident_event.type_specific_information['incident_obj']
        self.environment_model.add_incident(state, incident)

    def dispatch_to_active_incidents(self, state):
        resp_to_incident_tupples = self.dispatch_policy.get_responder_to_incident_assignments(state)

        for resp, incident in resp_to_incident_tupples:

            response_time = self.environment_model.respond_to_incident(state, resp.my_id, incident.my_id)

            self.metrics['resp_times'][incident.my_id] = response_time

            print('\tavg resp times to here: {}'.format(np.mean(list(self.metrics['resp_times'].values()))))

            # check if crossing region lines
            resp_region = resp.region_assignment
            incident_region = state.cells[incident.cell_loc]
            if resp_region != incident_region:
                print('\tcrossed region lines!')
                # self.metrics['region_violations'][incident.my_id] = (resp, incident)


        # TODO | How to account for responder saturation? Need to have events for responder availability
        if len(self.environment_model.get_pending_incidents(state)) > 0:
            # responders are saturated. Need event that corresponds to next responder state change
            responder_with_next_state_change = min(state.responders.values(),
                                              key=lambda _: _.t_state_change)

            return [Event(event_type=EventType.RESPONDER_AVAILABLE,
                         cell_loc=None,
                         time=responder_with_next_state_change.t_state_change,
                         type_specific_information=None)
            ]

        else:
            return []

    def process_failure_event(self, state, event):

        # current failure events are based on regions. Therefore delete a responder from given region

        region_to_remove_resp = event.type_specific_information['region']
        number_of_resp_to_fail =  event.type_specific_information['num_resp_to_fail']

        print('\t\tFailure event! Removing {} resp from region {}'.format(number_of_resp_to_fail, region_to_remove_resp))

        num_resp_total = len(state.responders)
        num_resp_in_region = len([_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp])
        print('\t\tnum resp before: {}, num in region before: {}'.format(num_resp_total, num_resp_in_region))

        # assume there is at least 'number_of_resp_to_fail' resp in the region
        for i in range(number_of_resp_to_fail):
            resp_keys_in_region = [_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp]

            # remove first responder
            del state.responders[resp_keys_in_region[0]]

        num_resp_total = len(state.responders)
        num_resp_in_region = len(
            [_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp])
        print('\t\tnum resp After: {}, num in region before: {}'.format(num_resp_total, num_resp_in_region))










