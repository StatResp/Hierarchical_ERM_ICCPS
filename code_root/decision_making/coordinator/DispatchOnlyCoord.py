from Environment.Simulator import Simulator
from Environment.enums import EventType
from Environment.DataStructures.Event import Event
import numpy as np

import random
import copy

class DispatchOnlyCoord:

    def __init__(self,
                 state,
                 environment_model,
                 travel_model,
                 dispatch_policy):
        # self.dispatch_decision_maker = dispatch_decision_maker
        self.dispatch_policy = dispatch_policy
        self.travel_model = travel_model
        self.environment_model = environment_model
        self.state = state
        self.metrics = dict()
        self.metrics['resp_times'] = dict()


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
            '''

            self.add_incident(state, curr_event)
            new_events = self.dispatch_to_active_incidents(state)

            return new_events

        elif curr_event.event_type == EventType.RESPONDER_AVAILABLE:
            '''
            In this case, we know that a responder is (or might be), so try dispatch
            '''
            print('curr_state_time: {}'.format(state.time))
            print('num_resp_available: {}'.format(len([_ for _ in state.responders.values() if _.available])))
            new_events = self.dispatch_to_active_incidents(state)
            print('num_resp_available: {}'.format(len([_ for _ in state.responders.values() if _.available])))
            print('\tresp_available')

            return new_events

    def add_incident(self, state, incident_event):
        incident = incident_event.type_specific_information['incident_obj']
        self.environment_model.add_incident(state, incident)

    def dispatch_to_active_incidents(self, state):

        # # TODO | testing just not trying dispatch if no resp available
        num_available_resp = len([_ for _ in state.responders.values() if _.available])

        if num_available_resp <= 0:
            # no resp available
            return []

        resp_to_incident_tupples = self.dispatch_policy.get_responder_to_incident_assignments(state)

        for resp, incident in resp_to_incident_tupples:
            response_time = self.environment_model.respond_to_incident(state, resp.my_id, incident.my_id)

            self.metrics['resp_times'][incident.my_id] = response_time

            print('\tavg resp times to here: {}'.format(np.mean(list(self.metrics['resp_times'].values()))))

            # check if crossing region lines
            resp_region = resp.region_assignment
            if incident.cell_loc in state.cells:
                incident_region = state.cells[incident.cell_loc]
                if resp_region != incident_region:
                    print('\tcrossed region lines!')
            else:
                print('***incident occured out of region***')
                _y = incident.cell_loc % 30
                _x = incident.cell_loc - (_y * 30)
                print('***cell: {}, x: {}, y{}'.format(incident.cell_loc, _x, _y))
                # self.metrics['region_violations'][incident.my_id] = (resp, incident)

        # TODO | How to account for responder saturation? Need to have events for responder availability
        # if len(self.environment_model.get_pending_incidents(state)) > 0:
        num_available_resp = len([_ for _ in state.responders.values() if _.available])
        if num_available_resp <= 0:
            # responders are saturated. Need event that corresponds to next responder state change
            responder_with_next_state_change = min(state.responders.values(),
                                              key=lambda _: _.available_time)

            return [Event(event_type=EventType.RESPONDER_AVAILABLE,
                         cell_loc=None,
                         time=responder_with_next_state_change.available_time,
                         type_specific_information=None)
            ]

        else:
            return []









