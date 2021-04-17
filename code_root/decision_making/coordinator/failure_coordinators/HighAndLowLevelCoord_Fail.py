import copy

from Environment.enums import EventType
from Environment.DataStructures.Event import Event

# import geopandas as gpd

import numpy as np



class HighAndLowLevelCoord_Fail:

    def __init__(self,
                 environment_model,
                 travel_model,
                 dispatch_policy,
                 low_level_policy,
                 min_time_between_allocation,
                 high_level_policy):
        # self.dispatch_decision_maker = dispatch_decision_maker
        self.high_level_policy = high_level_policy
        self.min_time_between_allocation = min_time_between_allocation
        self.low_level_policy = low_level_policy
        self.dispatch_policy = dispatch_policy
        self.travel_model = travel_model
        self.environment_model = environment_model
        self.metrics = dict()
        self.metrics['resp_times'] = dict()





    def event_processing_callback_funct(self, state, curr_event, next_event):
        '''
            Function that is called when each new event occurs in the underlying simulation.
        Change between this and the pure low level coord is that it now needs to check and see if
        high level re-allocations are needed, and execute them if necessary.

        When to check for re-allocations?
            1. After an incident occurs
            2. During allocation events

            Current high level policy is not interested in current state of responders
        (i.e. if responders are available, etc.), only in the 'steady state' of each region
        as represented as a queuing system. Still, should check each time

        :param state:
        :param curr_event:
        :param next_event:
        :return:
        '''

        if curr_event.event_type == EventType.INCIDENT:
            '''
            If the event is an incident, do the following: 
            - add the incident to the incident queue
            - send a responder to the incident
            - record the response times
            - check if high level planning is triggered, perform if true
            - perform low level allocation
            - see if new allocation event is needed to be added to the queue 
            '''

            # TODO | can low-level allocation be limited to just the regions affected by the event?

            self.add_incident(state, curr_event)
            new_events = self.dispatch_to_active_incidents(state)

            # high level check / allocation
            if self.high_level_policy.check_if_triggered(state):
                high_level_allocation = self.high_level_policy.region_responder_allocation(state, curr_event)
                self.process_high_level_action(high_level_allocation, state)

            # low level allocation (always occurs)
            allocation_action = self.low_level_policy.process(state)
            self.process_low_level_action(allocation_action, state)

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

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
            new_events = []

            # high level check / allocation
            if self.high_level_policy.check_if_triggered(state):
                high_level_allocation = self.high_level_policy.region_responder_allocation(state, curr_event)
                self.process_high_level_action(high_level_allocation, state)

            allocation_action = self.low_level_policy.process(state)

            self.process_low_level_action(allocation_action, state)

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

            return new_events


        elif curr_event.event_type == EventType.FAILURE:

            '''
            process the failure event, and then run the low level planner as normal
            '''

            self.process_failure_event(state, event=curr_event)

            new_events = []

            # high level check / allocation
            if self.high_level_policy.check_if_triggered(state):
                high_level_allocation = self.high_level_policy.region_responder_allocation(state, curr_event)
                self.process_high_level_action(high_level_allocation, state)

            allocation_action = self.low_level_policy.process(state)

            self.process_low_level_action(allocation_action, state)

            # self.make_failure_state_before_log_file(state, 'failure_log_after.geojson')

            if next_event is not None:

                if next_event.time > curr_event.time + self.min_time_between_allocation:
                    new_events.append(Event(event_type=EventType.ALLOCATION,
                                            cell_loc=None,
                                            time=curr_event.time + self.min_time_between_allocation))

            return new_events


    def process_high_level_action(self, region_allocation_action, state):
        '''
        There are several things that need to occur to process a high level re-allocation to regions.
            1. for any resp that has not changed their region, keep it where it is.
            2. responders that have moved, can be assigned to any depot in that region since
                the low level planner will be immediately called

        :param region_allocation_action:
        :param state:
        :return:
        '''

        current_responder_to_region_allocation = {_[0]: _[1].region_assignment for _ in state.responders.items()}

        for resp_id, assigned_region in region_allocation_action.items():
            if current_responder_to_region_allocation[resp_id] != assigned_region:

                print('\t\t1 - resp assignment region {}'.format(state.responders[resp_id].region_assignment))

                self.environment_model.assign_resp_to_region(state, resp_id, assigned_region)
                print('\t\t2 - resp assignment region {}'.format(state.responders[resp_id].region_assignment))

                # assign to available depot
                depots_in_region = self.environment_model.get_depot_ids_in_region(state, assigned_region)
                print(depots_in_region)
                depot_ids_with_responders = [_[1].assigned_depot_id for _ in state.responders.items()]
                print(depot_ids_with_responders)
                available_depots = [_ for _ in depots_in_region if _ not in depot_ids_with_responders]
                print(available_depots)
                # depot_assignment = next(self.environment_model.get_depot_ids_in_region(state, assigned_region))
                depot_assignment = available_depots[0]

                if state.responders[resp_id].available:
                    self.environment_model.assign_responder_to_depot_and_move_there(state,
                                                                                    resp_id,
                                                                                    depot_assignment)
                else:
                    self.environment_model.assign_responder_to_depot(state,
                                                                     resp_id,
                                                                     depot_assignment)





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


    # def make_failure_state_before_log_file(self, state, file_name):
    #
    #     # get depots with responders
    #     depots_with_resp = [(_[0], _[1].assigned_depot_id, _[1].region_assignment) for _ in state.responders.items()]
    #
    #     coords_x = list()
    #     coords_y = list()
    #     resp_list = list()
    #     for item in depots_with_resp:
    #
    #         depot_obj = state.depots[item[1]]
    #         coords_x.append(depot_obj.lon)
    #         coords_y.append(depot_obj.lat)
    #         resp_list.append({'resp': item[0], 'assigned_depot_id': item[1], 'asgn_region': item[2]})
    #
    #     crs = {'init': 'epsg:4326'}
    #     depot_geo_df = gpd.GeoDataFrame(resp_list, index=range(len(coords_x)), geometry=gpd.points_from_xy(coords_y, coords_x),
    #                                     crs=crs)
    #
    #     depot_geo_df.to_file(filename=file_name, driver='GeoJSON')
    #
    #     print(depot_geo_df)


    def process_failure_event(self, state, event):

        # current failure events are based on regions. Therefore delete a responder from given region

        # self.make_failure_state_before_log_file(state, 'failure_log_before.geojson')

        region_to_remove_resp = event.type_specific_information['region']
        number_of_resp_to_fail =  event.type_specific_information['num_resp_to_fail']

        print('\t\tFailure event! Removing {} resp from region {}'.format(number_of_resp_to_fail, region_to_remove_resp))

        num_resp_total = len(state.responders)
        num_resp_in_region = len([_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp])
        print('\t\tnum resp before: {}, num in region before: {}'.format(num_resp_total, num_resp_in_region))

        # assume there is at least 'number_of_resp_to_fail' resp in the region

        removed_resp = list()

        for i in range(number_of_resp_to_fail):
            resp_keys_in_region = [_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp]

            # remove first responder
            removed_resp.append(copy.deepcopy(state.responders[resp_keys_in_region[0]].__dict__))
            del state.responders[resp_keys_in_region[0]]

        print(removed_resp)

        num_resp_total = len(state.responders)
        num_resp_in_region = len(
            [_[0] for _ in state.responders.items() if _[1].region_assignment == region_to_remove_resp])
        print('\t\tnum resp After: {}, num in region before: {}'.format(num_resp_total, num_resp_in_region))




