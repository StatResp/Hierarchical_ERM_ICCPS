import copy

import numpy

from Environment.EnvironmentModel import EnvironmentModel
from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import ActionType, DispatchActions, LLEventType
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from Environment.DataStructures.Event import Event

import itertools

class DecisionEnvironmentDynamics(EnvironmentModel):

    def __init__(self, travel_model, send_nearest_dispatch, reward_policy=None):
        EnvironmentModel.__init__(self, travel_model=travel_model)
        self.reward_policy = reward_policy
        self.send_nearest_dispatch_model = send_nearest_dispatch
        self.travel_model = travel_model


    def generate_possible_actions(self, state, event, predictor):
        '''
        returns all possible actions in the current state and event.
        If it is an allocation event, it will correspond to responder to depot assignment combinations.
        If it is a dispatch event, it will correspond to the closest responder being dispatched?

        Question - should decision be which depots, then assign responders later? (assign closest non-responding responder)
        :param state:
        :return:
        '''

        if event.event_type == LLEventType.INCIDENT:
            # if it is an incident, need to send back 'send nearest' action
            possible_actions = [{'type': ActionType.DISPATCH, 'action': DispatchActions.SEND_NEAREST}]
            action_taken_tracker = [(_[0], False) for _ in enumerate(possible_actions)]

            return possible_actions, action_taken_tracker

        elif event.event_type == LLEventType.ALLOCATION:
            # for allocation events, we assume that all available dispatch has been completed.
            # it is possible that all responders are busy; in this case no allocation is necessary?

            # TODO | this assumes depots have unlimited capacities, correct?
            # TODO | could check each permutation if it violates a capacity constraint

            depots_ids = [_[0] for _ in state.depots.items()]
            responder_ids = [_[0] for _ in state.responders.items()]
            num_responders = len(responder_ids)

            # permutations = itertools.permutations(depots_ids, num_responders)
            _combinations = itertools.combinations(depots_ids, num_responders)

            possible_actions = []

            for _combination in _combinations:
                action = dict()
                action['type'] = ActionType.ALLOCATION
                # action['action'] = dict()

                # assign responders to depots based on distance. Greedily

                # for index, depots_id in enumerate(_combination):
                #     action['action'][responder_ids[index]] = depots_id

                action['action'] = self.resp_to_chosen_depots_assignment(responders=state.responders,
                                                                         depots=state.depots,
                                                                         chosen_depot_ids=list(_combination),
                                                                         resp_ids=responder_ids,
                                                                         curr_time=state.time,
                                                                         predictor=predictor)

                possible_actions.append(action)

            action_taken_tracker = [(_[0], False) for _ in enumerate(possible_actions)]

            return possible_actions, action_taken_tracker

        elif event.event_type == LLEventType.RESPONDER_AVAILABLE:
            # if we have resp available, it implies we need to dispatch. - check that there are incidents
            # TODO | future - could also mean that we can do some allocation
            if len(state.active_incidents) > 0:
                possible_actions = [{'type': ActionType.DISPATCH, 'action': DispatchActions.SEND_NEAREST}]
                action_taken_tracker = [(_[0], False) for _ in enumerate(possible_actions)]

                return possible_actions, action_taken_tracker
            else:
                possible_actions = [{'type': ActionType.DO_NOTHING}]
                # possible_actions =  [{'type': ActionType.DISPATCH, 'action': DispatchActions.SEND_NEAREST}]
                action_taken_tracker = [(_[0], False) for _ in enumerate(possible_actions)]

                return possible_actions, action_taken_tracker
                # raise Exception('should only have resp_available event if there are pending incidents')


    def resp_to_chosen_depots_assignment(self, responders,
                                         depots,
                                         chosen_depot_ids,
                                         resp_ids,
                                         curr_time,
                                         predictor):
        '''
        assign responders to the chosen depots. perform assignment based on distance traveled
        return dict - {resp_id: depot_id, ...}
        :param responders:
        :param depots:
        :param chosen_depot_ids:
        :return:
        '''

        _action = dict()
        # _working_chosen_depots = copy.deepcopy(chosen_depot_ids)
        # _working_resp_ids = copy.deepcopy(resp_ids)

        # TODO | try always having depots with highest rates go with available responders
        # TODO | so greedily assign responders to depots in decreasing order of rate


        # TODO | @AM - revert to original impl?

        _depot_rates = [(_[0], predictor.get_cell_rate(curr_time, _[1].cell_loc))
                        for _ in depots.items()
                        if _[0] in chosen_depot_ids]
        _depot_rates.sort(reverse=True, key=lambda _: _[1])

        _available_resp_ids = [_[0] for _ in responders.items() if _[1].available]
        _un_available_resp_ids = [_[0] for _ in responders.items() if not _[1].available]

        assert (len(_un_available_resp_ids) + len(_available_resp_ids)) == len(chosen_depot_ids)
        assert (len(_un_available_resp_ids) + len(_available_resp_ids)) == len(_depot_rates)

        while len(_available_resp_ids) > 0:
            # assign them to depots in decending order by rate
            depot_to_assign_to = _depot_rates.pop(0)[0]
            best_travel_time = float('inf')
            best_resp = None
            for _candidate_resp_id in _available_resp_ids:
                travel_time = self.travel_model.get_travel_time(responders[_candidate_resp_id].cell_loc,
                                                                depots[depot_to_assign_to].cell_loc,
                                                                curr_time)

                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_resp = _candidate_resp_id

            _available_resp_ids.remove(best_resp)
            _action[best_resp] = depot_to_assign_to

        # now unavailable responders
        while len(_un_available_resp_ids) > 0:
            # assign them to depots in decending order by rate
            depot_to_assign_to = _depot_rates.pop(0)[0]
            best_travel_time = float('inf')
            best_resp = None
            for _candidate_resp_id in _un_available_resp_ids:
                travel_time = self.travel_model.get_travel_time(responders[_candidate_resp_id].cell_loc,
                                                                depots[depot_to_assign_to].cell_loc,
                                                                curr_time)

                if travel_time < best_travel_time:
                    best_travel_time = travel_time
                    best_resp = _candidate_resp_id

            _un_available_resp_ids.remove(best_resp)
            _action[best_resp] = depot_to_assign_to

        assert len(list(_action.keys())) == len(chosen_depot_ids)
        assert len(_un_available_resp_ids) == 0
        assert len(_available_resp_ids) == 0
        assert len(_depot_rates) == 0
        return _action

        #####
        # old send nearest greedy code
        # while len(_working_chosen_depots) > 0:
        #     best_depot = None
        #     best_travel_time = float('inf')
        #     best_resp = None
        #
        #     for _candidate_depot_id in _working_chosen_depots:
        #         _resp = None
        #         _candidate_travel_time = float('inf')
        #         for _candidate_resp_id in _working_resp_ids:
        #             travel_time = self.travel_model.get_travel_time(responders[_candidate_resp_id].cell_loc,
        #                                                             depots[_candidate_depot_id].cell_loc,
        #                                                             curr_time)
        #
        #             if travel_time < _candidate_travel_time:
        #                 _candidate_travel_time = travel_time
        #                 _resp = _candidate_resp_id
        #
        #         if _candidate_travel_time < best_travel_time:
        #             best_resp = _resp
        #             best_depot = _candidate_depot_id
        #             best_travel_time = _candidate_travel_time
        #
        #     _working_resp_ids.remove(best_resp)
        #     _working_chosen_depots.remove(best_depot)
        #     _action[best_resp] = best_depot
        #
        # return _action









    def take_action(self, state, action):
        '''
        Action processing depends on the type of action.
        In all cases need to return the reward

        Dispatch actions => for now, just send nearest resource.
        Allocaiton action => need to assign responders to depots according to the aciton
        :param state:
        :param action:
        :return:
        '''

        if action['type'] == ActionType.DISPATCH:
            if action['action'] == DispatchActions.SEND_NEAREST:
                # send the nearest responder
                resp_time, new_event = self.dispatch_nearest_to_active_incidents(state, action)

                # TODO | should reward be more configurable?

                # try:
                #     assert len(resp_time) <= 1  # TODO | it is possible (however unlikely) that two responders become available at the same time
                #
                # except AssertionError as e:
                #     print(e)
                #     print(state.time, action)
                #     print(resp_time)
                #     raise e

                if len(resp_time) == 1:
                    resp_time_reward = resp_time[0]['resp_time']
                    incident_time = resp_time[0]['incident'].time

                    return -1 * resp_time_reward, new_event, incident_time

                elif len(resp_time) > 1:
                    resp_time_reward = numpy.mean([_['resp_time'] for _ in resp_time])
                    incident_time = resp_time[0]['incident'].time

                    return -1 * resp_time_reward, new_event, incident_time

                else:
                    return 0, new_event, state.time

            else:
                raise Exception('Dispatch Action not supported')

        elif action['type'] == ActionType.ALLOCATION:
            # Here, need to assign responders to the given depots
            # TODO | should I assert that the depot assignments are valid?
            for resp_id, dep_id in action['action'].items():
                if state.responders[resp_id].available:
                    # if available, assign and move
                    self.assign_responder_to_depot_and_move_there(state, resp_id, dep_id)
                else:
                    # only assign
                    self.resp_dynamics.assign_responder_to_depot(full_state=state,
                                                                 resp_id=resp_id,
                                                                 depot_id=dep_id)

                # print(state.responders[resp_id].assigned_depot_id)

            return 0, None, state.time # no reward for allocation actions at this time


        elif action['type'] == ActionType.DO_NOTHING:
            # this means we should do nothing. There is no reward, and no update to the state
            return 0, None, state.time

        else:
            raise Exception('Unsupported Action Type')

    def dispatch_nearest_to_active_incidents(self, state, action):
        resp_to_incident_tupples = self.send_nearest_dispatch_model.get_responder_to_incident_assignments(state)

        metrics = []

        for resp, incident in resp_to_incident_tupples:
            response_time = self.respond_to_incident(state, resp.my_id, incident.my_id)
            metrics.append({'resp': resp, 'incident': incident, 'resp_time': response_time})

        new_event = None

        if len(self.get_pending_incidents(state)) > 0 or len([_[1] for _ in state.responders.items() if _[1].available]) <= 0:
            # responders are saturated. Need event that corresponds to next responder availability

            responder_with_next_state_change = min(state.responders.values(),
                                              key=lambda _: _.available_time)

            new_event = Event(event_type=LLEventType.RESPONDER_AVAILABLE,
                         cell_loc=None,
                         time=responder_with_next_state_change.t_state_change,
                         type_specific_information=None)


        return metrics, new_event

