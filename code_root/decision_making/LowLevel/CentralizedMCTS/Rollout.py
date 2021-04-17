import copy
import time

from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import LLEventType, ActionType, DispatchActions
from decision_making.LowLevel.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.DataStructures.State import State

class DoNothingRollout:

    def __init__(self):

        self.deep_copy_time = 0

    def rollout(self,
                node,
                environment_model,
                discount_factor,
                solve_start_time):


        # rollout state until event queue is empty

        # s_copy_time = time.time()
        # _node = copy.deepcopy(node)
        # self.deep_copy_time += time.time() - s_copy_time

        s_copy_time = time.time()

        # TODO | might need to change this if actions can change cells, etc.
        _state = State(
            responders=copy.deepcopy(node.state.responders),
            depots=copy.deepcopy(node.state.depots),
            active_incidents=copy.copy(node.state.active_incidents),  # TODO | maybe just copy here as well?
            time=node.state.time,
            cells=node.state.cells,
            regions=node.state.regions
        )
        _node = TreeNode(
            state=_state,
            parent=None,
            depth=node.depth,
            is_terminal=node.is_terminal,
            possible_actions=None,
            action_to_get_here=None,
            score=node.score,
            num_visits=None,
            children=None,
            reward_to_here=node.reward_to_here,
            is_fully_expanded=False,  # TODO
            actions_taken_tracker=None,
            action_sequence_to_here=None,
            event_at_node=node.event_at_node,
            future_events_queue= copy.copy(node.future_events_queue) # TODO | make sure copying doesn't affect actual node
        )
        self.deep_copy_time += time.time() - s_copy_time

        # self.rollout_iter(_node, environment_model, discount_factor, solve_start_time)

        while _node.future_events_queue:
            self.rollout_iter(_node, environment_model, discount_factor, solve_start_time)

        return _node.reward_to_here



    def rollout_iter(self, node, environment_model, discount_factor, solve_start_time):
        _curr_event = node.event_at_node
        # _future_event_queue = node.future_events_queue

        # if the current event is dispatch, send nearest and get possible available action.
        # if the current event is available, try dispatch
        # if the current event is allocation, do nothing
        action_to_take = None
        if _curr_event.event_type == LLEventType.INCIDENT:
            action_to_take = {'type': ActionType.DISPATCH, 'action': DispatchActions.SEND_NEAREST}
        elif _curr_event.event_type == LLEventType.RESPONDER_AVAILABLE:
            action_to_take = {'type': ActionType.DISPATCH, 'action': DispatchActions.SEND_NEAREST}
        elif _curr_event.event_type == LLEventType.ALLOCATION:
            action_to_take = None
        else:
            raise Exception('unsupported event type')

        if action_to_take is not None:
            immediate_reward, new_event, event_time = environment_model.take_action(node.state,
                                                                        action_to_take)
        else:
            immediate_reward = 0
            new_event = None
            event_time = node.state.time

        if new_event is not None:
            node.future_events_queue.append(new_event)
            node.future_events_queue.sort(key= lambda _: _.time)

        node.depth += 1
        node.event_at_node = node.future_events_queue.pop(0)
        self.process_event(node.state, node.event_at_node, environment_model)

        discounted_immediate_score = self.standard_discounted_score(immediate_reward,
                                                                    event_time - solve_start_time,
                                                                    discount_factor)

        node.reward_to_here = node.reward_to_here + discounted_immediate_score

    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start
        discounted_reward = discount * reward
        return discounted_reward


    def process_event(self, state, event, environment_model):

        # first update the state to the event time
        environment_model.update(state, event.time)

        if event.event_type == LLEventType.INCIDENT:
            '''
            If the event is an incident, do the following: 
            - add the incident to the incident queue
            - send a responder to the incident
            - record the response times
            '''
            incident = event.type_specific_information['incident_obj']
            environment_model.add_incident(state, incident)