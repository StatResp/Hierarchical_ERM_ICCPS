import itertools
import copy
import math
import time
import random
from pprint import pprint

from decision_making.LowLevel.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.DataStructures.State import State
from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import LLEventType

class LowLevelMCTSSolver:
    def __init__(self,
                 mdp_environment_model,
                 discount_factor,
                 exploit_explore_tradoff_param,
                 iter_limit,
                 allowed_computation_time,
                 rollout_policy,
                 predictor):

        # assert (iter_limit is None) or (allowed_computation_time is None)

        self.predictor = predictor
        self.allowed_computation_time = allowed_computation_time
        self.rollout_policy = rollout_policy
        self.iter_limit = iter_limit
        self.exploit_explore_tradoff_param = exploit_explore_tradoff_param
        self.discount_factor = discount_factor
        self.mdp_environment_model = mdp_environment_model

        self.leaf_nodes = []
        self.solve_start_time = None
        self.number_of_nodes = None

        self.time_tracker = {'expand': 0,
                             'select': 0,
                             'rollout': 0}

        self.use_iter_lim = iter_limit is not None


    def solve(self,
              state,
              starting_event_queue):
        '''
        This will return the best action to take in the given state. Assumes that dispatching assignments are up to date.
        Assumes that state has been limited to only responders and depots for the zone of interest
        First event in event_queue is the current event
        :param state:
        :return:
        '''

        state = copy.deepcopy(state)
        self.solve_start_time = state.time
        self.number_of_nodes = 0

        possible_actions, actions_taken_tracker = self.get_possible_actions(state, starting_event_queue[0])

        _root_is_terminal = len(starting_event_queue[1:]) <= 0

        # init tree
        root = TreeNode(state=state,
                        parent=None,
                        depth=0,
                        is_terminal=_root_is_terminal,
                        possible_actions=possible_actions,
                        action_to_get_here=None,
                        score=0,
                        num_visits=0,
                        children=[],
                        reward_to_here=0.0,
                        is_fully_expanded=False,  # TODO
                        actions_taken_tracker=actions_taken_tracker,
                        event_at_node=starting_event_queue[0],
                        future_events_queue=starting_event_queue[1:])

        if self.use_iter_lim:
            iter_count = 0

            while iter_count < self.iter_limit:
                # print('\rIteration: {}'.format(iter_count), end='')
                iter_count += 1
                self.execute_iteration(root)
            # print('\niters complete')

        else:
            start_processing_time = time.time()
            curr_processing_time = 0
            iter_count = 0
            while curr_processing_time < self.allowed_computation_time:
                # print('\rIteration: {} at time {}/{}'.format(iter_count,
                #                                              curr_processing_time,
                #                                              self.allowed_computation_time),
                #       end='')
                curr_processing_time = time.time() - start_processing_time
                iter_count += 1
                self.execute_iteration(root)
            print('\niters complete')

        # self.leaf_nodes.sort(key=lambda _: _.reward_to_here, reverse=True)

        # print(self.leaf_nodes[0].action_sequence_to_here)
        # print(self.leaf_nodes[0].reward_to_here)

        # no_action_leaf_node = [_ for _ in self.leaf_nodes if len(list(filter(lambda _: _['c1'] is not None, _.action_sequence_to_here))) == 0]

        # normal child selection
        assert len(root.children) > 0  # TODO | it is possible for this to be NONE
        best_action = max(root.children, key= lambda _: _.score / _.num_visits).action_to_get_here

        # could also be best child?
        # best_action = self.leaf_nodes[0].action_sequence_to_here[0]
        # best_action = self.get_best_child(root).action_to_get_here

        # try max score
        # best_action = max(root.children, key=lambda _: _.best_score_seen_from_here).action_to_get_here

        # print('number of nodes: {}'.format(self.number_of_nodes))
        # print('time tracker: ')
        # print(self.time_tracker)

        # return best_action
        actions_with_scores = self.get_scored_child_actions(root)

        return {'scored_actions': actions_with_scores,
                'number_nodes': self.number_of_nodes,
                'time_taken': self.time_tracker}

    def get_scored_child_actions(self, node):

        scored_actions = []
        for child in node.children:
            action = child.action_to_get_here
            score = child.score / child.num_visits
            num_visits = child.num_visits

            scored_actions.append({'action': action,
                                   'score': score,
                                   'num_visits': num_visits})
        return scored_actions


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # testing code~~~~
    def get_sorted_actions(self, actions_with_scores):
        a = copy.deepcopy(actions_with_scores)
        a.sort(reverse=True, key=lambda _: _['score'])
        return a

    def get_current_resp_allocations(self, state):
        return [(_[0], _[1].assigned_depot_id) for _ in state.responders.items()]

    def get_incident_events(self, event_queue):
        return [_ for _ in event_queue if _.event_type == LLEventType.INCIDENT]

    def print_closest_depot_to_each_predicted_incident(self, state, event_queue):

        incident_events = self.get_incident_events(event_queue)
        for event in incident_events:
            print(self.mdp_environment_model.get_closest_depot_to_cell(event.cell_loc, state))

    def debug_(self, state, actions_with_scores, event_queue):
        pprint(self.get_sorted_actions(actions_with_scores)[:5])
        print(self.get_current_resp_allocations(state))
        self.print_closest_depot_to_each_predicted_incident(state, event_queue)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    # DONE
    def execute_iteration(self, node):

        select_start = time.time()
        selected_node = self.select_node(node)
        self.time_tracker['select'] += time.time() - select_start

        if not selected_node.is_terminal:
            expand_start = time.time()
            new_node = self.expand_node(selected_node)
            self.time_tracker['expand'] += time.time() - expand_start

            self.number_of_nodes += 1

        else:
            new_node = selected_node


        rollout_start = time.time()
        score = self.rollout_policy.rollout(new_node,
                                            self.mdp_environment_model,
                                            self.discount_factor,
                                            self.solve_start_time, )

        self.time_tracker['rollout'] += time.time() - rollout_start

        self.back_propagate(new_node, score)

    # DONE
    def pick_expand_action(self, node):
        # get unexplored actions
        # TODO check this
        unexplored_actions = [(node.possible_actions[_[0]], _[0]) for _ in node.actions_taken_tracker if not _[1]]
        num_unexplored_actions = len(unexplored_actions)
        if num_unexplored_actions == 0:
            print('num actions is 0?')
        if num_unexplored_actions == 1:
            node.is_fully_expanded = True

        random.seed(100)
        action_index = random.choice(range(num_unexplored_actions))
        picked_action = unexplored_actions[action_index][0]
        node.actions_taken_tracker[unexplored_actions[action_index][1]] = (unexplored_actions[action_index][1], True)

        return picked_action

    # DONE
    def add_event_to_event_queue(self, queue, event):

        queue.append(event)
        queue.sort(key=lambda _: _.time)

    # DONE
    def expand_node(self, node):
        # Pick action to take
        action_to_take = self.pick_expand_action(node)

        # copy the state
        # _new_state = copy.deepcopy(node.state)
        _new_state = State(
            responders=copy.deepcopy(node.state.responders),
            depots=copy.deepcopy(node.state.depots),
            active_incidents=copy.copy(node.state.active_incidents),  # TODO | maybe just copy here as well?
            time=node.state.time,
            cells=node.state.cells,
            regions=node.state.regions
        )

        # Take action in new state
        immediate_reward, new_event, event_time = self.mdp_environment_model.take_action(_new_state,
                                                                             action_to_take)

        _new_node_future_event_queue = copy.copy(node.future_events_queue)  # TODO | shallow copy okay?
        if new_event is not None:
            self.add_event_to_event_queue(_new_node_future_event_queue, new_event)

        # get the event associated with the new node and process it to fully update the state to the new time
        _expand_node_depth = node.depth + 1
        _expand_node_event = _new_node_future_event_queue.pop(0)
        self.process_event(_new_state, _expand_node_event)

        new_possible_actions, actions_taken_tracker = self.get_possible_actions(_new_state, _expand_node_event)

        assert len(new_possible_actions) > 0
        is_new_node_fully_expanded = False
        # if len(new_possible_actions) == 0:
        #     is_new_node_fully_expanded = True

        actions_taken_to_new_node = copy.copy(node.action_sequence_to_here)
        actions_taken_to_new_node.append(action_to_take)

        discounted_immediate_score = self.standard_discounted_score(immediate_reward,
                                                         event_time - self.solve_start_time,
                                                         self.discount_factor)

        reward_to_here = node.reward_to_here + discounted_immediate_score

        _expand_node_is_terminal = len(_new_node_future_event_queue) <= 0

        _new_node = TreeNode(
            state=_new_state,
            parent=node,
            depth=_expand_node_depth,
            is_terminal=_expand_node_is_terminal,
            possible_actions=new_possible_actions,
            action_to_get_here=action_to_take,
            score=0,
            num_visits=0,
            children=[],
            reward_to_here=reward_to_here,
            is_fully_expanded=is_new_node_fully_expanded,  # TODO
            actions_taken_tracker=actions_taken_tracker,
            action_sequence_to_here=actions_taken_to_new_node,
            event_at_node=_expand_node_event,
            future_events_queue= _new_node_future_event_queue
        )

        node.children.append(_new_node)
        # node.actions_taken_tracker[]
        if node in self.leaf_nodes:
            self.leaf_nodes.remove(node)
        self.leaf_nodes.append(_new_node)

        return _new_node

    # DONE
    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start
        discounted_reward = discount * reward
        return discounted_reward

    # DONE
    def get_best_child(self, node):

        best_val = float('-inf')
        best_nodes = []

        for child in node.children:
            value = self.uct_score(child)
            if value > best_val:
                best_val = value
                best_nodes = [child]
            elif value == best_val:
                best_nodes.append(child)

        random.seed(100)
        return random.choice(best_nodes)

    # DONE
    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node)
            else:
                return node
        return node  # simply returns terminal node if it is best child

    # DONE
    def back_propagate(self, node, score):
        while node is not None:
            node.num_visits += 1
            # if score > node.best_score_seen_from_here:
            #     node.best_score_seen_from_here = score
            node.score += score
            node = node.parent

    # DONE
    def uct_score(self, node):

        exploit = (node.score / node.num_visits)
        explore = math.sqrt(math.log(node.parent.num_visits) / node.num_visits)

        # TODO | I don't remember why I did this...to scale the parameter to the exploit value?
        # # trying with parameter on order of score
        scaled_explore_param = self.exploit_explore_tradoff_param * abs(exploit)
        scaled_explore_2 = scaled_explore_param * explore

        # scaled_explore = self.exploit_explore_tradoff_param * explore
        score = exploit + scaled_explore_2

        return score

    # DONE
    def get_possible_actions(self, state, event):

        return self.mdp_environment_model.generate_possible_actions(state, event, self.predictor)

    # DONE
    def process_event(self, state, event):
        '''
        Moves the state forward in time to the event. If it is an incident event, add the incident
        to pending incidents
        :param state:
        :param event:
        :return:
        '''

        # first update the state to the event time
        self.mdp_environment_model.update(state, event.time)

        if event.event_type == LLEventType.INCIDENT:
            incident = event.type_specific_information['incident_obj']
            self.mdp_environment_model.add_incident(state, incident)


# def create_test_incident_events():
#     incident_locations = [(1, 5),
#                           (5, 5),
#                           (1, 2),
#                           (1, 5)]
#
#     incident_times = [100,
#                       300,
#                       500,
#                       800]
#
#     clearance_time = 800
#
#     ids = [0, 1, 2, 3]
#
#     incident_events = []
#
#     for i in range(4):
#         incident = Incident(cell_loc=incident_locations[i],
#                             time=incident_times[i],
#                             clearance_time=clearance_time)
#
#         event = Event(event_type=LLEventType.INCIDENT,
#                       cell_loc=incident_locations[i],
#                       time=incident_times[i],
#                       # event_id=ids[i],
#                       type_specific_information={'incident_obj': incident})
#
#         incident_events.append(event)
#
#     return incident_events

# if __name__ == "__main__":
#
#     from scenarios.gridworld_example.definition.grid_world_gen_state import start_state
#     from Environment.EnvironmentModel import EnvironmentModel
#     from decision_making.LowLevel.CentralizedMCTS.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
#     from Environment.TravelModel import GridCellRouter
#     from  Environment.DataStructures.Event import Event
#     from Environment.DataStructures.Incident import Incident
#     from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import LLEventType
#     from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
#     from decision_making.LowLevel.CentralizedMCTS.Rollout import DoNothingRollout
#
#     travel_model = GridCellRouter(60.0 / 3600.0)
#     environment = DecisionEnvironmentDynamics(travel_model, SendNearestDispatchPolicy(travel_model))
#
#     test_solver = LowLevelMCTSSolver(mdp_environment_model=environment,
#                                      discount_factor=0.99999,
#                                      exploit_explore_tradoff_param=1.44,
#                                      iter_limit=20000,
#                                      rollout_policy=DoNothingRollout())
#
#
#     incident_events = create_test_incident_events()
#     final_event_queue = []
#     final_event_queue.append(Event(event_type=LLEventType.ALLOCATION,
#                                    cell_loc=None,
#                                    time=0))
#
#     for incident_event in incident_events:
#         final_event_queue.append(incident_event)
#         final_event_queue.append(Event(event_type=LLEventType.ALLOCATION,
#                                        cell_loc=None,
#                                        time=incident_event.time))
#
#
#     # print()
#     # test_solver.process_event(start_state, final_event_queue[1])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[1])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][0])
#     #
#     # test_solver.process_event(start_state, final_event_queue[2])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[2])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][24])
#     #
#     # test_solver.process_event(start_state, final_event_queue[3])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[3])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][0])
#     #
#     # test_solver.process_event(start_state, final_event_queue[5])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[5])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][0])
#     #
#     # # here we need to add new event
#     # test_solver.add_event_to_event_queue(final_event_queue, new_event)
#     #
#     # test_solver.process_event(start_state, final_event_queue[7])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[7])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][0])
#     #
#     # test_solver.process_event(start_state, final_event_queue[9])
#     # actions = test_solver.mdp_environment_model.generate_possible_actions(start_state, final_event_queue[9])
#     # reward, new_event, event_time = test_solver.mdp_environment_model.take_action(start_state, actions[0][0])
#     #
#     # print()
#
#     best_action = test_solver.solve(start_state, starting_event_queue=final_event_queue)
#
#     print(best_action)
#
#     print('deep time {}'.format(test_solver.rollout_policy.deep_copy_time))


