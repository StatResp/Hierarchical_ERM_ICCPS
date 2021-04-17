import copy
import sys
from multiprocessing import Pool
import multiprocessing
import time
from pprint import pprint

import numpy as np
import random

from Environment.CellTravelModel import GridCellRouter
from decision_making.LowLevel.CentralizedMCTS.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from decision_making.LowLevel.CentralizedMCTS.ModularMCTS import LowLevelMCTSSolver
from Environment.DataStructures.State import State
from Environment.DataStructures.Event import Event
from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import LLEventType, MCTStypes
from decision_making.LowLevel.CentralizedMCTS.Rollout import DoNothingRollout
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy


def run_low_level_mcts(arg_dict):
    '''
    arg dict needs:
    current_state,
    event_queue,
    iter_limit,
    allowed_compu_time,
    exploration_constant,
    discount_factor,
    rollout_policy,
    # reward_function,
    # travel_model,
    mdp_environment,
    MCTS_type
    :param arg_dict:
    :return:
    '''


    if arg_dict['MCTS_type'] == MCTStypes.CENT_MCTS:

        solver = LowLevelMCTSSolver(mdp_environment_model=arg_dict['mdp_environment'],
                                         discount_factor=arg_dict['discount_factor'],
                                         exploit_explore_tradoff_param=arg_dict['exploration_constant'],
                                         iter_limit=arg_dict['iter_limit'],
                                         allowed_computation_time=arg_dict['allowed_compu_time'],
                                         rollout_policy=arg_dict['rollout_policy'],
                                         predictor=arg_dict['predictor'])

        res = solver.solve(arg_dict['current_state'], arg_dict['event_queue'])

        return {'region_id': arg_dict['region_id'],
                'mcts_res': res}






class LowLevelCentMCTSPolicy:

    def __init__(self,
                 region_ids,
                 incident_prediction_model,
                 min_allocation_period,
                 lookahead_horizon_delta_t,
                 pool_thread_count,
                 mcts_type,
                 mcts_discount_factor,
                 mdp_environment_model,
                 rollout_policy,
                 uct_tradeoff,
                 iter_limit,
                 allowed_computation_time
                 ):
        self.allowed_computation_time = allowed_computation_time
        self.iter_limit = iter_limit
        self.uct_tradeoff = uct_tradeoff
        self.rollout_policy = rollout_policy
        self.mdp_environment_model = mdp_environment_model
        self.mcts_type = mcts_type
        self.mcts_discount_factor = mcts_discount_factor
        self.pool_thread_count = pool_thread_count
        self.lookahead_horizon_delta_t = lookahead_horizon_delta_t
        self.min_allocation_period = min_allocation_period
        self.incident_prediction_model = incident_prediction_model
        self.region_ids = region_ids

    def process(self, state):
        '''
        1. split up state into regions
        2. for each region, perform several paralellized mcts allocations
        3. for each region, merge paralell results; decide final action
        4. for each region, actuate actions on state

        will need:
        - environmental model (dispatch and response time, updates, etc.)
        - event queue (sampled incidents plus low level periodic events (if nessisary)
        - parallel pool setup
        - mcts: give state and event, get back each possible event with a score
        - score merging
        - actuation using evnironmental model on acutal state for best actions for each region
        :param state:
        :param event:
        :return:
        '''

        curr_allocation_event = Event(event_type=LLEventType.ALLOCATION,
                                cell_loc=None,
                                time=state.time)

        compu_state = copy.deepcopy(state)

        # dict of region_id => state
        region_states = self.split_state_to_regions(compu_state)

        # dict of region_id => event queue
        # TODO see what happens if future allocations are ignored
        region_event_queues = self.get_event_queues(compu_state, curr_allocation_event)
        # region_event_queues = self.get_event_queues_no_allocations(compu_state, curr_allocation_event)

        # process each of the regions
        return self.get_allocation(region_states, region_event_queues)



    def is_extra_low_level_event_needed(self, next_event):

        pass

    def split_state_to_regions(self, state):

        split_states = {}

        for region_id in self.region_ids:
            region_resp = {_[0]: _[1] for _ in state.responders.items() if _[1].region_assignment == region_id}
            # region_cells = {_[0]: _[1] for _ in state.cells.items() if _[1]['region'] == region_id}
            # region_cells = {_[0]: _[1] for _ in state.cells.items() if _[1] == region_id}
            region_cells = state.regions[region_id]
            region_depots = {_[0]: _[1] for _ in state.depots.items() if _[1].cell_loc in region_cells}
            region_active_incidents = [_ for _ in state.active_incidents if _.cell_loc in region_cells]

            region_state = State(responders=region_resp,
                                 depots=region_depots,
                                 active_incidents=region_active_incidents,
                                 time=state.time,
                                 cells=region_cells,
                                 regions=state.regions)

            split_states[region_id] = region_state

        return split_states

    def get_event_queues(self, state, current_event):
        '''
        This method creates the decision making event queues for each region.
        These queues contain predicted incident events for the region, and allocation events.
        Allocation events occur after each incident, and after a waiting period if no allocation event
        Has occured for a set amount of time.

        Assume we use all chains
        At the end of processing, have structure like the following:
        {'region_i': {'chain_1': [event, event, event...], 'chain_2': [event, ...], ...}, 'region_j': {...}, ...}
        :param state:
        :return:
        '''

        # TODO | should have a dummy last allocation event?

        region_event_queues = {}
        lookahead_horizon = state.time + self.lookahead_horizon_delta_t

        # TODO | this returns normal incident events. Need to change to LL events types.
        # format:[[0_event, 0_event, ...], [1_event, 1_event, ...], ....]
        full_event_queues = self.incident_prediction_model.get_chains(state.time)

        for region_id in self.region_ids:
            region_queues = []
            for raw_queue in full_event_queues:
                # first filter events to only events in this region and within time horizon
                queue_filtered_to_region = [_ for
                                            _ in raw_queue
                                            if state.cells[_.cell_loc] == region_id
                                            and _.time <= lookahead_horizon
                                            and _.time > current_event.time]
                queue_filtered_to_region.sort(key=lambda _: _.time)  # ensure sorted by time

                # now process the queue. Things to do are:
                # - change event types to LLEventType
                # - add allocation events. These occur...after each incident and if
                processed_queue = []

                processed_queue.append(copy.deepcopy(current_event))

                # possible that queue has no events
                if len(queue_filtered_to_region) > 0:

                    last_allocation_time = queue_filtered_to_region[0].time
                    for incident_event in queue_filtered_to_region:
                        # add allocation events until next incident time
                        while last_allocation_time + self.min_allocation_period < incident_event.time:
                            last_allocation_time += self.min_allocation_period
                            # print('extra allocation triggered - curr time: {}, next incident time: {}'.format(last_allocation_time, incident_event.time))
                            processed_queue.append(Event(event_type=LLEventType.ALLOCATION,
                                                         cell_loc=None,
                                                         time=last_allocation_time,
                                                         type_specific_information=None))

                        # add incident event
                        processed_queue.append(Event(event_type=LLEventType.INCIDENT,
                                                     cell_loc=incident_event.cell_loc,
                                                     time=incident_event.time,
                                                     type_specific_information={'incident_obj': incident_event.type_specific_information['incident_obj']}))

                        # add allocation event in response to incident
                        processed_queue.append(Event(event_type=LLEventType.ALLOCATION,
                                                     cell_loc=None,
                                                     time=incident_event.time,
                                                     type_specific_information=None))

                        last_allocation_time = incident_event.time

                else:
                    # queue is empty. fill using allocations until horizon
                    last_allocation_time = current_event.time
                    # add allocation events until next incident time
                    while last_allocation_time + self.min_allocation_period < lookahead_horizon:
                        last_allocation_time += self.min_allocation_period
                        # print('extra allocation triggered - curr time: {}, next incident time: {}'.format(last_allocation_time, incident_event.time))
                        processed_queue.append(Event(event_type=LLEventType.ALLOCATION,
                                                     cell_loc=None,
                                                     time=last_allocation_time,
                                                     type_specific_information=None))

                region_queues.append(processed_queue)
            region_event_queues[region_id] = region_queues

        return region_event_queues


    # def get_event_queues_no_allocations(self, state, current_event):
    #     '''
    #     # TODO - testing if having no future allocations is helpful
    #
    #     This method creates the decision making event queues for each region.
    #     These queues contain predicted incident events for the region, and allocation events.
    #     Allocation events occur after each incident, and after a waiting period if no allocation event
    #     Has occured for a set amount of time.
    #
    #     Assume we use all chains
    #     At the end of processing, have structure like the following:
    #     {'region_i': {'chain_1': [event, event, event...], 'chain_2': [event, ...], ...}, 'region_j': {...}, ...}
    #     :param state:
    #     :return:
    #     '''
    #
    #     # TODO | should have a dummy last allocation event?
    #
    #     region_event_queues = {}
    #     lookahead_horizon = state.time + self.lookahead_horizon_delta_t
    #
    #     # TODO | this returns normal incident events. Need to change to LL events types.
    #     # format:[[0_event, 0_event, ...], [1_event, 1_event, ...], ....]
    #     full_event_queues = self.incident_prediction_model.get_chains(state.time)
    #
    #     for region_id in self.region_ids:
    #         region_queues = []
    #         for raw_queue in full_event_queues:
    #             # first filter events to only events in this region and within time horizon
    #             queue_filtered_to_region = [_ for
    #                                         _ in raw_queue
    #                                         if state.cells[_.cell_loc] == region_id
    #                                         and _.time <= lookahead_horizon
    #                                         and _.time > current_event.time]
    #             queue_filtered_to_region.sort(key=lambda _: _.time)  # ensure sorted by time
    #
    #             # now process the queue. Things to do are:
    #             # - change event types to LLEventType
    #             # - add allocation events. These occur...after each incident and if
    #             processed_queue = []
    #
    #             processed_queue.append(copy.deepcopy(current_event))
    #
    #             # possible that queue has no events
    #             for incident_event in queue_filtered_to_region:
    #                 # add allocation events until next incident time
    #
    #                 # add incident event
    #                 processed_queue.append(Event(event_type=LLEventType.INCIDENT,
    #                                              cell_loc=incident_event.cell_loc,
    #                                              time=incident_event.time,
    #                                              type_specific_information={'incident_obj': incident_event.type_specific_information['incident_obj']}))
    #
    #             region_queues.append(processed_queue)
    #         region_event_queues[region_id] = region_queues
    #
    #     return region_event_queues
    #



    def get_allocation(self, region_states, region_event_queues):

        # TODO this is just for testing

        region_ids = list(region_states.keys())

        final_allocation = {}

        start_pool_time = time.time()
        with Pool(processes=self.pool_thread_count) as pool:

            pool_creation_time = time.time() - start_pool_time

            # print('pool time: {}'.format(pool_creation_time))

            inputs = self.get_mcts_inputs(region_states=region_states,
                                          region_event_queues=region_event_queues,
                                          discount_factor=self.mcts_discount_factor,
                                          mdp_environment_model=self.mdp_environment_model,
                                          rollout_policy=self.rollout_policy,
                                          uct_tradeoff=self.uct_tradeoff,
                                          iter_limit=self.iter_limit,
                                          allowed_computation_time=self.allowed_computation_time,
                                          mcts_type=self.mcts_type,
                                          predictor=self.incident_prediction_model)

            # random.shuffle(inputs)

            # run_start_ = time.time()
            res_dict = pool.map(run_low_level_mcts, inputs)
            # res_dict = pool.imap(run_low_level_mcts, inputs, chunksize=5)
            # print('pool run time: {}'.format(time.time() - run_start_))

            best_actions = dict()

            for region_id in region_ids:
                region_results = [_['mcts_res'] for _ in res_dict if _['region_id'] == region_id]

                # assumes that actions are the same across each run
                actions = [_['action']['action'] for _ in region_results[0]['scored_actions']]

                all_action_scores = []
                for action in actions:
                    action_scores = []
                    for result in region_results:
                        action_score = next((_ for _ in result['scored_actions'] if _['action']['action'] == action), None)
                        action_scores.append(action_score['score'])

                    all_action_scores.append({'action': action, 'scores': action_scores})

                avg_action_scores = list()
                for res in all_action_scores:
                    avg_action_scores.append({'action': res['action'],
                                              'avg_score': np.mean(res['scores'])})

                # TODO | do we want to minimize?
                # best_actions[region_id] = min(avg_action_scores, key=lambda _: _['avg_score'])['action']
                best_actions[region_id] = max(avg_action_scores, key=lambda _: _['avg_score'])['action']


            # print(best_actions)

            for region_id, action_dict in best_actions.items():
                for resp_id, depot_id in action_dict.items():
                    final_allocation[resp_id] = depot_id

        return final_allocation

            # now need to combine into one allocation





    def get_mcts_inputs(self,
                        region_states,
                        region_event_queues,
                        discount_factor,
                        mdp_environment_model,
                        rollout_policy,
                        uct_tradeoff,
                        iter_limit,
                        allowed_computation_time,
                        mcts_type,
                        predictor):

        inputs = []

        for region_id, region_state in region_states.items():
            for event_queue in region_event_queues[region_id]:
                input_dict = {}

                input_dict['MCTS_type'] = mcts_type
                input_dict['mdp_environment'] = copy.deepcopy(mdp_environment_model)
                input_dict['discount_factor'] = discount_factor
                input_dict['exploration_constant'] = uct_tradeoff
                input_dict['iter_limit'] = iter_limit
                input_dict['allowed_compu_time'] = allowed_computation_time
                input_dict['rollout_policy'] = copy.deepcopy(rollout_policy)
                input_dict['current_state'] = copy.deepcopy(region_state)
                input_dict['event_queue'] = copy.deepcopy(event_queue)
                input_dict['region_id'] = region_id
                input_dict['predictor'] = predictor

                inputs.append(input_dict)

        return inputs

        # region_inputs = dict()
        # num_event_queues = None
        # for region_id, queue in region_event_queues.items():
        #     if num_event_queues is None:
        #         num_event_queues = len(queue)
        #     else:
        #         assert len(queue) == num_event_queues
        #
        # for region_id, region_state in region_states.items():
        #     _region_input = list()
        #
        #
        #     for event_queue in region_event_queues[region_id]:
        #         input_dict = {}
        #
        #         input_dict['MCTS_type'] = mcts_type
        #         input_dict['mdp_environment'] = copy.deepcopy(mdp_environment_model)
        #         input_dict['discount_factor'] = discount_factor
        #         input_dict['exploration_constant'] = uct_tradeoff
        #         input_dict['iter_limit'] = iter_limit
        #         input_dict['allowed_compu_time'] = allowed_computation_time
        #         input_dict['rollout_policy'] = copy.deepcopy(rollout_policy)
        #         input_dict['current_state'] = copy.deepcopy(region_state)
        #         input_dict['event_queue'] = copy.deepcopy(event_queue)
        #         input_dict['region_id'] = region_id
        #         input_dict['predictor'] = predictor
        #
        #         _region_input.append(input_dict)
        #
        #     region_inputs[region_id] = _region_input
        #
        # processed_inputs = list()
        #
        # for i in range(num_event_queues):
        #     for region_id, input in region_inputs.items():
        #         processed_inputs.append(input[i])
        #
        #
        #
        #
        # # pprint(processed_inputs)
        # # print(len(processed_inputs))
        # #
        # # sys.exit()
        # return processed_inputs




if __name__ == "__main__":

    from scenarios.gridworld_example.definition.grid_world_gen_state import start_state
    from Environment.EnvironmentModel import EnvironmentModel
    from Prediction.Predictor_1.TESTIncidentPredictor import TESTIncidentPredictor

    predictor = TESTIncidentPredictor(num_chains=5)

    chains = predictor.get_chains(000)

    region_ids = list(start_state.regions.keys())

    travel_model = GridCellRouter(60.0 / 3600.0)
    environment = DecisionEnvironmentDynamics(travel_model, SendNearestDispatchPolicy(travel_model))


    llpolicy = LowLevelCentMCTSPolicy(region_ids=region_ids,
                                      incident_prediction_model=predictor,
                                      min_allocation_period=60*60*0.5,  # Every half hour
                                      lookahead_horizon_delta_t=60*60*2,  # look ahead 2 hours
                                      pool_thread_count=multiprocessing.cpu_count() - 3,
                                      mcts_type = MCTStypes.CENT_MCTS,
                                      mcts_discount_factor=0.99999,
                                      mdp_environment_model=environment,
                                      rollout_policy=DoNothingRollout(),
                                      uct_tradeoff=1.44,
                                      iter_limit=None,
                                      allowed_computation_time=10  # 5 seconds per thread
                                      )

    region_split_states = llpolicy.split_state_to_regions(start_state)

    split_event_queues = llpolicy.get_event_queues_no_allocations(start_state, Event(event_type=LLEventType.ALLOCATION,
                                                                      time=10,
                                                                      cell_loc=None))

    # print(split_event_queues)

    llpolicy.get_allocation(region_split_states, split_event_queues)






