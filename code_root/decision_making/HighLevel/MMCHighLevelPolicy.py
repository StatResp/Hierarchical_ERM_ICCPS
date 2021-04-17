from math import factorial, ceil
from numpy import average
import copy



class MMCHighLevelPolicy:

    # TODO | Question - what if there are less responders than the total number of regions?
    # assume start with 1 resp per region, yes?

    def __init__(self,
                 prediction_model,
                 within_region_travel_model,
                 trigger_wait_time_threshold,
                 incident_service_time_model,
                 between_region_travel_model):

        self.between_region_travel_model = between_region_travel_model
        self.incident_service_time_model = incident_service_time_model
        self.trigger_wait_time_threshold = trigger_wait_time_threshold
        self.within_region_travel_model = within_region_travel_model  # get_expected_travel_time(region_id, time, num_responders)
        self.prediction_model = prediction_model  # have get_rate(cell, time)

    def check_if_triggered(self,
                           state):
        '''
        Checks if the top level planner needs to be run.
        Based off average wait times between regions
        :param state:
        :return: bool
        '''

        region_ids = list(state.regions.keys())

        wait_times = []
        region_rates = []

        print('\ttime: {}'.format(state.time))

        for region in region_ids:
            region_incident_rate = self.get_region_total_rate(state.time, state.regions[region])
            print('\t{} - rate: {}'.format(region, region_incident_rate))
            region_rates.append(region_incident_rate)
            wait_times.append(self.get_region_ems_wait_time(state, region, region_incident_rate))

        weighted_average_wait_time = average(wait_times, weights=region_rates)  # TODO should this be rate-weighted?

        print('\tweighted_avg_wait_time: {}'.format(weighted_average_wait_time))

        # TODO | what if wait time is negative due to insufficient resources?
        # return weighted_average_wait_time > self.trigger_wait_time_threshold
        return True


    def region_responder_allocation(self,
                                    state,
                                    event):
        '''
        Do the following:
        - use greedy responder allocation to determine counts of responders to each region
        - use these counts to determine displaced responders and regions needing responders
        - non-displaced responders are kept in their current region
        - displaced responders are assigned to a new region by minimizing distance greedily
            * will need travel model between regions
        - now assign responders to their new region
        - move the responders to their new region -> use depot in new region in cell with highest rate?
        :param state:
        :param event:
        :return: dict => {responder_id: region}
        '''

        total_responders = len(state.responders)
        region_ids = list(state.regions.keys())

        region_incident_rates = dict()
        for r in region_ids:
            region_incident_rates[r] = self.get_region_total_rate(state.time, state.regions[r])

        region_depot_counts = dict()
        for region_id in region_ids:
            region_depot_counts[region_id] = len([_[0] for _ in state.depots.items() if
                                           _[1].cell_loc in state.regions[region_id]])

        ###################################
        # first get the new greedy assignment
        new_region_to_responder_allocation_count = self.greedy_responder_allocation_algorithm(region_ids=region_ids,
                                                                                              region_incident_rates=region_incident_rates,
                                                                                              total_responders=total_responders,
                                                                                              curr_time=state.time,
                                                                                              region_to_depot_count=region_depot_counts)


        print('\tallocation: {}'.format(new_region_to_responder_allocation_count))

        # TODO | ****** JUST FOR TESTING ****** |
        # new_region_to_responder_allocation_count = {'region_1': 2, 'region_2': 1}
        # TODO | ****** Remove for deployment ****** |

        #######################
        # get the reponders that should move and where to
        current_responder_to_region_allocation = {_[0]: _[1].region_assignment for _ in state.responders.items()}
        current_region_to_responder_allocation_count = dict()
        for region in region_ids:
            current_region_to_responder_allocation_count[region] = len([_ for _ in current_responder_to_region_allocation.values() if _ == region])

        displaced_responders = list()
        regions_needing_responders = list()

        # choose displaced responders
        for region_id in region_ids:
            curr_resp_count = current_region_to_responder_allocation_count[region_id]
            new_resp_count = new_region_to_responder_allocation_count[region_id]

            if curr_resp_count > new_resp_count:
                # some responders are displaced in this region
                # choose from available responders in region if there are enough
                count_discrepency = curr_resp_count - new_resp_count

                available_resp_in_region = [(_[0], _[1])
                                            for _ in state.responders.items()
                                            if _[1].region_assignment == region_id and _[1].available
                                            ]

                if len(available_resp_in_region) >= count_discrepency:
                    # there are enough to cover it. choose the 'first' ones
                    displaced_responders.extend(available_resp_in_region[:count_discrepency])

                else:
                    # there were not enough available responders.
                    # In this case, send all available and choose 'first' from rest
                    available_to_send = available_resp_in_region
                    num_still_needed = count_discrepency - len(available_to_send)

                    rest_to_send = [(_[0], _[1])
                                    for _ in state.responders.items()
                                    if _[1].region_assignment == region_id and not _[1].available][:num_still_needed]

                    displaced_responders.extend(available_to_send)
                    displaced_responders.extend(rest_to_send)

            elif curr_resp_count < new_resp_count:
                # this region has responder vacancies
                num_vacancies = new_resp_count - curr_resp_count
                regions_needing_responders.append({'region_id': region_id, 'num_needed': num_vacancies})

            # if responder counts are equal, do nothing


        assert len(displaced_responders) == sum([_['num_needed'] for _ in regions_needing_responders])

        ####
        # perform greedy matching between displaced responders and regions. Do it from the responder perspective

        working_displaced_resp = copy.deepcopy(displaced_responders)
        working_regions_needing_responders = copy.deepcopy(regions_needing_responders)
        assigned_resp_region_pairs = list()

        while len(working_displaced_resp) > 0:
            # goal: find responder-region pair that is the shortest distance
            best_candiate_resp_index = None
            best_candiate_resp_id = None
            best_candiate_resp_to_region_travel_time = float('inf')
            best_region_to_send_candiate_resp_to = None
            best_region_index = None

            # find next best resp -> depot assignment
            for resp_index, candidate_resp_info in enumerate(working_displaced_resp):
                candidate_resp_id = candidate_resp_info[0]
                candidate_resp_obj = candidate_resp_info[1]

                candidate_resp_curr_region = candidate_resp_obj.region_assignment

                _best_region_id = None
                _best_region_index = None
                _best_region_travel_time = float('inf')

                for _region_index, _region_info in enumerate(working_regions_needing_responders):
                    dist_between = self.between_region_travel_model.get_travel_time(region_1=candidate_resp_curr_region,
                                                                                    region_2=_region_info['region_id'],
                                                                                    curr_time=state.time)

                    if dist_between < _best_region_travel_time:
                        _best_region_id = _region_info['region_id']
                        _best_region_travel_time = dist_between
                        _best_region_index = _region_index

                if _best_region_travel_time < best_candiate_resp_to_region_travel_time:
                    best_candiate_resp_index = resp_index
                    best_candiate_resp_id = candidate_resp_id
                    best_candiate_resp_to_region_travel_time = _best_region_travel_time
                    best_region_to_send_candiate_resp_to = _best_region_id
                    best_region_index = _best_region_index

            # update info with chosen pair. involves
            # - update working displaced resp
            # - update working regions needing responders
            # - keep track of assigned pair
            assigned_resp_region_pairs.append((best_candiate_resp_id, best_region_to_send_candiate_resp_to))

            working_displaced_resp.pop(best_candiate_resp_index)

            working_regions_needing_responders[best_region_index]['num_needed'] -=1
            if working_regions_needing_responders[best_region_index]['num_needed'] <=0:
                # remove region; all reponders accounted for
                working_regions_needing_responders.pop(best_region_index)

            assert len(working_displaced_resp) == sum([_['num_needed'] for _ in working_regions_needing_responders])


        ######################
        # make return dictionary
        # will look like 'reps' -> 'region' (can be same region)

        final_allocation_dictionary = dict()

        displaced_responder_ids = [_[0] for _ in displaced_responders]
        for responder_id in state.responders.keys():
            if responder_id in displaced_responder_ids:
                # displaced
                region_to_move_resp_to = next(_[1] for _ in assigned_resp_region_pairs if _[0] == responder_id)
                final_allocation_dictionary[responder_id] = region_to_move_resp_to

            else:
                # not displaced, keep at current
                final_allocation_dictionary[responder_id] = state.responders[responder_id].region_assignment


        return final_allocation_dictionary

    def greedy_responder_allocation_algorithm(self,
                                              region_ids,
                                              region_incident_rates,
                                              total_responders,
                                              curr_time,
                                              region_to_depot_count):

        '''
        To do greedy responder allocation ...
        - first make sure that each region has enough responders to service their rates
        - Then begin greedy allocation

        Q:
        - what happens if there are not enough responders to handle the rates across the city?
        :param region_ids:
        :param region_incident_rates:
        :param total_responders:
        :return:
        '''


        ######################################
        # possibly do this
        # check that there is at least one resopnder for each region
        # if total_responders < len(region_ids):
        #     raise Exception('Allocation requires at least one responder per region')

        # first assign 1 responder to each region
        # region_num_assigned_resp = dict()
        # num_assigned_resp = len(region_ids)
        # for region_id in region_ids:
        #     region_num_assigned_resp[region_id] = 1


        #################################################
        # Assign responders to each region to match their incident rates
        # need E[service] > (E[incident] / n)
        region_num_assigned_resp = dict()
        num_assigned_resp = 0

        for region_id in region_ids:

            region_rate = region_incident_rates[region_id]
            # print('region {} rate: {}'.format(region_id, region_rate))
            expected_service_time = self.incident_service_time_model.get_expected_service_time(region_id, curr_time)
            expected_travel_time = self.within_region_travel_model.get_expected_travel_time(num_responders=1,
                                                                                            curr_time=curr_time,
                                                                                            region_id=region_id)  # assume worst case travel time

            region_service_rate = 1.0 / (expected_service_time + expected_travel_time)

            min_needed_resp = ceil(region_rate / region_service_rate)

            _start_assigned = None
            if min_needed_resp > region_to_depot_count[region_id]:
                _start_assigned = region_to_depot_count[region_id]
            else:
                _start_assigned = min_needed_resp

            region_num_assigned_resp[region_id] = _start_assigned

            num_assigned_resp += _start_assigned

            if num_assigned_resp > total_responders:
                raise('Not enough responders to meet incident rate')

        # sanity check to ensure all regions currently have positive waiting times
        is_some_region_negative = False
        for region_id, region_responder_count in region_num_assigned_resp.items():
            region_incident_rate = region_incident_rates[region_id]
            expected_service_time = self.incident_service_time_model.get_expected_service_time(region_id, curr_time)
            expected_travel_time = self.within_region_travel_model.get_expected_travel_time(num_responders=1,
                                                                                            curr_time=curr_time,
                                                                                            region_id=region_id)  # assume worst case travel time

            region_non_delay_service_rate = 1.0 / expected_service_time

            region_waiting_time = MMCHighLevelPolicy.expected_ems_response_time(number_of_servers=region_responder_count,
                                                                                incident_rate=region_incident_rate,
                                                                                service_rate=region_non_delay_service_rate,
                                                                                expected_travel_time=expected_travel_time)

            if region_waiting_time < 0:
                is_some_region_negative = True

        assert not is_some_region_negative

        #############################################
        # Greedily assign leftover responders
        # assume there are at least as many depots as responders
        while num_assigned_resp < total_responders:
            '''
            - Q: should 'score'  be weighted by rate within a region? -> yes
            '''

            best_waiting_time_so_far = float('inf')
            best_region_so_far = None
            for candiate_region_id in region_ids:

                if region_num_assigned_resp[candiate_region_id] >= region_to_depot_count[candiate_region_id]:
                    # skip region if it is full
                    continue

                _iter_region_counts = copy.deepcopy(region_num_assigned_resp)
                _iter_region_counts[candiate_region_id] += 1

                _wait_times = []
                _region_rates = []
                for r_id in region_ids:

                    region_incident_rate = region_incident_rates[r_id]
                    _region_rates.append(region_incident_rate)

                    expected_service_rate = 1.0 / self.incident_service_time_model.get_expected_service_time(r_id, curr_time)

                    r_waiting_time = self.get_ems_wait_time(responder_count=_iter_region_counts[r_id],
                                                            region_incident_rate=region_incident_rate,
                                                            service_rate=expected_service_rate,
                                                            time=curr_time,
                                                            region_id=r_id)

                    _wait_times.append(r_waiting_time)

                weighted_average_wait_time = average(_wait_times, weights=_region_rates)

                if weighted_average_wait_time < best_waiting_time_so_far:
                    best_waiting_time_so_far = weighted_average_wait_time
                    best_region_so_far = candiate_region_id

            region_num_assigned_resp[best_region_so_far] += 1
            num_assigned_resp += 1

        return region_num_assigned_resp


    def get_region_ems_wait_time(self,
                                 state,
                                 region_id,
                                 region_incident_rate):
        '''
        To determine the waiting time for a region, the following needs to occur:
        - cells that are a part of that region
        - the sum incident rates for that region => _lambda
        - get the expected travel time for the region given the number of responders
        - get the expected service time
        :param state:
        :param region_id:
        :return:
        '''

        responder_count = len([_[1] for _ in state.responders.items() if _[1].region_assignment == region_id])

        # region_cells = state.regions[region_id]
        # region_incident_rate = 0
        # for cell in region_cells:
        #     region_incident_rate += self.prediction_model.get_rate(cell, state.time)

        service_rate = 1.0 / self.incident_service_time_model.get_expected_service_time(region_id, state.time)

        return self.get_ems_wait_time(responder_count=responder_count,
                                      region_incident_rate=region_incident_rate,
                                      service_rate=service_rate,
                                      time=state.time,
                                      region_id=region_id)

    def get_region_total_rate(self, time, cells):
        region_incident_rate = 0
        for cell in cells:
            region_incident_rate += self.prediction_model.get_cell_rate(time, cell)

        return region_incident_rate



    def get_ems_wait_time(self,
                          responder_count,
                          region_incident_rate,
                          service_rate,
                          time,
                          region_id):

        expected_travel_time = self.within_region_travel_model.get_expected_travel_time(num_responders=responder_count,
                                                                                        curr_time=time,
                                                                                        region_id=region_id)

        return MMCHighLevelPolicy.expected_ems_response_time(number_of_servers=responder_count,
                                                             incident_rate=region_incident_rate,
                                                             service_rate=service_rate,
                                                             expected_travel_time=expected_travel_time)

    # Ayan suggested version
    @staticmethod
    def expected_ems_response_time(number_of_servers,
                                   incident_rate,
                                   service_rate,
                                   expected_travel_time):

        service_inter_arrival_time = 1.0 / service_rate
        delay_incorporated_service_time = expected_travel_time + service_inter_arrival_time

        delay_augmented_service_rate = 1.0 / delay_incorporated_service_time

        waiting_time = MMCHighLevelPolicy.waiting_time(_c=number_of_servers,
                                                       _lambda=incident_rate,
                                                       _mu=delay_augmented_service_rate)

        return waiting_time

    # @staticmethod
    # def expected_ems_response_time(number_of_servers,
    #                                incident_rate,
    #                                service_rate,
    #                                expected_travel_time):
    #     waiting_time = MMCHighLevelPolicy.waiting_time(_c=number_of_servers,
    #                                                    _lambda=incident_rate,
    #                                                    _mu=service_rate)
    #
    #     return waiting_time + expected_travel_time

    @staticmethod
    def Po(_c,
           _rho):

        _sum = 0
        for i in range(int(_c)):
            _sum += ((_c * _rho) ** i) / (factorial(i))

        return 1 / (_sum + (((_c * _rho) ** _c) / (factorial(_c) * (1 - _rho))))

    @staticmethod
    def Lq(_c, _lambda, _mu):
        _rho = _lambda / (_c * _mu)

        _top = MMCHighLevelPolicy.Po(_c, _rho) * ((_lambda / _mu) ** _c) * _rho
        _bottom = (factorial(_c) * ((1.0 - _rho) ** 2))

        return _top / _bottom

    @staticmethod
    def waiting_time(_c, _lambda, _mu):
        return MMCHighLevelPolicy.Lq(_c, _lambda, _mu) / _lambda





# if __name__ == "__main__":
#
#     # from scenarios.gridworld_example.definition.grid_world_gen_state import start_state
#
#     import random
#     class test_prediction_model:
#         def __init__(self, cell_ids, rate_lower_bound, rate_upper_bound):
#             self.cell_rates = {}
#             for cell_id in cell_ids:
#                 self.cell_rates[cell_id] = random.uniform(rate_lower_bound, rate_upper_bound)   # rates are incidents / second -> low number
#
#
#
#         def get_cell_rate(self, curr_time, cell_id):
#             # ignore time for now
#             return self.cell_rates[cell_id]
#
#
#     class test_region_travel_model:
#         def __init__(self, region_ids, total_number_responders):
#             self.model = dict()
#             for id in region_ids:
#                 region_model = dict()
#                 for i in range(1, total_number_responders + 1):
#                     region_model[i] = 600.0 / i
#
#                 self.model[id] = region_model
#
#         def  get_expected_travel_time(self, region_id, time, num_responders):
#
#             # ignore time for now
#             return self.model[region_id][num_responders]
#
#
#     class test_incident_service_time_model:
#         def __init__(self, region_ids):
#             self.model = dict()
#             for id in region_ids:
#                 self.model[id] = 0.5 * 60 * 60  # mean service time (NOT RATE) in seconds - 30 minutes?
#
#         def get_expected_service_time(self, region_id, curr_time):
#
#             # ignore time for now
#             return self.model[region_id]
#
#
#
#     class test_beteween_region_travel_model:
#         def __init__(self, travel_time_dict):
#             self.travel_time_dict = travel_time_dict
#
#         def get_travel_time(self, region_1, region_2, time):
#             return self.travel_time_dict[region_1][region_2]
#
#
#
#     prediction_model = test_prediction_model(cell_ids=list(start_state.cells.keys()),
#                                              rate_lower_bound= 0.25 / (1.0 * 60 * 60 * 24),  # 1 incident per 4 day
#                                              rate_upper_bound=  0.25 / (1.0 * 60 * 60 * 24) # 1 incident per 4 day
#                                              )
#     region_travel_model = test_region_travel_model(region_ids=list(start_state.regions.keys()),
#                                                    total_number_responders=len(start_state.responders))
#     trigger_wait_time_threshold = 5 * 60  # 5 minutes
#     incident_service_time_model = test_incident_service_time_model(region_ids=list(start_state.regions.keys()))
#
#     between_region_travel_model = test_beteween_region_travel_model({'region_1': {'region_2': 600},
#                                                                      'region_2': {'region_1': 600}})
#
#     a = MMCHighLevelPolicy(prediction_model=prediction_model,
#                            within_region_travel_model=region_travel_model,
#                            trigger_wait_time_threshold=trigger_wait_time_threshold,
#                            incident_service_time_model=incident_service_time_model,
#                            between_region_travel_model=between_region_travel_model)
#
#     # a.check_if_triggered(start_state)
#     #
#     # b = MMCHighLevelPolicy.expected_ems_response_time(number_of_servers=2,
#     #                                                   incident_rate= 1.0 / (0.8 * 60 * 60),
#     #                                                   service_rate= 1.0 / (0.5 * 60 * 60),
#     #                                                   expected_travel_time= 300
#     #                                                   )
#     #
#     # print(b)
#     #
#     # b = MMCHighLevelPolicy.expected_ems_response_time(number_of_servers=2,
#     #                                                   incident_rate=1.0 / (0.8 * 60 * 60),
#     #                                                   service_rate=1.0 / (300 + (0.5 * 60 * 60)),
#     #                                                   expected_travel_time=0
#     #                                                   )
#     #
#     # print(b)
#     #
#     # region_ids = list(start_state.regions.keys())
#     # region_incident_rates = dict()
#     # for r in region_ids:
#     #     region_incident_rates[r] = a.get_region_total_rate(start_state.time, start_state.regions[r])
#     #
#     #
#     # c = a.greedy_responder_allocation_algorithm(region_ids=region_ids,
#     #                                         region_incident_rates=region_incident_rates,
#     #                                         total_responders=len(start_state.responders),
#     #                                         curr_time=start_state.time)
#     #
#     # print(c)
#
#     d = a.region_responder_allocation(start_state, None)
#
#     print(d)

    # mu must be > than lambda
    # mu = 1 / E[service_time]
    # lambda = 1 / E[inter-arrival_time]

    # check: https://www.researchgate.net/profile/Sherif_Abdelwahed/publication/220654216_Performance_modeling_of_distributed_multi-tier_enterprise_systems/links/02e7e5331d24557f2a000000/Performance-modeling-of-distributed-multi-tier-enterprise-systems.pdf
    # source: http://web.mst.edu/~gosavia/queuing_formulas.pdf