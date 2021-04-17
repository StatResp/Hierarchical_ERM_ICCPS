### Imports
import copy
import pickle
from datetime import datetime
import numpy as np
import random

from Environment.DataStructures.Event import Event
from Environment.DataStructures.State import State
from Environment.EnvironmentModel import EnvironmentModel
from Environment.Simulator import Simulator
from Environment.enums import EventType
from Prediction.ChainPredictor.ChainPredictor_Subset_Times import ChainPredictor_Subset_Times
from decision_making.HighLevel.MMCHighLevelPolicy import MMCHighLevelPolicy
from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import MCTStypes
from decision_making.LowLevel.CentralizedMCTS.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from decision_making.LowLevel.CentralizedMCTS.LowLevelCentMCTSPolicy import LowLevelCentMCTSPolicy
from decision_making.LowLevel.CentralizedMCTS.Rollout import DoNothingRollout
from decision_making.coordinator.DispatchOnlyCoord import DispatchOnlyCoord
from decision_making.coordinator.HighAndLowLevelCoord import HighAndLowLevelCoord
from decision_making.coordinator.LowLevelCoordTest import LowLevelCoord
from decision_making.coordinator.failure_coordinators.HighAndLowLevelCoord_Fail import HighAndLowLevelCoord_Fail
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from scenarios.Nashville_1.definition.generate_start_state import NashvilleStateGenerator
from Prediction.ChainPredictor.ChainPredictor import ChainPredictor
from Prediction.ChainPredictor.ChainPredictorReturnOne import ChainPredictorReturnOne
from scenarios.Nashville_1.NashvilleCellToCellTravelModel import NashvilleCellToCellTravelModel
from scenarios.Nashville_1.BetweenRegionTravelModel import BetweenRegionTravelModel
from scenarios.Nashville_1.WithinRegionTravelModel import NashvilleWithinRegionTravelModel
from Prediction.RealIncidentPredictor.RealIncidentPredictor import RealIncidentPredictor
from Prediction.ChainPredictor.ChainPredictor_Subset import ChainPredictor_Subset
from Environment.Spatial.spatialStructure import  get_cell_id_and_rates


import random
random.seed(100)
np.random.seed(100)
def get_resp_init_assignments_to_regions_and_depots(hl_policy,
                                                    region_ids,
                                                    curr_time,
                                                    region_to_cell_dict,
                                                    total_responders,
                                                    depots,
                                                    predictor):


    region_incident_rates = dict()
    for r in region_ids:
        region_incident_rates[r] = hl_policy.get_region_total_rate(curr_time, region_to_cell_dict[r])

    region_depot_counts = dict()
    for region_id in region_ids:
        region_depot_counts[region_id] = len([_[0] for _ in depots.items() if
                                              _[1].cell_loc in region_to_cell_dict[region_id]])

    initial_resp_to_region_allocation = hl_policy.greedy_responder_allocation_algorithm(
        region_ids=region_ids,
        region_incident_rates=region_incident_rates,
        total_responders=total_responders,
        curr_time=curr_time,
        region_to_depot_count=region_depot_counts)

    region_to_depots = dict()
    for region_id in region_ids:
        region_to_depots[region_id] = {_[0]: _[1] for _ in depots.items() if
                                       _[1].cell_loc in region_to_cell_dict[region_id]}

    depot_cell_rates = dict()
    for region_id in region_ids:
        region_depot_rates = list()
        for region_depot in region_to_depots[region_id].values():
            depot_cell_rate = predictor.get_cell_rate(curr_time, region_depot.cell_loc)
            region_depot_rates.append((region_depot.my_id, depot_cell_rate))

        region_depot_rates.sort(key=lambda _: _[1], reverse=True)
        depot_cell_rates[region_id] = region_depot_rates

    resp_assigned = 0
    resp_assignments = dict()
    resp_ids = list()
    for region_id, num_resp in initial_resp_to_region_allocation.items():
        sorted_depot_rates = depot_cell_rates[region_id]
        pos_in_depots = 0
        num_depots_in_region = len(sorted_depot_rates)
        for _i_resp in range(num_resp):
            if pos_in_depots % num_depots_in_region == 0:
                pos_in_depots = 0

            resp_id = str(resp_assigned)
            resp_assignments[resp_id] = {'region_id': region_id,
                                         'depot_id': sorted_depot_rates[pos_in_depots][0]}
            resp_ids.append(resp_id)

            resp_assigned += 1
            pos_in_depots += 1


    return resp_assignments


def create_high_level_policy(_wait_time_threshold,
                             _predictor,
                             region_centers,
                             _cell_travel_model,
                             region_mean_distance_dict,
                             travel_rate_cells_per_second,
                             mean_service_time,
                             region_ids):
    '''
    Constructs the high level policy object
    :param _wait_time_threshold:
    :param _predictor:
    :param region_centers:
    :param _cell_travel_model:
    :param region_mean_distance_dict:
    :param travel_rate_cells_per_second:
    :param mean_service_time:
    :param region_ids:
    :return:
    '''

    _between_region_travel_model = BetweenRegionTravelModel(region_centers=region_centers,
                                                            cell_travel_model=_cell_travel_model)

    _within_region_travel_model = NashvilleWithinRegionTravelModel(mean_dist_for_region_dict=region_mean_distance_dict,
                                                                   travel_cells_per_second=travel_rate_cells_per_second)

    # TODO
    class test_incident_service_time_model:
        def __init__(self, region_ids, mean_service_time):
            self.model = dict()
            for id in region_ids:
                self.model[id] = mean_service_time  # mean service time (NOT RATE) in seconds - 20 minutes?

        def get_expected_service_time(self, region_id, curr_time):

            # ignore time for now
            return self.model[region_id]

    service_time_model = test_incident_service_time_model(region_ids=region_ids,
                                                          mean_service_time=mean_service_time)

    hl_policy = MMCHighLevelPolicy(prediction_model=_predictor,
                                   within_region_travel_model=_within_region_travel_model,
                                   trigger_wait_time_threshold=_wait_time_threshold,
                                   incident_service_time_model=service_time_model,
                                   between_region_travel_model=_between_region_travel_model)

    return hl_policy





if __name__ == "__main__":

    ################################################
    ## constants across runs
    num_rows = 30
    num_columns = 30
    num_resp = 26
    travel_rate_cells_per_second = 30.0 / 3600.0  # cells are 1x1 miles; 30 mph
    wait_time_threshold = 10 * 60  # 10 minutes
    mean_service_time = 20 * 60

    min_time_between_allocation = 60 * 60 * 1.0
    lookahead_horizon_delta_t = 60 * 60 * 1.5
    mcts_type = MCTStypes.CENT_MCTS
    mcts_discount_factor = 0.99995
    uct_tradeoff = 1.44
    iter_limit = 250  # None
    allowed_computation_time = None  #10

    ####################
    # run specific constants
    data_directory = '../../../../data/'

    depot_info_path = data_directory + 'depots_with_cells.json'
    structural_valid_cells = pickle.load(open(data_directory + 'valid_cells_out.pkl', 'rb'))
    preprocessed_chains_file_path = data_directory + 'non_stationary_chains_60_v1.pickle'

    pool_thread_count = 1
    region_cluster_file = data_directory + 'cluster_output_5_r.pk'
    num_regions = 5

    start_time = datetime(year=2019, month=1, day=1).timestamp()
    end_time = datetime(year=2019, month=1, day=1, hour=5).timestamp() ##

    index_of_test_chain = 50
    indexes_of_sample_chains = list(range(5))
    #################################################

    config_info = dict()
    config_info['iter_lim'] = iter_limit
    config_info['discount_factor'] = mcts_discount_factor
    config_info['uct_tradeoff'] = uct_tradeoff
    config_info['lookahead_horizon'] = lookahead_horizon_delta_t
    config_info['start_time'] = start_time
    config_info['end_time'] = end_time

    structural_valid_cells = [str(_) for _ in structural_valid_cells]
    state_generator = NashvilleStateGenerator(num_rows=num_rows, num_columns=num_columns,
                                              structuraly_valid_cells=structural_valid_cells)
    cell_label_dict = state_generator.get_cell_label_to_x_y_coord_dict()

    # values extracted from incident data
    valid_cell_ids = ['107', '108', '109', '110', '111', '112', '113', '114', '115', '122', '123', '124', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '153', '154', '155', '160', '161', '162', '163', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '182', '183', '184', '185', '186', '187', '188', '191', '192', '193', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '23', '230', '231', '232', '233', '234', '235', '236', '237', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '267', '268', '269', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '298', '299', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '333', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '355', '357', '365', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '395', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '454', '455', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '473', '474', '475', '476', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '503', '504', '505', '506', '51', '514', '515', '517', '52', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '53', '530', '531', '532', '533', '534', '535', '54', '544', '545', '546', '547', '548', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '578', '579', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '609', '61', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '667', '668', '669', '670', '671', '673', '675', '676', '677', '678', '679', '680', '681', '699', '700', '701', '702', '703', '704', '705', '707', '708', '709', '710', '711', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '739', '758', '759', '760', '761', '763', '764', '765', '766', '767', '768', '788', '789', '790', '791', '794', '795', '797', '798', '80', '81', '818', '819', '82', '821', '825', '827', '828', '83', '84', '848', '849', '855', '856', '857', '886', '90', '91', '92']
    raw_rate_dict = {'107': 1.4081434607011027e-06, '108': 5.250704429732925e-07, '109': 1.1694750775314243e-06, '110': 3.5800257475451765e-07, '111': 7.398719878260032e-07, '112': 2.864020598036141e-07, '113': 1.4558771373350385e-06, '114': 5.560973327853507e-06, '115': 9.546735326787136e-08, '122': 1.4320102990180706e-07, '123': 2.386683831696784e-07, '124': 9.06939856044778e-07, '136': 1.050140885946585e-06, '137': 1.2410755924823277e-06, '138': 2.386683831696784e-07, '139': 2.0048144186252986e-06, '140': 3.8186941307148546e-07, '141': 9.06939856044778e-07, '142': 2.3628169933798165e-06, '143': 2.124148610210138e-06, '144': 1.360409784067167e-06, '145': 2.2434828017949772e-06, '146': 1.1456082392144565e-06, '153': 4.773367663393568e-08, '154': 1.4320102990180706e-07, '155': 1.3842766223841349e-06, '160': 1.503610813968974e-06, '161': 6.205377962411639e-07, '162': 4.773367663393568e-07, '163': 6.205377962411639e-07, '165': 7.350986201626095e-06, '166': 1.4797439756520063e-06, '167': 3.914161483982726e-06, '168': 2.8162869214022055e-06, '169': 8.759129662327198e-06, '170': 1.9809475803083308e-06, '171': 9.475134811836233e-06, '172': 1.801946292931072e-05, '173': 9.785403709956815e-07, '174': 3.627759424179112e-06, '175': 3.4606915559603373e-06, '176': 5.966709579241961e-07, '182': 2.386683831696784e-08, '183': 1.4320102990180706e-07, '184': 1.4320102990180706e-07, '185': 3.5800257475451765e-07, '186': 8.114725027769066e-07, '187': 1.1933419158483922e-06, '188': 1.4320102990180706e-06, '191': 3.341357364375498e-07, '192': 4.773367663393568e-08, '193': 2.148015448527106e-07, '195': 1.6945455205047169e-06, '196': 3.8186941307148546e-07, '197': 1.4320102990180706e-06, '198': 5.059769723197182e-06, '199': 2.0525480952592343e-06, '200': 6.563380537166156e-06, '201': 3.818694130714854e-06, '202': 9.785403709956815e-07, '203': 6.945249950237642e-06, '204': 1.2649424307992956e-06, '205': 2.0048144186252986e-06, '206': 4.057362513884533e-07, '211': 1.1217414008974886e-06, '212': 1.3842766223841349e-06, '213': 4.53469928022389e-07, '214': 7.398719878260032e-07, '215': 2.529884861598591e-06, '216': 2.1718822868440736e-06, '217': 4.53469928022389e-06, '218': 2.864020598036141e-07, '219': 1.9093470653574273e-07, '220': 2.6253522148664626e-07, '221': 2.386683831696784e-07, '222': 5.250704429732925e-07, '223': 3.341357364375498e-07, '224': 1.4320102990180706e-07, '225': 1.790012873772588e-06, '226': 8.878463853912037e-06, '227': 3.007221627937948e-06, '228': 1.0286607314613141e-05, '229': 4.248297220420276e-06, '23': 1.193341915848392e-07, '230': 1.1909552320166954e-05, '231': 2.267349640111945e-06, '232': 3.9618951606166615e-06, '233': 7.160051495090352e-06, '234': 1.074007724263553e-06, '235': 1.2410755924823277e-06, '236': 2.6253522148664626e-07, '237': 4.773367663393568e-08, '241': 3.5800257475451765e-07, '242': 9.546735326787136e-08, '243': 3.1026889812058193e-07, '244': 4.53469928022389e-07, '245': 7.398719878260032e-07, '246': 6.205377962411639e-07, '247': 5.012036046563247e-07, '248': 2.6253522148664626e-07, '249': 9.546735326787136e-07, '250': 4.773367663393568e-07, '251': 6.205377962411639e-07, '252': 1.4081434607011027e-06, '253': 5.489372812902604e-07, '254': 1.3365429457501992e-06, '255': 2.3389501550628487e-06, '256': 3.0549553045718837e-06, '257': 2.3866838316967844e-06, '258': 4.152829867152404e-06, '259': 1.2410755924823278e-05, '260': 6.30084531567951e-06, '261': 9.06939856044778e-07, '262': 7.3032525249921595e-06, '263': 2.4582843466476876e-06, '264': 2.3628169933798165e-06, '265': 3.8186941307148546e-07, '267': 1.670678682187749e-07, '268': 2.386683831696784e-08, '269': 2.386683831696784e-07, '272': 1.4320102990180706e-07, '273': 4.773367663393568e-08, '274': 4.773367663393568e-08, '275': 7.160051495090353e-08, '276': 7.637388261429709e-07, '277': 1.8616133887234917e-06, '278': 2.386683831696784e-08, '279': 1.4320102990180706e-07, '280': 2.124148610210138e-06, '281': 5.728041196072282e-07, '282': 6.682714728750996e-07, '283': 6.062176932509832e-06, '284': 2.2434828017949772e-06, '285': 4.367631412005115e-06, '286': 6.992983626871578e-06, '287': 7.804456129648485e-06, '288': 1.171861761363121e-05, '289': 3.7232267774469833e-06, '290': 5.012036046563247e-07, '291': 6.491780022215253e-06, '292': 7.876056644599388e-07, '293': 6.921383111920674e-07, '294': 4.773367663393568e-08, '295': 2.386683831696784e-08, '298': 2.864020598036141e-07, '299': 4.773367663393568e-08, '303': 1.670678682187749e-07, '304': 1.670678682187749e-07, '305': 4.773367663393568e-08, '306': 1.4320102990180706e-07, '307': 2.696952729817366e-06, '308': 8.830730177278101e-07, '309': 3.5800257475451765e-07, '310': 9.546735326787136e-07, '311': 3.747093615763951e-06, '312': 1.670678682187749e-06, '313': 7.995390836184228e-06, '314': 8.711395985693263e-06, '315': 1.038207466788101e-05, '316': 8.472727602523583e-06, '317': 1.019113996134527e-05, '318': 9.06939856044778e-06, '319': 4.677900310125697e-06, '320': 1.9809475803083308e-06, '321': 9.546735326787136e-07, '322': 1.4320102990180706e-07, '323': 8.830730177278101e-07, '324': 7.160051495090353e-08, '333': 4.773367663393568e-08, '335': 2.864020598036141e-07, '336': 1.9093470653574273e-07, '337': 1.193341915848392e-07, '338': 5.584840166170475e-06, '339': 4.367631412005115e-06, '340': 4.057362513884533e-07, '341': 3.4129578793264016e-06, '342': 2.9356211129870448e-06, '343': 7.183918333407321e-06, '344': 1.360409784067167e-05, '345': 1.0286607314613141e-05, '346': 3.174289496156723e-06, '347': 1.670678682187749e-06, '348': 2.1408553970320153e-05, '349': 5.966709579241961e-06, '350': 8.114725027769066e-07, '351': 4.892701854978408e-06, '352': 8.830730177278101e-07, '353': 1.2649424307992956e-06, '355': 2.386683831696784e-08, '357': 9.546735326787136e-08, '365': 1.670678682187749e-07, '369': 1.6468118438707811e-06, '370': 4.582432956857826e-06, '371': 1.2506223278091149e-05, '372': 5.728041196072282e-06, '373': 1.5250909684542451e-05, '374': 2.1146018748833507e-05, '375': 2.7494597741146955e-05, '376': 2.4272574568356296e-05, '377': 1.8974136461989435e-05, '378': 4.0334956755675655e-06, '379': 7.446453554893967e-06, '380': 7.73285561469758e-06, '381': 8.305659734304809e-06, '382': 2.1957491251610415e-06, '383': 3.770960454080919e-06, '384': 2.7924200830852376e-06, '385': 7.160051495090353e-08, '386': 2.386683831696784e-07, '387': 9.546735326787136e-08, '395': 1.670678682187749e-07, '399': 7.160051495090353e-08, '400': 2.1957491251610415e-06, '401': 2.0048144186252986e-06, '402': 3.198156334473691e-06, '403': 1.1026479302439142e-05, '404': 2.854473862709354e-05, '405': 3.9976954180921136e-05, '406': 2.0525480952592345e-05, '407': 6.444046345581317e-07, '408': 1.670678682187749e-07, '409': 7.160051495090353e-07, '410': 4.248297220420276e-06, '411': 4.20056354378634e-06, '412': 1.5513444906029097e-06, '413': 1.1933419158483922e-06, '414': 3.770960454080919e-06, '415': 6.157644285777703e-06, '416': 1.21720875416536e-06, '426': 9.546735326787136e-08, '427': 2.386683831696784e-08, '428': 9.546735326787136e-08, '429': 1.4320102990180706e-07, '430': 1.933213903674395e-06, '431': 1.670678682187749e-07, '432': 6.205377962411639e-07, '433': 6.348578992313446e-06, '434': 1.4916773948104902e-05, '435': 1.2888092691162635e-05, '436': 1.6372651085439938e-05, '437': 7.470320393210935e-06, '438': 5.250704429732925e-07, '439': 9.546735326787136e-08, '440': 1.074007724263553e-06, '441': 7.637388261429709e-07, '442': 5.966709579241961e-07, '443': 6.682714728750996e-07, '444': 3.1265558195227872e-06, '445': 7.327119363309128e-06, '446': 4.53469928022389e-07, '454': 4.773367663393568e-08, '455': 2.148015448527106e-07, '458': 4.773367663393568e-08, '459': 4.773367663393568e-08, '460': 8.353393410938745e-07, '461': 1.193341915848392e-07, '462': 4.272164058737244e-06, '463': 3.98576199893363e-06, '464': 8.40112708757268e-06, '465': 1.3174494750966249e-05, '466': 3.86642780734879e-06, '467': 6.42017950726435e-06, '468': 1.957080741991363e-06, '469': 5.489372812902604e-07, '470': 1.2888092691162634e-06, '471': 2.864020598036141e-07, '473': 1.957080741991363e-06, '474': 6.062176932509832e-06, '475': 2.0286812569422665e-06, '476': 4.296030897054212e-07, '484': 1.193341915848392e-07, '485': 4.773367663393568e-08, '486': 1.193341915848392e-07, '487': 2.386683831696784e-07, '488': 1.9093470653574273e-07, '489': 1.9093470653574273e-07, '490': 1.1933419158483922e-06, '491': 4.296030897054212e-07, '492': 1.909347065357427e-06, '493': 1.2649424307992956e-06, '494': 1.503610813968974e-06, '495': 1.4773572918203094e-05, '496': 3.9618951606166615e-06, '497': 5.823508549340154e-06, '498': 2.3628169933798165e-06, '499': 4.057362513884533e-07, '500': 1.360409784067167e-06, '501': 9.546735326787136e-08, '503': 2.148015448527106e-07, '504': 2.4105506700137523e-06, '505': 1.21720875416536e-06, '506': 3.5800257475451765e-07, '51': 2.6253522148664626e-07, '514': 7.160051495090353e-08, '515': 1.4320102990180706e-07, '517': 2.386683831696784e-08, '52': 1.4320102990180706e-07, '520': 7.637388261429709e-07, '521': 4.53469928022389e-07, '522': 1.3365429457501992e-06, '523': 1.2888092691162634e-06, '524': 1.5274776522859418e-06, '525': 5.059769723197182e-06, '526': 2.0525480952592343e-06, '527': 1.503610813968974e-06, '528': 1.503610813968974e-06, '529': 4.53469928022389e-07, '53': 3.1026889812058193e-07, '530': 2.291216478428913e-06, '531': 4.773367663393568e-08, '532': 2.386683831696784e-08, '533': 7.160051495090353e-08, '534': 7.637388261429709e-07, '535': 4.20056354378634e-06, '54': 9.546735326787136e-08, '544': 2.386683831696784e-08, '545': 2.386683831696784e-08, '546': 2.386683831696784e-08, '547': 2.386683831696784e-08, '548': 2.386683831696784e-08, '550': 4.773367663393568e-08, '551': 3.341357364375498e-07, '552': 1.4081434607011027e-06, '553': 1.050140885946585e-06, '554': 1.6945455205047169e-06, '555': 7.828322967965452e-06, '556': 9.06939856044778e-06, '557': 3.794827292397887e-06, '558': 4.224430382103308e-06, '559': 1.3842766223841349e-06, '560': 2.386683831696784e-07, '561': 2.864020598036141e-07, '562': 1.193341915848392e-07, '563': 5.489372812902604e-07, '564': 1.0978745625805208e-06, '565': 2.386683831696784e-07, '578': 4.773367663393568e-08, '579': 4.773367663393568e-08, '581': 7.160051495090353e-08, '582': 7.160051495090353e-07, '583': 2.386683831696784e-08, '584': 2.6253522148664626e-07, '585': 2.2434828017949772e-06, '586': 2.3628169933798165e-06, '587': 4.844968178344472e-06, '588': 6.754315243701899e-06, '589': 4.152829867152404e-06, '590': 9.06939856044778e-07, '591': 7.160051495090353e-07, '592': 1.2888092691162634e-06, '593': 7.398719878260032e-07, '609': 9.546735326787136e-08, '61': 1.193341915848392e-07, '611': 2.386683831696784e-07, '612': 3.5800257475451765e-07, '613': 9.546735326787136e-08, '614': 6.921383111920674e-07, '615': 8.353393410938745e-07, '616': 3.198156334473691e-06, '617': 5.059769723197182e-06, '618': 3.627759424179112e-06, '619': 1.050140885946585e-05, '620': 3.31749052605853e-06, '621': 1.8377465504065238e-06, '622': 1.0978745625805208e-06, '636': 7.160051495090353e-08, '637': 7.160051495090353e-08, '638': 4.773367663393568e-08, '639': 4.773367663393568e-08, '640': 5.489372812902604e-07, '641': 1.4320102990180706e-07, '642': 1.9093470653574273e-07, '643': 4.773367663393568e-07, '644': 2.0048144186252986e-06, '645': 8.830730177278101e-07, '646': 4.296030897054212e-07, '647': 1.6945455205047169e-06, '648': 2.267349640111945e-06, '649': 2.577618538232527e-06, '650': 1.6945455205047169e-06, '651': 2.148015448527106e-07, '652': 2.386683831696784e-08, '667': 7.160051495090353e-08, '668': 4.773367663393568e-08, '669': 5.250704429732925e-07, '670': 2.386683831696784e-08, '671': 2.6253522148664626e-07, '673': 1.074007724263553e-06, '675': 2.864020598036141e-07, '676': 4.773367663393568e-08, '677': 1.9093470653574273e-07, '678': 2.3866838316967844e-06, '679': 5.107503399831118e-06, '680': 4.391498250322083e-06, '681': 2.386683831696784e-08, '699': 3.5800257475451765e-07, '700': 2.386683831696784e-07, '701': 2.386683831696784e-07, '702': 5.966709579241961e-07, '703': 1.1456082392144565e-06, '704': 1.193341915848392e-07, '705': 2.386683831696784e-08, '707': 4.773367663393568e-08, '708': 1.670678682187749e-07, '709': 4.463098765272986e-06, '710': 8.85459701559507e-06, '711': 1.3842766223841349e-06, '727': 9.546735326787136e-08, '728': 3.5800257475451765e-07, '729': 1.0978745625805208e-06, '730': 2.887887436353109e-06, '731': 1.0978745625805208e-06, '732': 7.876056644599388e-07, '733': 9.546735326787136e-08, '734': 2.386683831696784e-08, '735': 2.386683831696784e-08, '736': 9.546735326787136e-08, '739': 5.584840166170475e-06, '758': 7.637388261429709e-07, '759': 2.864020598036141e-07, '760': 3.8186941307148546e-07, '761': 2.386683831696784e-08, '763': 4.773367663393568e-08, '764': 2.386683831696784e-08, '765': 3.8186941307148546e-07, '766': 7.160051495090353e-08, '767': 1.4320102990180706e-07, '768': 9.546735326787136e-08, '788': 2.386683831696784e-08, '789': 1.4320102990180706e-07, '790': 1.670678682187749e-07, '791': 2.386683831696784e-08, '794': 2.386683831696784e-08, '795': 2.386683831696784e-08, '797': 2.386683831696784e-08, '798': 7.160051495090353e-08, '80': 1.4081434607011027e-06, '81': 4.057362513884533e-07, '818': 2.386683831696784e-08, '819': 2.6253522148664626e-07, '82': 3.1026889812058193e-07, '821': 7.160051495090353e-08, '825': 4.773367663393568e-08, '827': 7.160051495090353e-08, '828': 2.148015448527106e-07, '83': 5.728041196072282e-07, '84': 1.4797439756520063e-06, '848': 4.773367663393568e-08, '849': 2.148015448527106e-07, '855': 7.160051495090353e-08, '856': 2.386683831696784e-08, '857': 1.4320102990180706e-07, '886': 2.386683831696784e-08, '90': 1.193341915848392e-07, '91': 3.341357364375498e-07, '92': 2.386683831696784e-08}

    region_to_cell_dict, cell_to_region_dict, region_centers_dict, region_mean_distance_dict = state_generator.get_regions(
        region_cluster_file, cell_label_dict, number_of_regions=num_regions)

    cell_ids = list(cell_to_region_dict.keys())
    region_ids = list(region_centers_dict.keys())

    processed_rate_dict = dict()
    for cell in cell_ids:
        if cell in raw_rate_dict.keys():
            processed_rate_dict[cell] = raw_rate_dict[cell]
        else:
            processed_rate_dict[cell] = 0.0


    ################
    # Predictor
    predictor = ChainPredictor_Subset(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                      cell_rate_dictionary=processed_rate_dict,
                                      start_time=start_time,
                                      end_time=end_time + (2 * lookahead_horizon_delta_t),
                                      experimental_lookahead_horizon=lookahead_horizon_delta_t,
                                      chain_lookahead_horizon=4.0 * 60 * 60,
                                      check_if_time_diff_constraint_violoated=False,
                                      index_of_exp_chain=index_of_test_chain,
                                      chain_indexes_to_use_for_expierements=indexes_of_sample_chains)

    exp_incident_events = copy.deepcopy(predictor.exp_chain)  # TODO | remove overwrite
    exp_incident_events = [_ for _ in exp_incident_events if _.time < end_time]
    exp_incident_events = [_ for _ in exp_incident_events if _.time >= start_time]
    config_info['predictor'] = 'stationary_predicted_chains - also use as exp incident chain'
    policy_type = 'll_only'
    config_info['exp_id'] = policy_type + 'stationary_r{}_ch{}'.format(num_regions, index_of_test_chain)
    ####################
    # get depots
    depots = state_generator.get_depots(depot_info_path)
    # TODO | remove depots in region 0
    del depots[4]
    del depots[11]

    ######################################
    ######################################
    # Setup decision framework
    cell_travel_model = NashvilleCellToCellTravelModel(cell_to_xy_dict=cell_label_dict,
                                                       travel_rate_cells_per_second=travel_rate_cells_per_second)

    hl_policy = create_high_level_policy(_wait_time_threshold=wait_time_threshold,
                                         _predictor=predictor,
                                         region_centers=region_centers_dict,
                                         _cell_travel_model=cell_travel_model,
                                         region_mean_distance_dict=region_mean_distance_dict,
                                         travel_rate_cells_per_second=travel_rate_cells_per_second,
                                         mean_service_time=mean_service_time,
                                         region_ids=region_ids)

    resp_assignments = get_resp_init_assignments_to_regions_and_depots(hl_policy=hl_policy,
                                                                       region_ids=region_ids,
                                                                       curr_time=start_time,
                                                                       region_to_cell_dict=region_to_cell_dict,
                                                                       total_responders=num_resp,
                                                                       depots=depots,
                                                                       predictor=predictor)

    resp_ids = list(resp_assignments.keys())

    resp_objs = state_generator.get_responders(num_responders=num_resp,
                                               start_time=start_time,
                                               resp_to_region_and_depot_assignments=resp_assignments,
                                               resp_ids=resp_ids,
                                               depots=depots)

    start_state = State(responders=resp_objs,
                        depots=depots,
                        active_incidents=[],
                        time=start_time,
                        cells=cell_to_region_dict,
                        regions=region_to_cell_dict)

    state_copy = copy.deepcopy(start_state)

    #### Low level planner
    dispatch_policy = SendNearestDispatchPolicy(cell_travel_model)
    env_model = EnvironmentModel(cell_travel_model)
    mdp_environment_model = DecisionEnvironmentDynamics(cell_travel_model, SendNearestDispatchPolicy(cell_travel_model))
    rollout_policy = DoNothingRollout()
    llpolicy = LowLevelCentMCTSPolicy(region_ids=region_ids,
                                      incident_prediction_model=predictor,
                                      min_allocation_period=min_time_between_allocation,  # Every half hour
                                      lookahead_horizon_delta_t=lookahead_horizon_delta_t,  # look ahead 2 hours
                                      pool_thread_count=pool_thread_count,
                                      mcts_type=mcts_type,
                                      mcts_discount_factor=mcts_discount_factor,
                                      mdp_environment_model=mdp_environment_model,
                                      rollout_policy=rollout_policy,
                                      uct_tradeoff=uct_tradeoff,
                                      iter_limit=iter_limit,
                                      allowed_computation_time=allowed_computation_time  # 5 seconds per thread
                                      )

    ########
    ## set up failure events
    failure_time = datetime(year=2019, month=1, day=1, hour=0, minute=1).timestamp()

    region_incident_rates = dict()
    for r in region_ids:
        region_incident_rates[r] = hl_policy.get_region_total_rate(start_time, region_to_cell_dict[r])

    sorted_regions_by_rate = sorted([_ for _ in region_incident_rates.items()], key=lambda _:_[1], reverse=True)

    # region_to_fail = sorted_regions_by_rate[1][0]
    region_to_fail = sorted_regions_by_rate[4][0]
    num_resp_to_fail = 2

    test_fail_event = Event(event_type=EventType.FAILURE,
                            cell_loc=None,
                            time=failure_time,
                            type_specific_information={'region': region_to_fail, 'num_resp_to_fail': num_resp_to_fail})

    print(test_fail_event.__dict__)

    # exp_incident_events.sort(key=lambda _: _.time)
    exp_incident_events.append(test_fail_event)
    exp_incident_events.sort(key=lambda _: _.time)

    # sys.exit()

    coord = 'hl_ll'
    decision_coordinator = HighAndLowLevelCoord_Fail(environment_model=env_model,
                                                     travel_model=cell_travel_model,
                                                     dispatch_policy=dispatch_policy,
                                                     min_time_between_allocation=min_time_between_allocation,
                                                     low_level_policy=llpolicy,
                                                     high_level_policy=hl_policy)

    # coord = 'll_only'
    # decision_coordinator = LowLevelCoord_Fail(environment_model=env_model,
    #                                                  travel_model=cell_travel_model,
    #                                                  dispatch_policy=dispatch_policy,
    #                                                  min_time_between_allocation=min_time_between_allocation,
    #                                                  low_level_policy=llpolicy,
    #                                                  )

    config_info['exp_id'] = '__coord_{}__numFail_{}__'.format(coord, num_resp_to_fail) + config_info['exp_id']
    #################



    print('number of exp incidents: {}'.format(len(exp_incident_events)))

    simulator = Simulator(starting_event_queue=copy.deepcopy(exp_incident_events),
                          starting_state=state_copy,
                          environment_model=env_model,
                          event_processing_callback=decision_coordinator.event_processing_callback_funct)


    simulator.run_simulation()

    print('====== Done ======')
    print(decision_coordinator.metrics)
    print('avg resp_time: ', np.mean(list(decision_coordinator.metrics['resp_times'].values())))

    '''
    Expected output: 
    {'resp_times': {829: 0.0, 830: 120.0, 831: 169.70562744140625, 832: 379.4733192920685, 833: 120.0, 834: 0.0, 835: 0.0, 836: 120.0, 837: 120.0, 838: 240.0, 839: 169.70562744140625, 840: 169.70562744140625, 841: 120.0, 842: 120.0, 843: 339.4112548828125, 844: 0.0, 845: 0.0, 846: 120.0, 8736: 240.0, 8737: 120.0, 8738: 0.0}}
    avg resp_time:  127.04768840471904
    
    '''