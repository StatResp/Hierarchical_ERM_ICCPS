from datetime import datetime

import pandas as pd
import numpy as np
import pprint
import json
import pickle
import sys
from sklearn.cluster import KMeans

from matplotlib import colors
import matplotlib.pyplot as plt

from Environment.DataStructures.Responder import Responder
from Environment.DataStructures.State import State
from Environment.DataStructures.Depot import Depot

from Environment.DataStructures.Event import Event
from Environment.DataStructures.Incident import Incident

from Environment.Spatial.spatialStructure import  get_cell_id_and_rates
from Environment.enums import EventType, RespStatus

from Environment.Spatial.spatialStructure import get_mean_distance


class NashvilleStateGenerator:

    def __init__(self,
                 num_rows,
                 num_columns,
                 structuraly_valid_cells):
        self.structuraly_valid_cells = structuraly_valid_cells
        self.num_columns = num_columns
        self.num_rows = num_rows


    # DavidsonMVA_ETrims_Jan2018_April2019.json
    def get_incident_df(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data, orient='columns')

        df['cell_string_id'] = df.apply(lambda row: str(row.cell[1] * self.num_rows + row.cell[0]), axis=1)

        return df

    def get_responders(self, num_responders, start_time, resp_to_region_and_depot_assignments, resp_ids, depots):

        '''

        :param num_responders:
        :param start_time:
        :return:
        '''

        '''
        assume that all responders start with...
            - available
            - waiting status
            - located at their starting depot
            - t_state_change = start_time
            - avaialble_time = start-time
        '''

        responder_objs = dict()  # {resp_id: obj}

        for resp_id in resp_ids:

            assigned_depot = resp_to_region_and_depot_assignments[resp_id]['depot_id']
            assigned_region = resp_to_region_and_depot_assignments[resp_id]['region_id']

            resp = Responder(cell_loc=depots[assigned_depot].cell_loc,
                             status=RespStatus.WAITING,
                             available=True,
                             dest=None,
                             t_state_change=start_time,
                             region_assignment=assigned_region,
                             assigned_depot_id=assigned_depot,
                             my_id=resp_id,
                             incident=None,
                             available_time=start_time,
                             total_distance_moved=0,
                             total_depot_movement_distance=0,
                             total_incident_distance=0)

            responder_objs[resp_id] = resp

        return responder_objs

    def get_depots(self, file_path):
        depot_df = pd.read_json(open(file_path, 'rb'), orient='records')

        depot_df['cell'] = depot_df['cell'].apply(lambda _: tuple(_))
        depot_df['cell_string_id'] = depot_df.apply(lambda row: str(row.cell[1] * self.num_rows + row.cell[0]), axis=1)

        depot_cells = depot_df['cell_string_id'].unique()

        depot_objs = dict()

        for index, depot_cell in enumerate(depot_cells):

            # assume now capacity needed for now
            # TODO | get assigned responders
            depot_obj = Depot(cell_loc=depot_cell,
                              assigned_responders=list(),
                              capacity=None,
                              my_id=index)

            depot_objs[index] = depot_obj

        return depot_objs

    def get_depots_with_centers(self, file_path, cell_centers_dict):
        depot_df = pd.read_json(open(file_path, 'rb'), orient='records')

        depot_df['cell'] = depot_df['cell'].apply(lambda _: tuple(_))
        depot_df['cell_string_id'] = depot_df.apply(lambda row: str(row.cell[1] * self.num_rows + row.cell[0]), axis=1)

        depot_cells = depot_df['cell_string_id'].unique()

        depot_objs = dict()

        for index, depot_cell in enumerate(depot_cells):

            _y, _x = cell_centers_dict[depot_cell]


            # assume now capacity needed for now
            # TODO | get assigned responders
            depot_obj = Depot(cell_loc=depot_cell,
                              assigned_responders=list(),
                              capacity=None,
                              my_id=index,
                              lon=_x,
                              lat=_y)

            depot_objs[index] = depot_obj

        return depot_objs


    # # need depots, responders, time, cells, and region generation
    # def generate_start_state(self,
    #                          start_time,
    #                          incident_df_file):
    #
    #     incident_df = self.get_incident_df(incident_df_file)


    def get_regions(self,
                    region_cluster_file,
                    cell_id_to_xy_coords_dict,
                    number_of_regions):

        # load raw labels
        labels, cluster_obj = pickle.load(open(region_cluster_file, 'rb'))

        # region to cell dict - {'region_id': [cells], ...}
        region_to_cell_dict = dict()
        # region_ids = list(set(_ for _ in labels.values()))

        # TODO | this is specific to Ayan's implementation - he set invalid cells to <num_regions>, i.e. one more than last valid region
        # TODO | therefore I am assuming 5 regions
        # region_ids = [0, 1, 2, 3, 4]  # 5 regions
        region_ids = range(number_of_regions)    # 6 regions

        for region_id in region_ids:
            region_cells = [_[0] for _ in labels.items() if _[1] == region_id]
            region_to_cell_dict[str(region_id)] = region_cells

        print(region_to_cell_dict)

        cell_to_region_dict = dict()  # {cell_id: region_id, ...}
        cell_ids = set([_ for _ in labels.keys()])
        for cell_id in cell_ids:
            cell_region = labels[cell_id]
            if cell_region in region_ids:
                cell_to_region_dict[cell_id] = str(cell_region)

        print(cell_to_region_dict)

        raw_region_centers = cluster_obj.cluster_centers_

        print(raw_region_centers)

        processed_centers = dict()
        for index, region_center in enumerate(raw_region_centers):
            region_id = str(index)
            region_x = int(round(region_center[0]))
            region_y = int(round(region_center[1]))
            region_center_cell_id = str((region_y * self.num_rows) + region_x)

            processed_centers[region_id] = region_center_cell_id


        # mean distances for each region
        raw_dist_region_dict = get_mean_distance(labels, cluster_obj, cell_id_to_xy_coords_dict)
        processed_dist_region_dict = dict()
        for region_int, mean_distance in raw_dist_region_dict.items():
            processed_dist_region_dict[str(region_int)] = mean_distance


        print('region gen done')


        return region_to_cell_dict, cell_to_region_dict, processed_centers, processed_dist_region_dict

    # **** This is specific to this label impl
    def get_cell_label_to_x_y_coord_dict(self):

        cell_label_to_coord_dict = dict()

        for y in range(self.num_rows):
            for x in range(self.num_columns):

                cell_label_to_coord_dict[str(((y * self.num_rows) + x))] = (x, y)

        return cell_label_to_coord_dict


    def visualize_regions(self, cell_to_region_dict, cell_labels_to_xy_coords,
                          depots=None, cell_id_to_coords=None, num_regions=None):

        grid_labels = np.full(shape=[30, 30], fill_value=num_regions)

        for cell_id, region_id in cell_to_region_dict.items():
            xy_coord_tupple = cell_labels_to_xy_coords[cell_id]
            grid_labels[xy_coord_tupple[1], xy_coord_tupple[0]] = region_id

        cmap = colors.ListedColormap(['red', 'blue', 'green', 'yellow', 'black', 'purple', 'white', 'teal', 'orange', 'pink'])
        bounds = range(num_regions + 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        # ax.imshow(grid_labels, cmap=cmap, norm=norm, origin='lower')
        ax.imshow(grid_labels, cmap=cmap, origin='lower')


        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

        if depots is not None:
            xs = []
            ys = []
            for depot in depots.values():
                cell_id = depot.cell_loc
                coord = cell_id_to_coords[cell_id]
                xs.append(coord[0])
                ys.append(coord[1])
            ax.scatter(xs, ys, c='grey')

        [ax.scatter([], [], c=cmap(i), label=str(i)) for i in range(num_regions)]
        ax.legend()

        plt.show()
        print('Plotted Regions')

    def get_expieremental_incident_events(self,
                                          incident_df,
                                          start_time_epoch,
                                          end_time_epoch,
                                          mean_clearance_time,
                                          valid_cells):  # assume static clearance time for now
        filtered_df = incident_df.copy(deep=True)

        filtered_df['datetime'] = filtered_df['datetime'] / 1000
        filtered_df = filtered_df[filtered_df['datetime'] < end_time_epoch]
        filtered_df = filtered_df[filtered_df['datetime'] > start_time_epoch]
        filtered_df = filtered_df[filtered_df['cell_string_id'].isin(valid_cells)]

        exp_incident_events = list()
        for index, row in filtered_df.iterrows():

            incident_obj = Incident(cell_loc=row['cell_string_id'],
                                    time = row['datetime'],
                                    clearance_time=mean_clearance_time)

            event_obj = Event(event_type=EventType.INCIDENT,
                              cell_loc=row['cell_string_id'],
                              time=row['datetime'],
                              type_specific_information={'incident_obj': incident_obj})

            exp_incident_events.append(event_obj)

        exp_incident_events.sort(key= lambda _: _.time)

        return exp_incident_events



if __name__ == "__main__":

    num_rows = 30
    num_columns = 30
    num_resp = 26
    data_directory = '../../../../data/'
    incident_df_path = data_directory + 'DavidsonMVA_ETrims_Jan2018_April2019.json'
    depot_info_path = data_directory +  'testing/depots_with_cells.json'
    region_cluster_file = data_directory + 'cluster_output_5_r.pk'
    num_regions = 5
    structural_valid_cells = pickle.load(open(data_directory + 'valid_cells_out.pkl', 'rb'))
    structural_valid_cells = [str(_) for _ in structural_valid_cells]

    state_generator = NashvilleStateGenerator(num_rows=num_rows, num_columns=num_columns, structuraly_valid_cells=structural_valid_cells)

    cell_label_dict = state_generator.get_cell_label_to_x_y_coord_dict()

    region_to_cell_dict, cell_to_region_dict, region_centers_dict, region_mean_distance_dict = state_generator.get_regions(region_cluster_file, cell_label_dict, num_regions)





    print(cell_label_dict)

    depots = state_generator.get_depots(depot_info_path)

    # TODO | remove depots in region 0
    del depots[4]
    del depots[11]

    # num_regions = len(list(region_to_cell_dict.keys()))

    # state_generator.visualize_regions(cell_to_region_dict=cell_to_region_dict,
    #                                   cell_labels_to_xy_coords=cell_label_dict,
    #                                   depots=depots,
    #                                   cell_id_to_coords=cell_label_dict,
    #                                   num_regions=num_regions)

    # sys.exit()



    ######################################
    # testing starting region allocation

    from Prediction.ChainPredictor.ChainPredictor import ChainPredictor

    incident_df = state_generator.get_incident_df(incident_df_path)

    # TODO | rates should only include incidents from 'training' data period? Incorrect impl
    valid_cell_ids, raw_rate_dict = get_cell_id_and_rates(incident_df, structural_valid_cells)

    processed_rate_dict = dict()
    for cell in cell_to_region_dict.keys():
        if cell in raw_rate_dict.keys():
            processed_rate_dict[cell] = raw_rate_dict[cell]
        else:
            processed_rate_dict[cell] = 0.0


    # state_generator.visualize_regions(cell_to_region_dict={_[0]: _[1] for _ in cell_to_region_dict.items() if _[0] in valid_cell_ids},
    #                                   cell_labels_to_xy_coords=cell_label_dict)

    preprocessed_chains_file_path = data_directory + 'chain_data/oct_6_2020_macros_processed_chains.pickle'
    chain_predictor = ChainPredictor(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                     cell_rate_dictionary=processed_rate_dict,
                                     max_time_diff_on_lookup=31*60,
                                     check_if_time_diff_constraint_violoated=True)

    from scenarios.Nashville_1.NashvilleCellToCellTravelModel import NashvilleCellToCellTravelModel

    travel_rate_cells_per_second = 60.0 / 3600.0  # cells are 1x1 miles; 60 mph
    cell_travel_model = NashvilleCellToCellTravelModel(cell_to_xy_dict=cell_label_dict,
                                                       travel_rate_cells_per_second=travel_rate_cells_per_second)

    # test_interpolate = cell_travel_model.interpolate_distance(valid_cell_ids[0], valid_cell_ids[100], 0.5)


    # get expieremental incidents - jan 2019
    start_time = datetime(year=2019, month=1, day=1).timestamp()
    end_time = datetime(year=2019, month=1, day=11).timestamp()

    # exp_incident_events = state_generator.get_expieremental_incident_events(incident_df,
    #                                                                         start_time,
    #                                                                         end_time,
    #                                                                         mean_clearance_time=20 * 60)


    from scenarios.Nashville_1.BetweenRegionTravelModel import BetweenRegionTravelModel
    between_region_travel_model = BetweenRegionTravelModel(region_centers=region_centers_dict,
                                                           cell_travel_model=cell_travel_model)


    from scenarios.Nashville_1.WithinRegionTravelModel import NashvilleWithinRegionTravelModel
    within_region_travel_model = NashvilleWithinRegionTravelModel(mean_dist_for_region_dict=region_mean_distance_dict,
                                                                  travel_cells_per_second=travel_rate_cells_per_second)

    from decision_making.HighLevel.MMCHighLevelPolicy import MMCHighLevelPolicy

    class test_incident_service_time_model:
        def __init__(self, region_ids, mean_service_time):
            self.model = dict()
            for id in region_ids:
                self.model[id] = mean_service_time  # mean service time (NOT RATE) in seconds - 20 minutes?

        def get_expected_service_time(self, region_id, curr_time):

            # ignore time for now
            return self.model[region_id]

    service_time_model = test_incident_service_time_model(region_ids=list(region_centers_dict.keys()),
                                                          mean_service_time=20 * 60)

    wait_time_threshold = 10 * 60  # 10 minutes
    hl_policy = MMCHighLevelPolicy(prediction_model=chain_predictor,
                                   within_region_travel_model=within_region_travel_model,
                                   trigger_wait_time_threshold=wait_time_threshold,
                                   incident_service_time_model=service_time_model,
                                   between_region_travel_model=between_region_travel_model)

    ###############################
    ## Testing
    region_ids = list(region_centers_dict.keys())

    region_incident_rates = dict()
    for r in region_ids:
        region_incident_rates[r] = hl_policy.get_region_total_rate(None, region_to_cell_dict[r])

    region_depot_counts = dict()
    for region_id in region_ids:
        region_depot_counts[region_id] = len([_[0] for _ in depots.items() if
                                              _[1].cell_loc in region_to_cell_dict[region_id]])

    initial_resp_to_region_allocation = hl_policy.greedy_responder_allocation_algorithm(region_ids=list(region_centers_dict.keys()),
                                                                                        region_incident_rates=region_incident_rates,
                                                                                        total_responders=26,
                                                                                        curr_time=None,
                                                                                        region_to_depot_count=region_depot_counts)

    region_to_depots = dict()
    for region_id in region_ids:
        region_to_depots[region_id] = {_[0]: _[1] for _ in depots.items() if _[1].cell_loc in region_to_cell_dict[region_id]}

    print(region_to_depots)

    depot_cell_rates = dict()
    for region_id in region_ids:
        region_depot_rates = list()
        for region_depot in region_to_depots[region_id].values():
            depot_cell_rate = chain_predictor.get_cell_rate(None, region_depot.cell_loc)
            region_depot_rates.append((region_depot.my_id, depot_cell_rate))

        region_depot_rates.sort(key=lambda _: _[1], reverse=True)
        depot_cell_rates[region_id] = region_depot_rates

    # for depot in depots.values():
    #     region_for_depot = next([_[0] for _ in region_to_depots.items() if depot in _[1]])
    #     depot_cell_rate = chain_predictor.get_cell_rate(None, depot.cell_loc)
    #     depot_cell_rates. = depot_cell_rate


    # assign resp to depots with highest rates in their regions
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

    resp_objs = state_generator.get_responders(num_responders=num_resp,
                                               start_time=start_time,
                                               resp_to_region_and_depot_assignments=resp_assignments,
                                               resp_ids=resp_ids,
                                               depots=depots)

    print('done')








