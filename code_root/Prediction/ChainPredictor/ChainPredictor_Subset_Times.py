import numpy as np
import pickle


class ChainPredictor_Subset_Times:

    def __init__(self,
                 preprocessed_chain_file_path,
                 cell_rate_dictionary,
                 start_time,
                 end_time,
                 experimental_lookahead_horizon,
                 chain_lookahead_horizon,
                 index_of_exp_chain,
                 chain_indexes_to_use_for_expierements,
                 max_time_diff_on_lookup=None,
                 check_if_time_diff_constraint_violoated=True,
                 ):

        self.chain_lookahead_horizon = chain_lookahead_horizon
        self.experimental_lookahead_horizon = experimental_lookahead_horizon
        self.cell_rate_dictionary = cell_rate_dictionary
        self.check_if_time_diff_constraint_violoated = check_if_time_diff_constraint_violoated
        if max_time_diff_on_lookup is None:
            max_time_diff_on_lookup = 30 * 60  # 30 minutes

        self.max_time_diff_on_lookup = max_time_diff_on_lookup
        _precomputed_chains = pickle.load(open(preprocessed_chain_file_path, 'rb'))
        _precomputed_chains.sort(key=lambda _: _[0])


        # construct the leave-one-out incident chain
        # TODO | might want to set this up so that the
        self.number_of_prediction_chains = len(chain_indexes_to_use_for_expierements)
        self.exp_chain = list()

        _difs_per_chain = list()

        _curr_chain_time = _precomputed_chains[0][0]
        _end_time = _precomputed_chains[-1][0]
        while _curr_chain_time <= _end_time:
            closest_match = min(_precomputed_chains, key=lambda _: abs(_[0] - _curr_chain_time))

            _chain = closest_match[1][index_of_exp_chain]

            _chain_times = [_.time for _ in _chain]

            _diffs = np.diff(_chain_times)

            # avg_diff = np.mean(_diffs)
            _difs_per_chain.extend(_diffs)

            self.exp_chain.extend(_chain)

            _curr_chain_time += chain_lookahead_horizon

        self.precomputed_chains = list()
        for chain in _precomputed_chains:
            filtered_chains = list()
            for index in chain_indexes_to_use_for_expierements:
                filtered_chains.append(chain[1][index])

            self.precomputed_chains.append((chain[0], filtered_chains))

        # compute nearest_rate dictionary
        # TODO
        # _curr_procssing_time = self.precomputed_chains[0][0]
        _curr_procssing_time = start_time - (2 *chain_lookahead_horizon)
        _end_processing_time = self.precomputed_chains[-1][0]

        #
        print('time dict construction')
        print('start time: {}, curr_processing_start: {}'.format(start_time, _curr_procssing_time))
        print('end_time: {}, end_processing_time: {}'.format(end_time, _end_processing_time))

        self.nearest_time_key_lookup_dict = dict()
        while _curr_procssing_time <= _end_processing_time + (2 *chain_lookahead_horizon):


            nearest_time = min(self.cell_rate_dictionary.keys(), key=lambda _: abs(_ - _curr_procssing_time))
            self.nearest_time_key_lookup_dict[_curr_procssing_time] =  nearest_time

            _curr_procssing_time += 1

            # chain[1].pop(index_of_leave_one_out)

        self.avg_incident_diffs = np.mean(_difs_per_chain)

        # test = sorted(self.plus_one_chain, key=lambda _: _.time)
        #
        # if test != self.plus_one_chain:
        #     print('oops')

        self.exp_chain.sort(key=lambda _: _.time)

        self.full_plus_one_diffs = np.mean(np.diff([_.time for _ in self.exp_chain]))

        self.exp_chain = [_ for _ in self.exp_chain if _.time >= start_time]
        self.exp_chain = [_ for _ in self.exp_chain if _.time <= end_time]


    def get_chains(self, curr_time):
        '''
        Note: assumes 'time' is epoch time (seconds)
        :param curr_time:
        :return:
        '''

        # might have to append chains...
        '''        
        get the first chain time
        curr time = first chain time
        full chains = list()
        full chains.append([]) for num chains
        extract while curr_time < exp look ahead horizon
            chains = closest chain to time[1]
            for index, chain in enumerate(chains): 
                full_chains[index].extend(chain)
            
            
            curr_time += chain time horizon
        '''

        _chains_curr_time = min(self.precomputed_chains, key=lambda _: abs(_[0] - curr_time))[0]
        extracted_chains = list()
        for i in range(self.number_of_prediction_chains):
            extracted_chains.append(list())

        while _chains_curr_time <= (curr_time + self.experimental_lookahead_horizon):
            chains = min(self.precomputed_chains, key=lambda _: abs(_[0] - _chains_curr_time))[1]
            for index, chain in enumerate(chains):
                extracted_chains[index].extend(chain)

            _chains_curr_time += self.chain_lookahead_horizon

        for chain in extracted_chains:
            chain.sort(key=lambda _: _.time)

        return extracted_chains


    def get_cell_rate(self, curr_time, cell_id):
        # ignore time for now
        try:
            dict_closest_time = self.nearest_time_key_lookup_dict[curr_time]  # TODO | should this be handeled somewhere else?
        except:
            print('error')
            print(curr_time)
            # print(self.nearest_time_key_lookup_dict)
            raise(Exception)
        # print('\t\tcurr: {}'.format(curr_time))
        # print('\t\tclosest: {}'.format(dict_closest_time))
        # print(self.cell_rate_dictionary[dict_closest_time][cell_id])
        return self.cell_rate_dictionary[dict_closest_time][cell_id]