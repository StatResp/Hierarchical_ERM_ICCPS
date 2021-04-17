import pickle
import numpy as np

class ChainPredictorReturnOne:

    def __init__(self,
                 preprocessed_chain_file_path,
                 cell_rate_dictionary,
                 start_time,
                 end_time,
                 max_time_diff_on_lookup=None,
                 check_if_time_diff_constraint_violoated=True,
                 total_chains=5,
                 chain_lookahead_horizon=None):

        self.cell_rate_dictionary = cell_rate_dictionary
        self.check_if_time_diff_constraint_violoated = check_if_time_diff_constraint_violoated
        if max_time_diff_on_lookup is None:
            max_time_diff_on_lookup = 30 * 60  # 30 minutes

        self.max_time_diff_on_lookup = max_time_diff_on_lookup
        self.precomputed_chains = pickle.load(open(preprocessed_chain_file_path, 'rb'))
        self.precomputed_chains.sort(key=lambda _: _[0])

        self.number_of_prediction_chains = total_chains - 1
        plus_one_chain_index = total_chains - 1
        self.plus_one_chain = list()

        if chain_lookahead_horizon is None:
            chain_lookahead_horizon = 4.0 * 60 * 60

        # append chains together to get 1 long chain
        _start_chain_time = closest_match = min(self.precomputed_chains, key=lambda _: abs(_[0] - start_time))[0]
        _end_chain_time = min(self.precomputed_chains, key=lambda _: abs(_[0] - end_time))[0]

        _difs_per_chain = list()

        _curr_chain_time = _start_chain_time
        while _curr_chain_time <= _end_chain_time:
            closest_match = min(self.precomputed_chains, key=lambda _: abs(_[0] - _curr_chain_time))

            _chain = closest_match[1][plus_one_chain_index]

            _chain_times = [_.time for _ in _chain]

            _diffs = np.diff(_chain_times)

            # avg_diff = np.mean(_diffs)
            _difs_per_chain.extend(_diffs)

            self.plus_one_chain.extend(_chain)

            _curr_chain_time += chain_lookahead_horizon

        self.avg_incident_diffs = np.mean(_difs_per_chain)

        self.full_plus_one_diffs = np.mean(np.diff([_.time for _ in self.plus_one_chain]))



    def get_chains(self, curr_time):
        '''
        Note: assumes 'time' is epoch time (seconds)
        :param curr_time:
        :return:
        '''

        closest_match = min(self.precomputed_chains, key=lambda _: abs(_[0] - curr_time))
        if self.check_if_time_diff_constraint_violoated:
            time_diff = abs(closest_match[0] - curr_time)
            if time_diff > self.max_time_diff_on_lookup:
                raise Exception('prediction time match constraint delta violated: curr_time={}, match_time={}'.format(curr_time, closest_match[0]))

        return closest_match[1][:self.number_of_prediction_chains]


    def get_cell_rate(self, curr_time, cell_id):
        # ignore time for now
        return self.cell_rate_dictionary[cell_id]

