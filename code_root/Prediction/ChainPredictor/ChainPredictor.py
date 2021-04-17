import pickle

class ChainPredictor:

    def __init__(self,
                 preprocessed_chain_file_path,
                 cell_rate_dictionary,
                 max_time_diff_on_lookup=None,
                 check_if_time_diff_constraint_violoated=True):

        self.cell_rate_dictionary = cell_rate_dictionary
        self.check_if_time_diff_constraint_violoated = check_if_time_diff_constraint_violoated
        if max_time_diff_on_lookup is None:
            max_time_diff_on_lookup = 30 * 60  # 30 minutes

        self.max_time_diff_on_lookup = max_time_diff_on_lookup
        self.precomputed_chains = pickle.load(open(preprocessed_chain_file_path, 'rb'))


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

        return closest_match[1]


    def get_cell_rate(self, curr_time, cell_id):
        # ignore time for now
        return self.cell_rate_dictionary[cell_id]

