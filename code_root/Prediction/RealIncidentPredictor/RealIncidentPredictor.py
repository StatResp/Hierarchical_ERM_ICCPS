

class RealIncidentPredictor:

    def __init__(self, real_incident_events, cell_rate_dictionary):
        self.cell_rate_dictionary = cell_rate_dictionary
        self.real_incident_events = real_incident_events


    def get_chains(self, curr_time):
        filtered_queue = [_ for
                          _ in self.real_incident_events
                          if _.time > curr_time]

        chains = list()
        chains.append(filtered_queue)

        return chains

    def get_cell_rate(self, curr_time, cell_id):
        # ignore time for now
        return self.cell_rate_dictionary[cell_id]
