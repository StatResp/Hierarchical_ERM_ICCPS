import random
import copy

from Environment.DataStructures.Incident import Incident
from Environment.DataStructures.Event import Event
# from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import

from Environment.enums import EventType

class TESTIncidentPredictor:
    '''
    An incident predictor class needs functions for
    - getting incident chains at a given time
    '''

    def __init__(self, num_chains):
        '''
        Have default set of chains, starting at time 0? Or simply one set of chains.
        '''

        self.num_chains = num_chains
        self.chains = list()

        # this is based on original grid world example
        possible_xs = range(7)
        possible_ys = range(7)
        time_dif_range = range(200, 1200)
        clearance_time_range = range(300, 800)


        # have 5 chains, each with 100 incidents at random points
        for i in range(num_chains):
            chain = list()

            curr_time = 0
            for incident_num in range(100):
                x = random.choice(possible_xs)
                y = random.choice(possible_ys)
                i_time = curr_time + random.choice(time_dif_range)
                curr_time = i_time
                clearance_time = random.choice(clearance_time_range)

                incident = Incident(cell_loc=(x, y),
                                    time=i_time,
                                    clearance_time=clearance_time)

                event = Event(event_type=EventType.INCIDENT,
                              cell_loc=(x, y),
                              time=i_time,
                              # event_id=ids[i],
                              type_specific_information={'incident_obj': incident})

                chain.append(event)
            self.chains.append(chain)

        print(self.chains)



    def get_chains(self, time):
        '''
        Returns a set of chains for the given time.
        TODO | question: should it expect the exact time, or just return the nearest or next?
        TODO | or should it be based on ti me intervals or something like that?
        FOR NOW == assume that some exact time is given, we work in intervals
        :param time:
        :return:
        '''

        return copy.deepcopy(self.chains)



class WeightedTestPredictor:
    '''
    An incident predictor class needs functions for
    - getting incident chains at a given time
    '''

    def __init__(self,
                 num_chains,
                 start_time,
                 coords,
                 num_incidents,
                 time_difs,
                 clearance_times,
                 time_difs_weights=None,
                 coords_weights=None,
                 clearance_times_weights=None):
        '''
        Have default set of chains, starting at time 0? Or simply one set of chains.
        '''

        self.num_chains = num_chains
        self.chains = list()

        for i in range(num_chains):
            chain = list()

            curr_time = start_time

            sampled_coords = random.choices(coords, weights=coords_weights, k=num_incidents)
            sampled_time_difs = random.choices(time_difs, weights=time_difs_weights, k=num_incidents)
            sampled_clearence_times = random.choices(clearance_times, weights=clearance_times_weights, k=num_incidents)


            for incident_num in range(num_incidents):
                i_time = curr_time + sampled_time_difs[incident_num]
                curr_time = i_time

                incident = Incident(cell_loc=sampled_coords[incident_num],
                                    time=i_time,
                                    clearance_time=sampled_clearence_times[incident_num])

                event = Event(event_type=EventType.INCIDENT,
                              cell_loc=sampled_coords[incident_num],
                              time=i_time,
                              # event_id=ids[i],
                              type_specific_information={'incident_obj': incident})

                chain.append(event)
            self.chains.append(chain)

        print(self.chains)



    def get_chains(self, time):
        '''
        Returns a set of chains for the given time.
        TODO | question: should it expect the exact time, or just return the nearest or next?
        TODO | or should it be based on ti me intervals or something like that?
        FOR NOW == assume that some exact time is given, we work in intervals
        :param time:
        :return:
        '''

        return copy.deepcopy(self.chains)

