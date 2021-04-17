

import pickle
from datetime import datetime, timedelta
from copy import copy
from math import floor

from Environment.DataStructures.Event import Event
from Environment.DataStructures.Incident import Incident

from Environment.enums import EventType



# def getXYForGrid(gridNum):
#     x = int(floor(gridNum / 30))
#     y = gridNum - (x * 30)
#     return x, y

month = 1
start_time = datetime(year=2019, month=1, day=1)
end_time = datetime(year=2019, month=1, day=12)
interval = timedelta(minutes=30)

# TODO make this configurable and possibly drawn from a distribution
MEAN_CLEARANCE_TIME = 20 * 60  # 20 minutes


curr_time = copy(start_time)

all_chains = list()

curr_number = 0

max_time_diff = 0

rate_dict = pickle.load(open('../../../data/chain_data/chains_non_stationary/rates_non_s.pickle', 'rb'))

processed_rate_dict = dict()


while curr_time < end_time:

    # process rate dict correctly
    curr_rate_dict = rate_dict[curr_number]
    processed_rate_dict[curr_time.timestamp()] = dict()
    for key, val in curr_rate_dict.items():
        processed_rate_dict[curr_time.timestamp()][str(key)] = float(val) / 60.0  # minutes -> seconds




    sampled_chains = pickle.load(
        # open('../../Data/IncidentChainsTDOT/python3/' + str(month) + '/predictions_' + str(curr_number) + '_p3.pickle', 'rb'))
        # open('../../../data/chain_data/' + str(month) + '/predictions_' + str(curr_number) + '.pickle',
        #  'rb'))
        open('../../../data/chain_data/chains_non_stationary' + '/predictions_' + str(curr_number) + '.pickle',
             'rb'))

    print('sampled_chains')
    # 5 chains

    processed_incident_chains = list()
    for raw_chain in sampled_chains:
        processed_chain = list()
        for incident_info in raw_chain:
            incident_time = incident_info[1].timestamp()

            incident_obj = Incident(cell_loc=str(incident_info[0]),
                                    time = incident_time,
                                    clearance_time=MEAN_CLEARANCE_TIME)

            event_obj = Event(event_type=EventType.INCIDENT,
                              cell_loc=str(incident_info[0]),
                              time=incident_time,
                              type_specific_information={'incident_obj': incident_obj})

            processed_chain.append(event_obj)

        processed_chain.sort(key=lambda _:_.time)

        if len(processed_chain) > 0:
            time_diff = processed_chain[-1].time - curr_time.timestamp()
            if time_diff > max_time_diff:
                max_time_diff = time_diff

        processed_incident_chains.append(processed_chain)

        # processed_incident_chains.append(
        #     [Event(EventType.INCIDENT, getXYForGrid(_[0]), _[1]) for _ in raw_chain])

    # chain_dict[str(curr_time.year) + '_' + str(curr_time.month) + '_' + str(curr_time.day) + '_' + str(curr_time.hour) + '_' + str(curr_time.minute)] = processed_incident_chains

    # index chain dictionary via epoch time? Or should it be a list, and you choose the chains that are closest?
    # go with second option:
    all_chains.append((curr_time.timestamp(), processed_incident_chains))

    curr_time = curr_time + interval
    curr_number += 1

print(max_time_diff)
print(all_chains[-1][0])

print(len(all_chains[0][1]))

pickle.dump(all_chains, open('../../../data/chain_data/non_stationary_chains_60_v1.pickle', 'wb'))
pickle.dump(processed_rate_dict, open('../../../data/chain_data/non_stationary_rates.pkl', 'wb'))

# [(time, [chain1, chain2])]