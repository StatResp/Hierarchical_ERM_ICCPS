import pickle
import time
import math

from Environment.enums import EventType


class Simulator:
    '''
    This class simulates time ocurring in the incident response domain
    '''

    def __init__(self,
                 starting_state,
                 environment_model,
                 event_processing_callback,
                 starting_event_queue):

        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback

        self.event_queue = starting_event_queue

        self.sim_metrics = dict()
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        self.start_sim_time = None


    def run_simulation(self):
        '''
        runs the given simulation. will run until the event queue is empty
        :return:
        '''

        print('running simulation')
        print()

        self.start_sim_time = time.time()

        while len(self.event_queue) > 0:

            self.update_sim_info()
            curr_event = self.event_queue.pop(0)

            self.print_before_iter(curr_event)

            self.environment_model.update(self.state, curr_event.time)

            next_event = None
            if len(self.event_queue) > 0:
                next_event = self.event_queue[0]

            new_events = self.event_processing_callback(self.state, curr_event, next_event)
            for event in new_events:
                self.add_event(event)

            self.print_after_iter()


        print()
        print('completed_simulation')

    def print_after_iter(self):

        # determine event progress
        estimated_num_events = self.num_events_processed + len(self.event_queue) - 1
        curr_progress_fraction = float(self.num_events_processed) / float(estimated_num_events)
        curr_progress_percent = math.trunc(curr_progress_fraction * 100.0)

        # estimate remaining compu time
        time_taken = time.time() - self.start_sim_time
        estimated_total_time = float(time_taken) / float(curr_progress_fraction)
        estimated_time_remaining = estimated_total_time - time_taken

        print('Iter {} complete. Progress: ~{}% - {} seconds left ({} / {} s complete)'.format(self.num_events_processed,
                                                                                             curr_progress_percent,
                                                                                             estimated_time_remaining,
                                                                                             time_taken,
                                                                                             estimated_total_time))

    def print_before_iter(self, event):

        estimated_num_events = self.num_events_processed + len(self.event_queue) - 1
        print('----')
        print('Starting Iter: {} / ~{} | at exp time: {}'.format(self.num_events_processed,
                                               estimated_num_events,
                                                                 event.time))
        print('\tevent info | type: {}'.format(EventType(event.event_type).name))



    def update_sim_info(self):

        self.num_events_processed += 1




    def add_event(self, new_event):
        self.event_queue.append(new_event)

        self.event_queue.sort(key=lambda _: _.time)

    def print_resp_info(self):
        for key, value in self.state.responders.items():
            print(key, value.__dict__)



