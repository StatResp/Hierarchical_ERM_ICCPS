import copy

class SendNearestDispatchPolicy:

    def __init__(self, travel_model):
        self.travel_model = travel_model


    def get_responder_to_incident_assignments(self, state):
        available_responders = [_[1] for _ in state.responders.items() if _[1].available]
        active_incidents = copy.copy(state.active_incidents)  # TODO | check normal copy is okay

        assignments = []

        while len(available_responders) > 0 and len(active_incidents) > 0:
            # attend to incidents in fifo order
            chosen_incident = min(active_incidents, key=lambda _: _.time)

            # find closest responder
            closest_available_responder = min(available_responders,
                                              key=lambda _: self.travel_model.get_travel_time(_.cell_loc,
                                                                                           chosen_incident.cell_loc,
                                                                                              None))

            available_responders.remove(closest_available_responder)
            active_incidents.remove(chosen_incident)

            assignments.append((closest_available_responder, chosen_incident))

        return assignments

    # def send_random_responders_to_pending_incidents(self, state):
    #
    #     available_responders = [copy.deepcopy(_[1]) for _ in state.responders.items() if _[1].available]
    #     active_incidents = copy.deepcopy(state.active_incidents)
    #
    #     assignments = []
    #
    #     while len(available_responders) > 0 and len(active_incidents) > 0:
    #         chosen_resp = random.choice(available_responders)
    #         available_responders.remove(chosen_resp)
    #
    #         chosen_incident = random.choice(active_incidents)
    #         active_incidents.remove(chosen_incident)
    #
    #         assignments.append((chosen_resp, chosen_incident))
    #
    #     return assignments