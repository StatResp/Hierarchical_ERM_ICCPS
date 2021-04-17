class Event:

    def __init__(self,
                 event_type,
                 cell_loc,
                 time,
                 # event_id,
                 type_specific_information=None):

        self.cell_loc = cell_loc
        self.time = time
        # self.event_id = event_id

        '''
        Incident information: 
        - 'incident_obj' : incident object associated with the event
        '''
        self.type_specific_information = type_specific_information


        self.event_type = event_type

