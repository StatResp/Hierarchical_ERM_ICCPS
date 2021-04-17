

class State:

    def __init__(self,
                 responders,
                 depots,
                 active_incidents,
                 time,
                 cells,
                 regions):

        self.depots = depots
        self.active_incidents = active_incidents
        self.time = time
        self.cells = cells
        self.regions = regions
        self.responders = responders
