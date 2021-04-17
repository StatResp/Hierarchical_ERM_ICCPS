from math import hypot


class GridCellRouter:

    def __init__(self,
                 travel_rate):
        '''
        each grid cell is 1 mile. Travel rate should be in miles per second
        :param travel_rate:
        '''

        self.travel_rate = travel_rate


    @staticmethod
    def get_distance(cell1, cell2):
        if cell1 is None:
            raise ValueError('cell1 is none')
        elif cell2 is None:
            raise ValueError('cell2 is none')
        return hypot(cell1[0] - cell2[0], cell1[1] - cell2[1]) #  + 1

    def get_travel_time(self, cell1, cell2):

        distance = self.get_distance(cell1, cell2)

        return distance / self.travel_rate

