from enum import IntEnum


class ActionType(IntEnum):
    DISPATCH = 0
    ALLOCATION = 1
    DO_NOTHING = 2

class DispatchActions(IntEnum):
    SEND_NEAREST = 0

class LLEventType(IntEnum):
    INCIDENT = 0
    RESPONDER_AVAILABLE = 1
    ALLOCATION = 2

class MCTStypes(IntEnum):
    CENT_MCTS = 0