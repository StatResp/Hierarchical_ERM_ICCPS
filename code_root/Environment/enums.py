from enum import IntEnum


class EventType(IntEnum):
    INCIDENT = 0
    RESPONDER_AVAILABLE = 1
    ALLOCATION = 2
    FAILURE = 3

class RespStatus(IntEnum):
    WAITING = 0
    RESPONDING = 1
    SERVICING = 2
    IN_TRANSIT = 3

