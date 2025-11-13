from enum import IntEnum

class HostTypes(IntEnum):
    UNKNOWN = 0
    USER = 1
    SERVER = 2
    DECOY_USER = 3
    DECOY_SERVER = 4

host_types_map = {
    "UNKNOWN": HostTypes.UNKNOWN,
    "USER": HostTypes.USER,
    "SERVER": HostTypes.SERVER,
    "DECOY_USER": HostTypes.DECOY_USER,
    "DECOY_SERVER": HostTypes.DECOY_SERVER
}