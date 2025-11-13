from abc import ABC, abstractmethod
#from typing import Type



class ObservationAttribute(ABC):
    def __init__(self, name: str, type: str) -> None:
        self.name = name # name of attribute
        self.type = int if type == 'int' else bool if type == 'bool' else None  # 'int' or 'bool'?

    def get_obs_value(self, attributes, default = 0) -> list[int]:
        return int(attributes.get(self.name, default))


class StandaloneQuadrantAttribute(ObservationAttribute):
    def __init__(self):
        super().__init__(name="quadrant", type='int')

    def get_obs_value(self, kwargs):
        current_step = kwargs.get("current_step", None)
        total_steps = kwargs.get("total_steps", None)
        if current_step == None and total_steps == None:
            return 4
        quad = current_step / total_steps
        return 1 if quad < 0.25 else 2 if quad < 0.50 else 3 if quad < 0.75 else 4

"""
class HostTypeAttribute(RedObservationAttribute):
    def __init__(self):
        super().__init__(name="type", type='int')

    def get_obs_value(self, value: HostTypes):
        return value


class HostBooleanAttribute(RedObservationAttribute):
    def __init__(self, name: str):
        super().__init__(name=name, type='bool')

    def get_obs_value(self, value: bool):
        return int(value)

class HostIntegerAttribute(RedObservationAttribute):
    def __init__(self, name):
        super().__init__(name=name, type='int')
    
    def get_obs_value(self, value: int):
        return value
"""