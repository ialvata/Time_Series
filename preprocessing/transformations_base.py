from  abc import ABC, abstractmethod

class Transformation(ABC):
    _type:str
    @property
    @abstractmethod
    def parameters(self):
        pass
    @abstractmethod
    def apply(self):
        pass
    @abstractmethod
    def invert(self):
        pass