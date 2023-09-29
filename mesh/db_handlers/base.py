import numpy as np

from typing import Any, List, Union, Dict
from abc import ABC, abstractclassmethod

class VectorDB(ABC):
    @abstractclassmethod
    def connect(self):
        pass

    @abstractclassmethod
    def disconnect(self):
        pass
    
    @abstractclassmethod
    def search(self, vec: np.ndarray, k: int, **kwargs):
        pass

    @abstractclassmethod
    def add(self, ids: Union[np.ndarray, List], vec: Union[np.ndarray, List]):
        pass

    @abstractclassmethod
    def delete(self, targets: Any):
        pass

    @abstractclassmethod
    def erase(self):
        pass

    @abstractclassmethod
    def to_json(self, include_instance: bool = False):
        pass

    @staticmethod
    @abstractclassmethod
    def from_json(dict: Dict, return_dict: bool = False):
        pass