from abc import ABC, abstractmethod
from typing import Type

import numpy as np


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass


# This class is just an example and is not useful in any way
class SumFi(Fi):
    def __call__(self, state):
        return np.sum(state)


class HumanoidFi(Fi):
    @staticmethod
    def _normal_dist_density(x: float, mean: float, sd: float):
        prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        return prob_density

    def __call__(self, state):
        return 0.0 if 1.35 < state[0] < 1.45 else -np.power((1.4 - state[0]) * 100, 2)

class TStudentHeightLowPenaltyShiftedFiveHundred(Fi):
    def __call__(self, state: np.ndarray):
        return self._base_penalty(state) + 500

    def _base_penalty(self, state: np.ndarray):
        index = Constants.HEIGHT_INDEX
        middle_of_dist = Constants.HEIGHT_NOMINAL_VALUE

        degree_of_freedom = 0.01
        scale = 0.35

        return 10 * t.pdf(state[index], df=degree_of_freedom, scale=scale, loc=middle_of_dist)


class FiFactory:
    FI_MAPPING = {
        'sum': SumFi,
        'humanoid': HumanoidFi,
        'default': HumanoidFi,
        'tStudentFromSeminary' :TStudentHeightLowPenaltyShiftedFiveHundred
    }

    @staticmethod
    def get_fi(name: str):
        fi = FiFactory.FI_MAPPING.get(name, None)

        if fi is None:
            raise ValueError(f"Unknown fi: {name}, viable options are: {FiFactory.FI_MAPPING.keys()}")

        return fi()

    @staticmethod
    def register(name: str, _class=Type[Fi]):
        assert issubclass(_class, Fi), "Can only register classes that are subclasses of Fi"
        assert name not in FiFactory.FI_MAPPING, "This name is already taken"

        FiFactory.FI_MAPPING[name] = _class
