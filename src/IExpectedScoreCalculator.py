from abc import ABC, abstractmethod
from Player import Player
class IExpectedScoreCalculator(ABC):

    @abstractmethod
    def calculate_expected_score(self, player: Player) -> float:
        """
        Calculate the expected score for a given player.

        :param player: Player object
        :return: Expected score as a float
        """
        pass