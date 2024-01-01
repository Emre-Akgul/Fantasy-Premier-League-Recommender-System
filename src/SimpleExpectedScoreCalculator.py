from IExpectedScoreCalculator import IExpectedScoreCalculator
from Player import Player
class SimpleExpectedScoreCalculator(IExpectedScoreCalculator):

    def calculate_expected_score(self, player: Player) -> float:
        # Simple expected score calculator
        # Just returns a fixed score of 10.0
        return 10.0