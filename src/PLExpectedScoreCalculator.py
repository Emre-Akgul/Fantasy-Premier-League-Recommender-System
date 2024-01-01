from IExpectedScoreCalculator import IExpectedScoreCalculator
from Player import Player
import pandas as pd
class PLExpectedScoreCalculator(IExpectedScoreCalculator):

    def __init__(self, gw: int) -> None:
        super().__init__()
        self.gw = gw

    def calculate_expected_score(self, player: Player) -> float:
        """
        Calculate expected score based on Premier Leagues expected score announcement
        Expected scores are at Data/2023-24/gws/xP{self.gw}.csv
        Players expected score is in the xP column correspond to players id.
        """
        # Path to the CSV file
        file_path = f"Data/2023-24/gws/xP{self.gw}.csv"

        # Load the data
        data = pd.read_csv(file_path)

        # Find the expected score for the player
        player_data = data[data['id'] == player.id]
        if not player_data.empty:
            return player_data['xP'].iloc[0]
        else:
            # Return -100 if the player's data is not found
            return -100.0


