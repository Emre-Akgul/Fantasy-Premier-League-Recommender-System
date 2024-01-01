import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
from PLExpectedScoreCalculator import PLExpectedScoreCalculator
from Player import Player


class TestPLExpectedScoreCalculator(unittest.TestCase):

    def test_calculate_expected_score_found(self):
        # Test case where the player is found in the CSV
        calculator = PLExpectedScoreCalculator(gw=17)
        player = Player(355) # Haalands score is 1.1
        self.assertEqual(calculator.calculate_expected_score(player), 1.1) 

    def test_calculate_expected_score_not_found(self):
        # Test case where the player is not found in the CSV
        calculator = PLExpectedScoreCalculator(gw=1)
        player = Player(999)  # Assuming player ID 999 does not exist in the CSV
        self.assertEqual(calculator.calculate_expected_score(player), -100.0)

if __name__ == '__main__':
    unittest.main()