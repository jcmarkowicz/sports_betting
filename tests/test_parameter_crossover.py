import unittest
import numpy as np
from ufc_betting.Models.RiskManagement.parameter_crossover import compound, winning_ranges


class CrossoverTests(unittest.TestCase):
    def test_compounding_and_stop_without_topups(self):
        np.testing.assert_allclose(compound([.1,-.5,.2]),[500,550,275,330])
        np.testing.assert_allclose(compound([-.9,1]),[500,50,50])

    def test_multiple_winning_regions_and_ties(self):
        self.assertEqual(winning_ranges(np.arange(6),[-1,2,3,0,-1,1]),[(1.,2.),(5.,5.)])
        self.assertEqual(winning_ranges(np.arange(3),[0,0,0]),[])


if __name__ == '__main__':
    unittest.main()
