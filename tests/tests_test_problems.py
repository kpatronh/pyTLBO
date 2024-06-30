import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.utilities.test_problems import f1, g1_1, g2_1, g3_1, g4_1, g5_1, g6_1, g7_1, g8_1, g9_1
from optimizers.single_obj import ConstrainedTLBOptimizer
from backend.core.helpers.stopping_criteria.single_obj import MaxIterations 
from backend.core.helpers.wrappers import SelfAdaptivePenaltyFunction


def test_G1():
    stopping_criteria = [MaxIterations(max_iter=1000)]
    lower_bounds = 13 * [0]
    upper_bounds = [1,1,1,1,1,1,1,1,1,100,100,100,1]
    opt = ConstrainedTLBOptimizer(num_learners=70, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria)

    opt.optimize(objective_function=f1,
                 inequality_constraints=[g1_1, g2_1, g3_1, g4_1, g5_1, g6_1, g7_1, g8_1, g9_1],
                 equality_constraints=[])



if __name__ == "__main__":
    test_G1()