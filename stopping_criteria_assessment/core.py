import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from optimizers.single_obj import BasicTLBOptimizer
from backend.utilities.test_functions import ackley, bukin6, cross_in_tray, drop_wave, eggholder, gramacylee, griewank, holder_table, levy
from backend.utilities.test_functions import levy13, rastrigin, schwefel, shubert
from backend.utilities.test_functions import bohachevksy1, sphere, sum_different_powers, sum_squares
from backend.utilities.test_functions import booth, matyas, mccormick, zakharov
from backend.utilities.test_functions import three_hump_camel, six_hump_camel, rosenbrock
from backend.utilities.test_functions import easom, michalewicz
from backend.utilities.test_functions import beale, branin, colville, forrester, goldstein_price

from backend.core.helpers.stopping_criteria.single_obj import ReferenceBased 
from backend.core.helpers.stopping_criteria.single_obj import MaxIterations, MaxFuncEval, MaxRunTime 
from backend.core.helpers.stopping_criteria.single_obj import ImprovementBestFuncEval, ImprovementAvgFuncEval 
from backend.core.helpers.stopping_criteria.single_obj import MovementAvgFuncEval, MovementPositions 
from backend.core.helpers.stopping_criteria.single_obj import MaxDistanceToBestDesignVariables, MaxPDistanceToBestDesignVariables, StandardDeviation, DifferenceBestWorstFuncEval 

import pandas as pd


class StoppingCriteriaAssessment:
    def __init__(self, max_iter, num_learners, stopping_criteria, test_functions, lower_bounds, upper_bounds, opt_funceval, tol, num_experiments) -> None:
        """This class can be used to conduct performance assessments of a list of stopping criteria for a list of test functions

        Args:
            stopping_criteria (list of StoppingCriteria object): list of stopping criteria
            functions (list of functions): list of Python functions that define test functions
            num_experiments (int): number of experiments to assess the performance of every stopping criteria for each test function
        """
        self._max_iter = max_iter
        self._num_learners = num_learners
        self._stopping_criteria = stopping_criteria

        self._test_functions = test_functions
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._opt_funceval = opt_funceval
        
        self._tol = tol
        self._num_experiments = num_experiments

        self._data = pd.DataFrame()
        
    @property
    def num_stopping_criteria(self):
        return len(self._stopping_criteria)
    
    @property
    def num_test_functions(self):
        return len(self._test_functions)

    def run(self):
        for i, test_func in enumerate(self._test_functions):
            lower_bounds_i = self._lower_bounds[i]
            upper_bounds_i = self._upper_bounds[i]
            opt_funceval_i = self._opt_funceval[i]
            
            func_name = test_func.__name__
            file_name = f'{func_name}.txt'
            file = open(file_name, "w")
            
            for criterion in self._stopping_criteria:
                stopping_criteria_ = [MaxIterations(max_iter=self._max_iter), criterion]
                opt = BasicTLBOptimizer(num_learners=self._num_learners,
                                        lower_bounds=lower_bounds_i,
                                        upper_bounds=upper_bounds_i,
                                        stopping_criteria=stopping_criteria_,
                                        log_file_name=f'{func_name}_{criterion.name}')    
                opt.optimize3(objective_function=test_func, opt_funceval=opt_funceval_i, tol=self._tol, num_times=self._num_experiments)
                
                file.write(str(opt.num_successful_run/self._num_experiments)+" "+str(opt.avg_func_eval)+"\n")

            file.close()


                
if __name__ == '__main__':

    def test1():
        max_iter = 100
        num_learners = 20
        test_functions = [sphere, rosenbrock, rastrigin, griewank, ackley, goldstein_price, easom, schwefel]
        d = 2
        lower_bounds = [2 * [-5.12],
                        d * [-5],
                        4 * [-5.12],
                        2 * [-600],
                        2 * [-32.768],
                        2 * [-2],
                        2 * [-100],
                        5 * [-500]]
        
        upper_bounds = [2 * [5.12],
                        d * [10],
                        4 * [5.12],
                        2 * [600],
                        2 * [32.768],
                        2 * [2],
                        2 * [100],
                        5 * [500]]
        
        opt_funceval = [0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        3.0,
                        -1.0,
                        0.0]

        stopping_criteria = [ImprovementBestFuncEval(threshold=1e-6, max_iter=20),
                             ImprovementAvgFuncEval(threshold=1e-6, max_iter=20),
                             MovementAvgFuncEval(threshold=1e-6, max_iter=20),
                             MovementPositions(threshold=1e-6, max_iter=20),
                             MaxDistanceToBestDesignVariables(threshold=1e-4),
                             MaxPDistanceToBestDesignVariables(threshold=1e-4, percentage=1.0),
                             StandardDeviation(threshold=1e-4),
                             DifferenceBestWorstFuncEval(threshold=1e-6)]

        tol = 1e-3
        num_experiments = 100

        assessment = StoppingCriteriaAssessment(max_iter,
                                                num_learners,
                                                stopping_criteria,
                                                test_functions,
                                                lower_bounds,
                                                upper_bounds,
                                                opt_funceval,
                                                tol,
                                                num_experiments)

        assessment.run()




    test1()
