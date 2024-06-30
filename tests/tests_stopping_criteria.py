import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.helpers.stopping_criteria.single_obj import StoppingCriteria, ReferenceBased
from backend.core.helpers.stopping_criteria.single_obj import MaxIterations, MaxFuncEval, MaxRunTime
from backend.core.helpers.stopping_criteria.single_obj import percentage_improvement, ImprovementBestFuncEval, ImprovementAvgFuncEval
from backend.core.helpers.stopping_criteria.single_obj import MovementAvgFuncEval, MovementPositions
from backend.core.helpers.stopping_criteria.single_obj import MaxDistanceToBestDesignVariables, MaxPDistanceToBestDesignVariables, StandardDeviation, DifferenceBestWorstFuncEval


"""
Tests
"""

def test1():
    optimum = [0, 0]
    design_variables = np.array([[0, 0.00001], [0.001, 0.1], [0, 0.00001]])
    sc = StoppingCriteria(criteria=[ReferenceBased(optimum=optimum, percentage=70, tol=1e-3)])
    print(sc.info)    
    print('optimum: \n{}\ndesign variables: \n{}\n'.format(optimum, design_variables))
    print(sc.stop(design_variables=design_variables))
    print(sc.report)

def test2():
    sc = StoppingCriteria(criteria=[MaxIterations(max_iter=2), MaxFuncEval(max_funceval=3), MaxRunTime(max_runtime=100)])
    print(sc.stop(by_all=False, num_iter=12, num_funcevals=2, elapsed_time=100))
    print(sc.info)
    print(sc.stop(by_all=True, num_iter=12, num_funcevals=10, elapsed_time=100))
    print(sc.info)
    print(sc.report)

def test3():    
    xref = 0.1
    x = 0.01
    print('x = {}\nxref = {}'.format(x, xref))
    print('percentage improvement: ', percentage_improvement(x, xref, min_problem=True))
    
def test4():
    sc = StoppingCriteria(criteria=[ImprovementBestFuncEval(threshold=1, max_iter=2)])
    print(sc.info)    
    print(sc.stop(by_all=True, previous_best_funceval=0.01, best_funceval=0.01))
    print(sc.report)
    print(sc.stop(by_all=True, previous_best_funceval=0.01, best_funceval=0.001))
    print(sc.report)
    print(sc.stop(by_all=True, previous_best_funceval=0.001, best_funceval=0.00001))
    print(sc.report)
    print(sc.stop(by_all=True, previous_best_funceval=0.00001, best_funceval=0.00001))
    print(sc.report)
    print(sc.stop(by_all=True, previous_best_funceval=0.00001, best_funceval=0.00001))
    print(sc.report)
    print(sc.stop(by_all=True, previous_best_funceval=0.00001, best_funceval=0.00001))
    print(sc.report)

def test5():
    sc = StoppingCriteria(criteria=[ImprovementAvgFuncEval(threshold=1, max_iter=2)])
    print(sc.info)
    print(sc.stop(by_all=True, previous_avg_funceval=0.01, avg_funceval=0.01))
    print(sc.report)
    print(sc.stop(by_all=True, previous_avg_funceval=0.01, avg_funceval=0.01))
    print(sc.report)  
    print(sc.stop(by_all=True, previous_avg_funceval=0.01, avg_funceval=0.01))
    print(sc.report)

def test6():
    sc = StoppingCriteria(criteria=[MovementAvgFuncEval(threshold=5, max_iter=2, tol=1e-6)])
    print(sc.info)
    print(sc.stop(by_all='False', funceval=np.array([0.01, 0.01, 0.1, 0.1]), previous_avg_funceval=0.1))

    print(sc.report)
    print(sc.stop(by_all='False', funceval=np.array([0.01, 0.01, 0.1, 0.1]), previous_avg_funceval=0.1))

    print(sc.report)
    print(sc.stop(by_all='False', funceval=np.array([0.1, 0.1, 0.1, 0.1]), previous_avg_funceval=0.1))
    
    print(sc.report)
    print(sc.stop(by_all='False', funceval=np.array([0.1, 0.1, 0.1, 0.1]), previous_avg_funceval=0.1))
    
    print(sc.report)
    print(sc.stop(by_all='False', funceval=np.array([0.1, 0.1, 0.1, 0.1]), previous_avg_funceval=0.1))

def test7():
    sc = StoppingCriteria(criteria=[MovementPositions(threshold=5, max_iter=2, tol=1e-6)])

    print(sc.info)
    print(sc.stop(by_all=False, design_variables=np.array([[0.09, 0.1],[0.1, 0.1], [0.1, 0.1]]),
                                previous_design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]])))

    print(sc.report)
    print(sc.stop(by_all=False, design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]]),
                                previous_design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]])))

    print(sc.report)
    print(sc.stop(by_all=False, design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]]),
                                previous_design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]])))

    print(sc.report)
    print(sc.stop(by_all=False, design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]]),
                                previous_design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]])))
    print(sc.report)
    print(sc.stop(by_all=False, design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]]),
                                previous_design_variables=np.array([[0.1, 0.1],[0.1, 0.1], [0.1, 0.1]])))

    print(sc.report)

def test8():
    sc = StoppingCriteria([MaxDistanceToBestDesignVariables(threshold=1e-3)])
    print(sc.info)
    design_variables = np.array([[0.100, 0.10],
                                         [0.0999, 0.1],
                                         [0.1, 0.100001]])
    best_design_variables = np.array([0.1, 0.1])

    print(sc.stop(design_variables=design_variables, best_design_variables=best_design_variables))
    print(sc.report)

def test9():
    sc = StoppingCriteria([MaxPDistanceToBestDesignVariables(threshold=1e-3, percentage=50)])
    print(sc.info)
    design_variables = np.array([[1, 1.2],
                                 [1, 1.2],
                                 [1, 1.199],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1]])
    
    best_design_variables = np.array([1, 1.2])

    f = lambda x: 10 - np.sum(x)

    funceval = np.array([f(xi) for xi in design_variables])
    
    print('current funceval = ', funceval)
    print(sc.stop(design_variables=design_variables,
                  best_design_variables=best_design_variables,
                  funceval=funceval))
    print(sc.report)

def test10():
    sc = StoppingCriteria(criteria=[StandardDeviation(threshold=1e-2)])
    print(sc.info)
    current_design_variables = np.array([[1, 1.2],
                                        [1, 1.2],
                                        [1, 1.199],
                                        [1, 1.2],
                                        [1, 1.2],
                                        [1, 1.17]])
    print(sc.stop(design_variables=current_design_variables))
    print(sc.report)

def test11():
    sc = StoppingCriteria(criteria=[DifferenceBestWorstFuncEval(threshold=1e-2)])
    print(sc.info)
    funcevals = np.array([0.1, 0.1, 0.1001, 0.1])
    current_best_funceval = funcevals.min() 
    current_worst_funceval = funcevals.max()
    #print('current_best_funceval = ', current_best_funceval)
    #print('current_worst_funceval = ', current_worst_funceval)
    print(sc.stop(best_funceval=current_best_funceval, worst_funceval=current_worst_funceval))
    print(sc.report)

def test12():
    sc = StoppingCriteria([19.0, ['a']])
    print(sc.info)

def test13():
    sc = StoppingCriteria('my criteria')
    print(sc.info)

def test14():
    sc = StoppingCriteria([MaxFuncEval(10), MaxIterations(12)])
    print(sc.info)
    print()
    print(sc.report)

if __name__ == "__main__":
    test14()

