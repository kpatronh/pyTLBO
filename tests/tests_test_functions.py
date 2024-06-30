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
from backend.core.helpers.stopping_criteria.single_obj import MaxIterations, MaxPDistanceToBestDesignVariables, ImprovementBestFuncEval, ImprovementAvgFuncEval

def test_ackley():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=50)]
    lower_bounds = 2 * [-32.768]
    upper_bounds = 2 * [32.768]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='ackley')    
    opt.optimize(objective_function=ackley, stop_by_all_criteria=False)
    
def test_bukin6():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = [-15, -5]
    upper_bounds = [-3,3]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='bukin6')    
    opt.optimize(objective_function=bukin6, stop_by_all_criteria=False)
    
def test_cross_in_tray():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='cross_in_tray')    
    opt.optimize(objective_function=cross_in_tray, stop_by_all_criteria=False)
    
def test_drop_wave():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-5.12]
    upper_bounds = 2 * [5.12]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='drop_wave')    
    opt.optimize(objective_function=drop_wave, stop_by_all_criteria=False)
    
def test_eggholder(): 
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-512]
    upper_bounds = 2 * [512]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='eggholder')    
    opt.optimize(objective_function=eggholder, stop_by_all_criteria=False)
    
def test_gramacylee():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=20)]     
    lower_bounds = [0.5]
    upper_bounds = [2.5]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='gramacylee')    
    opt.optimize(objective_function=gramacylee, stop_by_all_criteria=False)
    
def test_griewank():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=20)]     
    lower_bounds = 2 * [-600]
    upper_bounds = 2 * [600]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='griewank')    
    opt.optimize(objective_function=griewank, stop_by_all_criteria=False)
    
def test_holder_table():
    stopping_criteria = [MaxIterations(max_iter=200), ImprovementBestFuncEval(threshold=1, max_iter=20)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='holder_table')    
    opt.optimize(objective_function=holder_table, stop_by_all_criteria=False)
    
def test_levy():
    stopping_criteria = [MaxIterations(max_iter=50)] #, ImprovementBestFuncEval(threshold=1, max_iter=20)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='levy')    
    opt.optimize(objective_function=levy, stop_by_all_criteria=False)
    
def test_levy13():
    # stopping_criteria = [MaxIterations(max_iter=50)] 
    stopping_criteria = [MaxIterations(max_iter=50), ImprovementBestFuncEval(threshold=1, max_iter=10)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='levy13')    
    opt.optimize(objective_function=levy13, stop_by_all_criteria=False)
    
def test_rastrigin():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 4 * [-5.12]
    upper_bounds = 4 * [5.12]
    opt = BasicTLBOptimizer(num_learners=50, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='rastrigin')    
    opt.optimize(objective_function=rastrigin, stop_by_all_criteria=False)
    
def test_schaffer2():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=10)]     
    lower_bounds = 2 * [-100]
    upper_bounds = 2 * [100]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='schaffer2')    
    opt.optimize(objective_function=schaffer2, stop_by_all_criteria=False)

def test_schwefel():
    stopping_criteria = [MaxIterations(max_iter=300)]
    lower_bounds = 5 * [-500]
    upper_bounds = 5 * [500]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='schwefel')    
    opt.optimize(objective_function=schwefel, stop_by_all_criteria=False)

def test_shubert():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='shubert')    
    opt.optimize(objective_function=shubert, stop_by_all_criteria=False)

def test_bohachevsky1():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-100]
    upper_bounds = 2 * [100]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='bohachevksy1')    
    opt.optimize(objective_function=bohachevksy1, stop_by_all_criteria=False)

def test_sphere():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-5.12]
    upper_bounds = 2 * [5.12]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='sphere')    
    opt.optimize(objective_function=sphere, stop_by_all_criteria=False)

def test_sum_different_powers():    
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-1]
    upper_bounds = 2 * [1]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='sum_different_powers')    
    opt.optimize(objective_function=sum_different_powers, stop_by_all_criteria=False)

def test_sum_squares():    
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-5.12]
    upper_bounds = 2 * [5.12]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='sum_squares')    
    opt.optimize(objective_function=sum_squares, stop_by_all_criteria=False)

def test_booth():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='booth')    
    opt.optimize(objective_function=booth, stop_by_all_criteria=False)

def test_matyas():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-10]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='matyas')    
    opt.optimize(objective_function=matyas, stop_by_all_criteria=False)

def test_mccormick():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = [-1.5, -3]
    upper_bounds = [4, 4]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='mccormick')    
    opt.optimize(objective_function=mccormick, stop_by_all_criteria=False)

def test_zakharov():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-5]
    upper_bounds = 2 * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='zakharov')    
    opt.optimize(objective_function=zakharov, stop_by_all_criteria=False)

def test_three_hump_camel():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-5]
    upper_bounds = 2 * [5]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='three_hump_camel')    
    opt.optimize(objective_function=three_hump_camel, stop_by_all_criteria=False)
    
def test_six_hump_camel():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = [-3, -2]
    upper_bounds = [3, 2]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='six_hump_camel')    
    opt.optimize(objective_function=six_hump_camel, stop_by_all_criteria=False)
    
def test_rosenbrock():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    d = 2
    lower_bounds = d * [-5]
    upper_bounds = d * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='rosenbrock')    
    opt.optimize(objective_function=rosenbrock, stop_by_all_criteria=False)

def test_easom():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-100]
    upper_bounds = 2 * [100]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='easom')    
    opt.optimize(objective_function=easom, stop_by_all_criteria=False)

def test_michalewicz():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    d = 5
    lower_bounds = d * [0.0]
    upper_bounds = d * [np.pi]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='michalewicz')    
    opt.optimize(objective_function=michalewicz, stop_by_all_criteria=False)

def test_beale():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-4.5]
    upper_bounds = 2 * [4.5]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='beale')    
    opt.optimize(objective_function=beale, stop_by_all_criteria=False)

def test_branin():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = [-5, 0]
    upper_bounds = [10, 15]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='branin')    
    opt.optimize(objective_function=branin, stop_by_all_criteria=False)

def test_colville():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 4 * [-10]
    upper_bounds = 4 * [10]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='colville')    
    opt.optimize(objective_function=colville, stop_by_all_criteria=False)

def test_forrester():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = [0]
    upper_bounds = [1]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='forrester')    
    opt.optimize(objective_function=forrester, stop_by_all_criteria=False)

def test_goldstein_price():
    stopping_criteria = [MaxIterations(max_iter=300), ImprovementBestFuncEval(threshold=1, max_iter=50)]     
    lower_bounds = 2 * [-2]
    upper_bounds = 2 * [2]
    opt = BasicTLBOptimizer(num_learners=100, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='goldstein_price')    
    opt.optimize(objective_function=goldstein_price, stop_by_all_criteria=False)



def test_sphere2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    lower_bounds = 2 * [-5.12]
    upper_bounds = 2 * [5.12]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='sphere')    
    opt.optimize3(objective_function=sphere, opt_funceval=0.0, tol=1e-6, num_times=1000)

def test_rosenbrock2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    d = 2
    lower_bounds = d * [-5]
    upper_bounds = d * [10]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='rosenbrock')    
    opt.optimize3(objective_function=rosenbrock, opt_funceval=0.0, tol=1e-6, num_times=1000)

def test_rastrigin2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    lower_bounds = 4 * [-5.12]
    upper_bounds = 4 * [5.12]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='rastrigin')    
    opt.optimize3(objective_function=rastrigin, opt_funceval=0.0, tol=1e-6, num_times=1000)
    
def test_griewank2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    lower_bounds = 2 * [-600]
    upper_bounds = 2 * [600]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='griewank')    
    opt.optimize3(objective_function=griewank, opt_funceval=0.0, tol=1e-6, num_times=1000)


def test_ackley2():
    stopping_criteria = [MaxIterations(max_iter=1000), ImprovementAvgFuncEval(threshold=1e-2, max_iter=5)]
    lower_bounds = 2 * [-32.768]
    upper_bounds = 2 * [32.768]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='ackley')    
    opt.optimize3(objective_function=ackley, opt_funceval=0.0, tol=1e-3, num_times=500)

def test_goldstein_price2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    lower_bounds = 2 * [-2]
    upper_bounds = 2 * [2]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='goldstein_price')    
    opt.optimize3(objective_function=goldstein_price, opt_funceval=3.0, tol=1e-6, num_times=1000)


def test_easom2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]     
    lower_bounds = 2 * [-100]
    upper_bounds = 2 * [100]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='easom')    
    opt.optimize3(objective_function=easom, opt_funceval=-1.0, tol=1e-6, num_times=1000)


def test_schwefel2():
    stopping_criteria = [MaxIterations(max_iter=100), ImprovementBestFuncEval(threshold=1e-2, max_iter=30)]
    lower_bounds = 5 * [-500]
    upper_bounds = 5 * [500]
    opt = BasicTLBOptimizer(num_learners=20, lower_bounds=lower_bounds, upper_bounds=upper_bounds, stopping_criteria=stopping_criteria, log_file_name='schwefel')    
    opt.optimize3(objective_function=schwefel, opt_funceval=0.0, tol=1e-6, num_times=1000)
    


if __name__ == "__main__":
    #test_sphere2()
    #test_rosenbrock2()
    #test_rastrigin2()
    #test_griewank2()
    test_ackley2()
    #test_goldstein_price2()
    #test_easom2()
    #test_schwefel2()


    
    #test_ackley()
    #test_bukin6()
    #test_cross_in_tray()
    #test_drop_wave()
    #test_eggholder()
    #test_gramacylee()
    #test_griewank()
    #test_holder_table()
    #test_levy()
    #test_levy13()
    #test_rastrigin()
    #test_schaffer2()
    #test_schwefel()
    #test_shubert()
    #test_bohachevsky1()
    #test_sphere()
    #test_sum_different_powers()
    #test_sum_squares()
    #test_booth()
    #test_matyas()
    #test_mccormick()
    #test_zakharov()
    #test_three_hump_camel()
    #test_six_hump_camel()
    #test_rosenbrock()
    #test_easom()
    #test_michalewicz()
    #test_beale()
    #test_branin()
    #test_colville()
    #test_forrester()
    #test_goldstein_price()
