import datetime
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.helpers.wrappers import ObjectiveFunction, Constraints
from backend.core.tlbo.objects import Bounds, Class
from backend.core.tlbo.operators import TeachingLearningOperators
from backend.core.helpers.wrappers import SelfAdaptivePenaltyFunction
from backend.utilities.reporter import Reporter
from backend.utilities.visualizer import Plotter

from backend.core.helpers.stopping_criteria.single_obj import MaxIterations, MaxFuncEval, MaxRunTime
from backend.core.helpers.stopping_criteria.single_obj import ImprovementBestFuncEval, ImprovementAvgFuncEval
from backend.core.helpers.stopping_criteria.single_obj import MovementAvgFuncEval, MovementPositions
from backend.core.helpers.stopping_criteria.single_obj import StandardDeviation, DifferenceBestWorstFuncEval
from backend.core.helpers.stopping_criteria.single_obj import StoppingCriteria



"""
Teaching-Learning-Based optimizers for single-objective continuous optimization problems
"""



class _BaseTLBOptimizer:
    """
    An abstract class for building teaching-learning-based optimizers.
    """

    def __init__(self, num_learners, lower_bounds, upper_bounds, stopping_criteria, log_file_name='tlbo'):
        """
        Initializes the base teaching-learning-based optimizer.

        Arguments:
            num_learners {int} -- number of learners (individuals) in the class
            lower_bounds {1D numpy array/ list} -- lower bounds for the learners' grades
            upper_bounds {1D numpy array/ list} -- upper bounds for the learners' grades
            stopping_criteria {list} -- list of stopping criteria to be used during the optimization.
                                        The components of this list might a (non-duplicated) set of 
                                        stopping criteria from backend/core/helpers/stopping_criteria/single_obj.py

        
        Keyword Arguments:
            log_file_name {str} -- name of the corresponding .log file (default: 'tlbo').
                                   If a .log file is not desired, log_file_name should be set to None.
        """

    
        self._reporter = Reporter(name='optimizer', on_console=True, log_file_name=log_file_name, info_format=2)
        self._reporter.info('TEACHING-LEARNING-BASED OPTIMIZATION\n\t\t\t\t------------------------------------\n')
        self._reporter.info('Initializing the teaching-learning-based optimizer...')
        
        self._class = Class(num_learners, lower_bounds, upper_bounds)
        self._stopping_criteria = StoppingCriteria(stopping_criteria)
        self._operators = TeachingLearningOperators()  # TLBO operators for the learners to evolve
        self._state_variables = None  
        self._initial_time = None   # A reference time for measuring the elapsed time during the optimization

        self._reporter.info('Number of design variables (subjects): {}'.format(self._class.num_subjects))
        self._reporter.info('Number of individuals (learners): {}'.format(self._class.num_learners))
        self._reporter.info('Lower bounds for design variables: {}'.format(self._class.grades.bounds.lower))
        self._reporter.info('Upper bounds for design variables: {}'.format(self._class.grades.bounds.upper))
        self._reporter.info('Stopping criteria: \n\t\t\t\t{}'.format(self._stopping_criteria.info))

    def _init_state_variables(self, objective_function, *args, **kwargs):
        """
        Initializes the optimization state variables.

        Arguments: 
            objective_function {object of type ObjectiveFunction} -- Wrapper of the objective function
                                                                     See ObjectiveFunction at backend/core/helpers/wrappers.py

        Positional and keyworded arguments can be used for the evaluation of 
        the objective function.

        """
        self._reporter.info('Initializing the optimization state variables...')
        self._initial_time = datetime.datetime.now()
        self._class.evaluate_learners(objective_function, *args, **kwargs)
        min_eval = self._class.min_evaluation
        max_eval = self._class.max_evaluation
        self._state_variables = dict(funceval=self._class.evaluations,
                                     best_funceval=min_eval['value'],
                                     avg_funceval=self._class.avg_evaluation,
                                     worst_funceval=max_eval['value'],

                                     design_variables=self._class.grades.values,
                                     best_design_variables=min_eval['grades'],
                                     worst_design_variables=max_eval['grades'],
                                     
                                     previous_funceval=None,
                                     previous_best_funceval=None,
                                     previous_avg_funceval=None,
                                     previous_worst_funceval=None,
                                     
                                     previous_design_variables=None,
                                     previous_best_design_variables= None,
                                     previous_worst_design_variables=None,
                                     
                                     best_all_funceval=[min_eval['value']],
                                     
                                     num_iter=0,  
                                     num_funceval=objective_function._num_evals,
                                     elapsed_time=datetime.datetime.now() - self._initial_time,
                                     converged=False,
                                     successful=False)
                                     
    def _update_state_variables(self, objective_function):
        """
        Updates the optimization state variables as the optimization moves along.
        This method must be called after the learner phase in every iteration of the TLBO.

        Arguments: 
            objective_function {object of type ObjectiveFunction} -- Wrapper of the objective function

        """
        self._reporter.info('Updating the optimization state variables...')
        min_eval = self._class.min_evaluation
        max_eval = self._class.max_evaluation

        self._state_variables['previous_funceval'] = self._state_variables['funceval']
        self._state_variables['previous_best_funceval'] = self._state_variables['best_funceval']
        self._state_variables['previous_avg_funceval'] = self._state_variables['avg_funceval']
        self._state_variables['previous_worst_funceval'] = self._state_variables['worst_funceval']

        self._state_variables['previous_design_variables'] = self._state_variables['design_variables']
        self._state_variables['previous_best_design_variables'] = self._state_variables['best_design_variables']
        self._state_variables['previous_worst_design_variables'] = self._state_variables['worst_design_variables']

        self._state_variables['funceval'] = self._class.evaluations
        self._state_variables['best_funceval'] = min_eval['value']
        self._state_variables['avg_funceval'] = self._class.avg_evaluation
        self._state_variables['worst_funceval'] = max_eval['value']

        self._state_variables['design_variables'] = self._class.grades.values
        self._state_variables['best_design_variables'] = min_eval['grades']
        self._state_variables['worst_design_variables'] = max_eval['grades']

        if self._state_variables['previous_best_funceval'] < self._state_variables['best_funceval']:
            self._state_variables['best_all_funceval'].append(self._state_variables['previous_best_funceval'])
        else:
            self._state_variables['best_all_funceval'].append(self._state_variables['best_funceval'])
        
        self._state_variables['num_iter'] += 1
        self._state_variables['num_funceval'] = objective_function._num_evals
        self._state_variables['elapsed_time'] = datetime.datetime.now() - self._initial_time  
        
    @property
    def state_variables(self):
        """
        Returns the optimization state variables
        
        Returns:
            dict -- a dictionary with the main state variables for the optimization.
                    The dictionary holds the following keys:
                        - num_iter: number of current iterations
                        - num_funceval: number of objective function evaluations
                        - elapsed time: elapsed time in seconds
                        - previous_best_funceval: minimum function evaluation at the previous iteration
                        - best_funceval: minimum function evaluation at the current iteration
                        - previous_avg_funceval: average function evaluation at the previous iteration
                        - avg_funceval: average function evalation at the current iteration
                        - previous_funceval: the objective function evaluation for each learner at the previous iteration
                        - funceval: the objective function evaluation for each learner
                        - previous_design_variables: design variables at the previous iteration
                        - design_variables: design variables at the current iteration
                        - best_design_variables: design variables with the minimum objective function evaluation
                        - worst_funceval: maximum objective function evaluation
            
        """
        return self._state_variables
    
    def show_partial_results(self):
        self._reporter.info('Minimum objective function evaluation so far: {}'.format(self._state_variables['best_funceval']))
        self._reporter.info('Optimal design variables so far: {}'.format(self._state_variables['best_design_variables']))
 
    def show_results(self):
        self._reporter.info('RESULTS\n\t\t\t\t-------')
        self._reporter.info('Minimum objective function evaluation: {}'.format(self._state_variables['best_funceval']))
        self._reporter.info('Optimal design variables: {}'.format(self._state_variables['best_design_variables']))
        self._reporter.info('Number of iterations: {}'.format(self._state_variables['num_iter']))
        self._reporter.info('Number of objective function evaluations: {}'.format(self.state_variables['num_funceval']))
        self._reporter.info('Elapsed time: {} \n'.format(self._state_variables['elapsed_time']))
            
    def plot_convergence(self):
        x = np.arange(len(self._state_variables['best_all_funceval']))
        y = np.array(self._state_variables['best_all_funceval'])      
        title = 'TLBO Convergence Plot'
        xaxis_title = 'iterations'
        yaxis_title = 'objective function value'
        Plotter.scatter_interactive(x=x, y=y, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)



class BasicTLBOptimizer(_BaseTLBOptimizer):
    """
    A class for the basic (original) teaching-learning-based optimizer.

    Instances from this class allow to find (nearly) optimal solutions 
    for unconstrained single objective optimization problems.

    Bounds constraints are considered by the optimizer by clipping the 
    learners' grades when infeasible grades (out of the bounds) 
    are obtained during the search.

    """

    def __init__(self, num_learners, lower_bounds, upper_bounds, stopping_criteria, log_file_name='tlbo'):

        """
        Initializes the basic teaching-learning-based optimizer.

        Arguments:
            num_learners {int} -- number of learners (individuals) in the class
            lower_bounds {1D numpy array/ list} -- lower bounds for the learners' grades
            upper_bounds {1D numpy array/ list} -- upper bounds for the learners' grades
            stopping_criteria {list} -- a list with the selected stopping criteria objects.
                                        There are available five categories of stopping criteria,
                                        which are described at backend/core/helpers/stopping_criteria/single_obj.py

        Keyword Arguments:
            log_file_name {str} -- name of the corresponding .log file (default: 'tlbo').
                                   If a .log file is not desired, log_file_name should be set to None.
        """

        _BaseTLBOptimizer.__init__(self, num_learners,
                                  lower_bounds, upper_bounds,
                                  stopping_criteria, log_file_name)
    
        self._avg_func_eval = None
        self._num_successful_run = None

    def optimize(self, objective_function, stop_by_all_criteria=False, convergence_plot=True, *args, **kwargs):
        """
        By calling this method, the optimizer searches for the minimum of the objective function.
    
        Arguments:
            objective_function {function} -- the objective function to be minimized

        Keyword Arguments:
            stop_by_all_criteria {bool} -- if True, the optimizer will stop when every criterion has been met;
                                           otherwise, it will stop when at least one criterion has been met.
                                           (default: False)

            convergence_plot {bool} -- if True, a convergence plot is shown

        Positional and keyworded arguments are allowed for the evaluation of the objective function.
        """
        
        _wrapped_objfunc = ObjectiveFunction(objective_function)
        
        self._reporter.info('Searching for the minimum of the objective function...')
        self._reporter.info('Objective function: {}\n'.format(_wrapped_objfunc.name))
        
        self._init_state_variables(_wrapped_objfunc, *args, **kwargs) 

        while True:
            self._operators.teacher_phase(self._class, _wrapped_objfunc, *args, **kwargs)
            self._operators.learner_phase(self._class, _wrapped_objfunc, *args, **kwargs)

            self._update_state_variables(_wrapped_objfunc)  
            self.show_partial_results()

            if self._stopping_criteria.stop(by_all=stop_by_all_criteria, **self._state_variables):
                break
            else:
                self._reporter.info(self._stopping_criteria.report)
            
            
        self._reporter.info('Optimization finished according to the stopping criteria\n')
        self.show_results()
                
        if convergence_plot:
            self.plot_convergence()

    def optimize2(self, objective_function, opt_funceval, tol=1e-3, convergence_plot=False, *args, **kwargs):
        """
        By calling this method, the optimizer searches for the minimum of the objective function.
    
        Arguments:
            objective_function {function} -- the objective function to be minimized

        Keyword Arguments:
            stop_by_all_criteria {bool} -- if True, the optimizer will stop when every criterion has been met;
                                           otherwise, it will stop when at least one criterion has been met.
                                           (default: False)

            convergence_plot {bool} -- if True, a convergence plot is shown

        Positional and keyworded arguments are allowed for the evaluation of the objective function.
        """
        
        _wrapped_objfunc = ObjectiveFunction(objective_function)
        
        self._reporter.info('Searching for the minimum of the objective function...')
        self._reporter.info('Objective function: {}\n'.format(_wrapped_objfunc.name))
        
        self._init_state_variables(_wrapped_objfunc, *args, **kwargs) 

        while True:
            self._operators.teacher_phase(self._class, _wrapped_objfunc, *args, **kwargs)
            self._operators.learner_phase(self._class, _wrapped_objfunc, *args, **kwargs)

            self._update_state_variables(_wrapped_objfunc)  
                        
            stop_execution = self._stopping_criteria.stop(by_all=False, **self._state_variables)

            self._state_variables['successful'] = abs(self._state_variables['best_funceval'] - opt_funceval) < tol
            
            if not self._stopping_criteria.stop_per_criterion[0]: # MaxIter must be the first criterion in the criteria list
                if stop_execution:
                    self._state_variables['converged'] = True
            
            self.show_partial_results()


            if stop_execution:
                break
            else:
                self._reporter.info(self._stopping_criteria.report)
            
            
        self._reporter.info('Optimization finished according to the stopping criteria\n')
        self.show_results()
                
        if convergence_plot:
            self.plot_convergence()

    def optimize3(self, objective_function, opt_funceval, tol=1e-3, num_times=100, *args, **kwargs):
        """
        Run the optimization problem many times to retrieve information (state variables) for further statistical results.

        Positional and keyworded arguments are allowed for the evaluation of the objective function.

        """
        self._reporter.info(f'Running {num_times} times the optimization problem...')
        
        avg_func_eval = 0
        num_successful_run = 0

        for _ in range(num_times):
            self.optimize2(objective_function, opt_funceval, tol, convergence_plot=False, *args, **kwargs)    

            if self.state_variables['successful']:
                num_successful_run += 1
                avg_func_eval += self.state_variables['num_funceval']

        if num_successful_run > 0:
            avg_func_eval /= num_successful_run
            avg_func_eval = int(avg_func_eval)
        else:
            avg_func_eval = 'undefined'


        self._reporter.info(f'Number of average function evaluations : {avg_func_eval}')
        self._reporter.info(f'Number of successful runs              : {num_successful_run}')

        self._reporter.info(f'Finished running {num_times} times the optimization problem.')
        
        self._avg_func_eval = avg_func_eval
        self._num_successful_run = num_successful_run       


    @property
    def avg_func_eval(self):
        return self._avg_func_eval
    
    @property
    def num_successful_run(self):
        return self._num_successful_run



    def parametric(self, min_learners=10, max_learners=20, div=3):
        """
        Run a parametric analysis with respect to the number of learners, with fixed stopping criteria
        
        Arguments:
            max_learners {[type]} -- [description]
        
        Keyword Arguments:
            min_learners {int} -- [description] (default: {10})
        """
        pass

    

class ConstrainedTLBOptimizer(_BaseTLBOptimizer):
    """
    A class for the constrained teaching-learning-based optimizer.

    Instances from this class allow to find (nearly) optimal solutions
    for constrained single objective optimization problems, by using 
    a self-adaptive penalty function based algorithm, described in the following article:

    Tessema, B. and Yen, G. (2006). A Self Adaptive Penalty Function Based Algorithm for Constrained Optimization,
    IEEE Congress on Evolutionary Computation. Vancouver, Canada

    """

    def __init__(self, num_learners,
                       lower_bounds, upper_bounds,
                       stopping_criteria,
                       log_file_name='tlbo'):

        """
        Initializes the basic teaching-learning-based optimizer.

        Arguments:
            num_learners {int} -- number of learners (individuals) in the class
            lower_bounds {1D numpy array/ list} -- lower bounds for the learners' grades
            upper_bounds {1D numpy array/ list} -- upper bounds for the learners' grades
            stopping_criteria {list} -- a list with the selected stopping criteria objects.
                                        There are available five categories of stopping criteria,
                                        which are described at backend/core/helpers/stopping_criteria/single_obj.py

        Keyword Arguments:
            log_file_name {str} -- name of the corresponding .log file (default: 'tlbo').
                                   If a .log file is not desired, log_file_name should be set to None.
        """

        _BaseTLBOptimizer.__init__(self, num_learners,
                                  lower_bounds, upper_bounds,
                                  stopping_criteria, log_file_name)


    def optimize(self, objective_function, inequality_constraints, equality_constraints, tol=1e-4, stop_by_all_criteria=False, convergence_plot=True, *args, **kwargs):
        """
        By calling this method, the optimizer searches for the minimum of the objective function,
        under the defined set of equality and inequality constraints.

        Every component of ´inequality_constraints´ defines the left side (g) of an inequality of type g(x) <= 0.
        Similarly, every component of ´equality_constraints´ defines the left side (h) of an inequality of type h(x) = 0.
        (An equality constraint h(x) = 0 is converted into an equivalent inequality constraint: abs(h(x)) - tol <= 0).

        Bounds constraints are automatically added by the optimizer.


        Arguments:
        objective_function {function} -- the objective function to be minimized
        inequality_constraints {list of functions} -- list of inequality constraints
        equality_constraints {list of functions} -- list of equality constraints

        Keyword Arguments:
            tol {float} -- tolerance for the equivalent inequality constraints (default: {1e-4})
            
            stop_by_all_criteria {bool} -- if True, the optimizer will stop when every criterion has been met;
                                           otherwise, it will stop when at least one criterion has been met.
                                           (default: False)

            convergence_plot {bool} -- if True, a convergence plot is shown

        Positional and keyworded arguments are allowed for the evaluation of the objective function and constraints.
        """


        _wrapped_objfunc = SelfAdaptivePenaltyFunction(objective_function=objective_function, 
                                                       inequality_constraints=inequality_constraints,
                                                       equality_constraints=equality_constraints,
                                                       tol=float(tol))   

        self._reporter.info('Searching for the minimum of the objective function...')
        self._reporter.info('Objective function: {}\n'.format(_wrapped_objfunc.name))
        
        self._init_state_variables(_wrapped_objfunc, *args, **kwargs) 

        while True:
            self._operators.teacher_phase(self._class, _wrapped_objfunc, *args, **kwargs)
            self._operators.learner_phase(self._class, _wrapped_objfunc, *args, **kwargs)

            self._update_state_variables(_wrapped_objfunc)  
            self.show_partial_results()

            if self._stopping_criteria.stop(by_all=stop_by_all_criteria, **self._state_variables):
                break
            else:
                self._reporter.info(self._stopping_criteria.report)
            
            
        self._reporter.info('Optimization finished according to the stopping criteria\n')
        self.show_results()
                
        if convergence_plot:
            self.plot_convergence()

















        