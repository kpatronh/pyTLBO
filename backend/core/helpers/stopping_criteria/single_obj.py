import numpy as np
import inspect

'''
Stopping criteria for single-objective optimization problems
------------------------------------------------------------

There are available five categories of stopping criteria that can be used with 
population-based optimization algorithms [1]:
1) Reference-based criteria
2) Exhaustion-based criteria
3) Improvement-based criteria
4) Movement-based criteria
5) Distribution-based criteria

This module provides classes for each of these stopping criteria.


References:
[1] Zielinski, K., Peters, D. and Laur, R.,  Stopping criteria for Single-Objective Optimization.
'''



'''
1) Reference-based criteria
'''

class ReferenceBased: 
    """
    The algorithm stops when a certain percentage of the population
    converged to the optimum. That percentage is named convergence percentage.

    Applicable only for test functions where the optimum is known.
    """

    def __init__(self, optimum, percentage=40, tol=1e-3):
        """
        Initializes the stopping criterion       
        
        Arguments:
            optimum {1D numpy array} -- the optimum solution, whose length is the number of design variables
        
        Keyword Arguments:
            percentage {float} -- min. percentage of population for convergence (default: {40})
            tol {float} -- tolerance to compute the convergence ratio  (default: {1e-3})
        """
        
        self.percentage = float(percentage)
        self.tol = float(tol)
        self.optimum = np.array(optimum)
        self._convergence_perc = None

    def stop(self, design_variables):
        """
        Indicates whether the algorithm should stop based on the convergence ratio.

        Arguments:
            design_variables {2D numpy array} -- the current values of the design variables for every individual; 
                                                 it has n rows and m columns, where n: number of individuals, 
                                                 and m: number of design variables
                                    
        Returns: 
            bool -- stopping indicator
        """

        self._convergence_perc = 100 * np.sum(np.linalg.norm(design_variables - self.optimum, axis=1) < self.tol)/design_variables.shape[0]
        return self._convergence_perc >= self.percentage
    
    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion
        
        Returns:
            str -- description
        """
        return 'ReferenceBased(min. percentage = {}, tol = {}, optimum = \n{})\n'.format(self.percentage, self.tol, self.optimum)


    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'ReferenceBased - convergence percentage = {}\n'.format(self._convergence_perc)
    
    @property
    def name(self):
        return 'ReferenceBased'


'''
2) Exhaustion-based criteria
'''
class MaxIterations:
    """
    The algorithm stops when a maximum number of iterations is reached 
    """
    
    def __init__(self, max_iter=100):
        """
        Initializes the criterion
        
        Keyword Arguments:
            max_iter {int} -- maximum number of iterations (default: {100})
        """
        self.max_iter = int(max_iter)
        self._num_iter = None

    def stop(self, num_iter):
        """
        Indicates whether the algorithm should stop based on the current number of iterations passed
        
        Arguments:
            num_iter {int} -- current number of iterations
        
        Returns:
            bool -- stopping indicator
        """

        self._num_iter = int(num_iter)
        return self._num_iter > self.max_iter

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion
        
        Returns:
            str -- description
        """
        return 'MaxIterations(max_iter = {})\n'.format(self.max_iter)


    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MaxIterations -  number of iterations = {}   ({}/{})  \n'.format(self._num_iter, self._num_iter, self.max_iter)


    @property
    def name(self):
        return 'MaxIterations'

class MaxFuncEval:
    """
    The algorithm stops when a maximum number of (objective) function evaluations is reached 
    """
    def __init__(self, max_funceval=1000):
        """
        Initializes the criterion
        
        Keyword Arguments:
            max_funceval {int} -- maximum number of function evaluations (default: {1000})
        """

        self.max_funceval = int(max_funceval)
        self._num_funcevals = None

    def stop(self, num_funcevals):
        """
        Indicates whether the algorithm should stop based on the current number of function evaluations
        
        Arguments:
            num_funcevals {int} -- current number of function evaluations
        
        Returns:
            bool -- stopping indicator
        """
        self._num_funcevals = num_funcevals
        return self._num_funcevals > self.max_funceval


    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MaxFuncEval(max_funceval = {})\n'.format(self.max_funceval)

 
    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MaxFuncEval - number of function evaluations = {}\n'.format(self._num_funcevals)

    @property
    def name(self):
        return 'MaxFuncEval'

class MaxRunTime:
    """
    The algorithm stops when a maximum runtime (in seconds) is reached 
    """
    def __init__(self, max_runtime=3600):
        """
        Initializes the criterion
        
        Keyword Arguments:
            max_runtime {int} -- maximum runtime in seconds (default: {3600})
        """
        self.max_runtime = max_runtime
        self._elapsed_time = None

    def stop(self, elapsed_time):
        """
        Indicates whether the algorithm should stop based on the current elapsed time

        Arguments:
            elapsed_time {float} -- number of elapsed seconds
        
        Returns:
            bool -- stopping indicator
        """
        self._elapsed_time = elapsed_time
        return elapsed_time > self.max_runtime


    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MaxRunTime(max_runtime = {})\n'.format(self.max_runtime)

 
    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MaxRunTime - elapsed time = {}\n'.format(self._elapsed_time)

    @property
    def name(self):
        return 'MaxRunTime'

'''
3) Improvement-based criteria
'''

class ImprovementBestFuncEval:
    """
    The algorithm should stop if the improvement of the best (objective) function evaluation
    is below a threshold for a number of iterations
    """

    def __init__(self, threshold=1, max_iter=20):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- improvement threshold in percentage (default: {1 %})
            max_iter {int} -- number of iterations (default: {20})
        """
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self._num_iter = 0
        self._improvement = None

    def stop(self, previous_best_funceval, best_funceval):
        """
        Indicates whether the algorithm should stop based on the improvement of the 
        best function evaluation in the population
        
        Arguments:
            previous_best_funceval {float} -- previous best function evaluation       
            best_funceval {float} -- current best function evaluation
        
        Returns:
            bool -- stopping indicator
        """

        self._improvement = abs(best_funceval - previous_best_funceval)
        if self._improvement < self.threshold:
            self._num_iter += 1
        else:
            self._num_iter = 0
        return self._num_iter > self.max_iter

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'ImprovementBestFuncEval(threshold = {}, max_iter = {})\n'.format(self.threshold, self.max_iter)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'ImprovementBestFuncEval - improvement of the best function evaluation after {} consecutive iterations = {}%\n'.format(self._num_iter, self._improvement)

    @property
    def name(self):
        return 'ImprovementBestFuncEval'

class ImprovementAvgFuncEval:
    """
    The algorithm should stop if the improvement of the average objective function value
    is below a threshold for a number of iterations
    """
    def __init__(self, threshold=1, max_iter=20):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- improvement threshold in percentage (default: {1 %})
            max_iter {int} -- number of iterations (default: {20})
        """
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self._num_iter = 0
        self._improvement = None

    def stop(self, previous_avg_funceval, avg_funceval):
        """
        Indicates whether the algorithm should stop based on the improvement of the 
        average function evaluation in the population
        
        Arguments:
            previous_avg_funceval {float} -- previous average function evaluation       
            avg_funceval {float} -- current average function evaluation
        
        Returns:
            bool -- stopping indicator
        """

        self._improvement = abs(avg_funceval - previous_avg_funceval)
        if self._improvement < self.threshold:
            self._num_iter += 1
        else:
            self._num_iter = 0
        return self._num_iter > self.max_iter

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'ImprovementAvgFuncEval(threshold = {}, max_iter = {})\n'.format(self.threshold, self.max_iter)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """

        return 'ImprovementAvgFuncEval - improvement of the average function evaluation after {} consecutive iterations = {}%\n'.format(self._num_iter, self._improvement)

    @property
    def name(self):
        return 'ImprovementAvgFuncEval'


'''
4) Movement-based criteria
'''

class MovementAvgFuncEval:
    """
    The algorithm should stop if the movement in the population with respect to the average
    objective function evaluation (objective space) is below a threshold for a number of generations/iterations
    """

    def __init__(self, threshold=1e-2, max_iter=20):
        """
        Initializes the criterion
        
        Keyword Arguments:
            max_iter {int} -- number of iterations (default: {20})
            threshold {float} -- tolerance to determine the movement ratio (default: {1e-2})
        """
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self._num_iter = 0
        self._max_mov = None

    def stop(self, previous_funceval, funceval, previous_avg_funceval, avg_funceval):
        """
        Indicates whether the algorithm should stop based on the movement in the population
        with respect to the average objective function evaluation
        
        Arguments:
            funceval {1D numpy array} -- (current) function evaluation for every individual in the population
            previous_avg_funceval {float} -- previous average objective function evaluation
        
        Returns:
            bool -- stopping indicator
        """
        previous_dif = previous_funceval - previous_avg_funceval
        dif = funceval - avg_funceval
        mov = np.abs(previous_dif - dif)
        self._max_mov = np.max(mov)
        self._num_iter = self._num_iter + 1 if self._max_mov < self.threshold else 0
        return self._num_iter > self.max_iter


    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MovementAvgFuncEval(threshold = {}, max_iter = {})\n'.format(self.threshold, self.max_iter)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MovementAvgFuncEval - max. movement of the average function evaluation after {} consecutive iterations = {}%\n'.format(self._num_iter, self._max_mov)

    @property
    def name(self):
        return 'MovementAvgFuncEval'
       
class MovementPositions:
    """
    The algorithm should stop if the movement in the population with respect to positions
    (design variables space) is below a threshold for a number of generations
    """

    def __init__(self, threshold=1e-2, max_iter=20):
        """
        Initializes the criterion
        
        Keyword Arguments:
            max_iter {int} -- number of iterations (default: {20})
            threshold {float} -- tolerance to determine the movement ratio (default: {1e-2})
        """
        self.threshold = float(threshold)
        self.max_iter = int(max_iter)
        self._num_iter = 0
        self._max_mov = None

    def stop(self, design_variables, previous_design_variables):
        """
        Indicates whether the algorithm should stop based on the movement in the population
        with respect to positions (parameter space)
        
        Arguments:
            design_variables {2D numpy array} -- the (current) design variables for every individual;
                                                 it has n rows (n: number of individuals) and m columns
                                                 (m: number of design variables)
            previous_design_variables {2D numpy array} --  the previous design variables for every individual;
                                                           it has n rows (n: number of individuals) and m columns
                                                           (m: number of design variables)
        
        Returns:
            bool -- stopping indicator
        """
        mov = np.abs(design_variables - previous_design_variables)
        self._max_mov = np.max(mov)

        self._num_iter = self._num_iter + 1 if self._max_mov < self.threshold else 0
        return self._num_iter > self.max_iter


    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MovementPositions(threshold = {}, max_iter = {})\n'.format(self.threshold, self.max_iter)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MovementPositions - max. movement of the design variables after {} consecutive iterations = {}%\n'.format(self._num_iter, self._max_mov)

    @property 
    def name(self):
        return 'MovementPositions'


'''
5) Distribution-based criteria
'''

class MaxDistanceToBestDesignVariables:
    """
    The algorithm should stop if the maximum distance from the individuals' design variables
    to the best population vector is below a threshold.
    """

    def __init__(self, threshold=1e-3):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- distance threshold (default: {1e-3})
        """
        self.threshold = float(threshold)
        self._max_distance = None

    def stop(self, design_variables, best_design_variables):
        """
        Indicates whether the algorithm should stop based on the maximum distance 
        between the individuals' design variables and the best design variables

        Arguments:
            design_variables {2D numpy array} -- the (current) design variables for every individual;
                                                 it has n rows (n: number of individuals) and m columns
                                                 (m: number of design variables)
            best_design_variables {1D numpy array} -- the (current) best design variables among 
                                                      the individuals' design variables; it has m components,
                                                      (m: number of design variables)

        Returns:
            bool -- stopping indicator
        
        """ 
        self._max_distance = np.max(np.linalg.norm(design_variables - best_design_variables, axis=1))
        return self._max_distance < self.threshold
 
    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MaxDistanceToBestDesignVariables(threshold = {})\n'.format(self.threshold)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MaxDistanceToBestDesignVariables - maximum distance between all the design variables and the best design variables = {}\n'.format(self._max_distance)


    @property
    def name(self):
        return 'MaxDistanceToBestDesignVariables'

class MaxPDistanceToBestDesignVariables:
    """
    The algorithm should stop if the maximum distance from the best p percent of the individuals' design variables
    to the best individual's design variables is below a threshold.
    """
    def __init__(self, threshold=1e-3, percentage=40):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- distane threshold (default: {1e-3})
            percentage {float} -- percentage p  (default: {40}). It must be less than 100.
        """     
        self.threshold = float(threshold)
        self.percentage = float(percentage)
        self._max_distance = None
    
    def stop(self, design_variables, funceval, best_design_variables):
        """
        Indicates whether the algorithm should stop based on the maximum distance 
        between the best p percent of the individuals' design variables to the 
        best individual's design variables

        Arguments:
            design_variables {2D numpy array} -- the (current) design variables for every individual;
                                                 it has n rows (n: number of individuals) and m columns
                                                 (m: number of design variables)
            
            funceval {1D numpy array} -- the (current) objective function evaluation for every individual;
                                         it has n components (n: number of individuals)

            best_design_variables {1D numpy array} -- the current best design variables among 
                                                      the individuals' design variables;
                                                      it has m components (m: number of design variables)

        Returns:
            bool -- stopping indicator
        
        """
        n = int(np.ceil(self.percentage/100 * funceval.shape[0]))  # number of best grades to select
        idxs = np.argsort(funceval)[0:n]  # indexes of the best p grades in order (for minimization problems; the best are the smallest funceval)
        bestp_design_variables = design_variables[idxs]    
        self._max_distance = np.linalg.norm(bestp_design_variables - best_design_variables, axis=1)[-1]
        return self._max_distance < self.threshold

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'MaxPDistanceToBestDesignVariables(threshold = {}, percentage = {})\n'.format(self.threshold, self.percentage)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'MaxPDistanceToBestDesignVariables - maximum distance between the best {}% design variables and the best design variables = {}\n'.format(self.percentage, self._max_distance) 


    @property
    def name(self):
        return 'MaxPDistanceToBestDesignVariables'

class StandardDeviation:
    """
    The algorithm should stop if the standard deviation of the individuals' design variables
    is below a given threshold.
    """
    def __init__(self, threshold=1e-3):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- standard deviation threshold (default: {1e-3})
        """
        self.threshold = float(threshold)
        self._standard_deviation = None
    
    def stop(self, design_variables):
        """
         Indicates whether the algorithm should stop based on the standard deviation of the 
         individuals' design variables
        
        Arguments:
            design_variables {2D numpy array} -- the (current) design variables for every individual;
                                                 it has n rows (n: number of individuals) and m columns
                                                 (m: number of design variables)
        
        Returns:
            bool -- stopping indicator
        """
        xr = np.linalg.norm(design_variables, axis=1)
        xm = np.mean(xr)
        self._standard_deviation = np.sqrt(np.sum((xr - xm)**2)/(design_variables.shape[0]-1))
        return self._standard_deviation < self.threshold

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'StandardDeviation(threshold = {})\n'.format(self.threshold)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'StandardDeviation - standard deviation of the design variables = {}\n'.format(self._standard_deviation)

    @property
    def name(self):
        return 'StandardDeviation'

class DifferenceBestWorstFuncEval:
    """
    The algorithm should stop if the difference between the best and the worst objective function evaluation
    is below a given threshold.
    """
    def __init__(self, threshold=1e-3):
        """
        Initializes the criterion
        
        Keyword Arguments:
            threshold {float} -- difference threshold (default: {1e-3})
        """ 
        self.threshold = float(threshold)
        self._difference = None

    def stop(self, best_funceval, worst_funceval):
        """
        Indicates whether the algorithm should stop based on the difference between
        the current best and worst (objective) function evaluations
        
        Arguments:
            best_funceval {float} -- the (current) best objective function evaluation
            worst_funceval {float} -- the (current) worst objective function evaluation
        
        Returns:
            bool -- stopping indicator
        """
        self._difference = np.abs(best_funceval - worst_funceval)
        return self._difference < self.threshold

    @property
    def info(self):
        """
        A description of the parameters for the stopping criterion

        Returns:
            str -- description
        """
        return 'DifferenceBestWorstFuncEval(threshold = {})\n'.format(self.threshold)

    @property
    def report(self):
        """ 
        A report of the current state of the criterion  
        
        Returns:
            str -- report
        """
        return 'DifferenceBestWorstFuncEval - absolute difference between the best and worst function evaluations = {}\n'.format(self._difference)

    @property
    def name(self):
        return 'DifferenceBestWorstFuncEval'


'''
Stopping criteria for population-based optimizers
'''
class StoppingCriteria:
    """
    This class can be used to set the stopping criteria for population-based optimizers.
    """
    
    __CRITERIA = [ReferenceBased,
                  MaxIterations, MaxFuncEval, MaxRunTime,
                  ImprovementBestFuncEval, ImprovementAvgFuncEval,
                  MovementAvgFuncEval, MovementPositions,
                  MaxDistanceToBestDesignVariables, MaxPDistanceToBestDesignVariables, StandardDeviation, DifferenceBestWorstFuncEval]

    def __init__(self, criteria):
        """
        Initializes the stopping criteria object.
        
        Arguments:
            criteria {list} -- a list with the selected stopping criteria objects.
                               Every stopping criterion must be initialized in this list.
                               The following stopping criteria are available:

                                1) Reference-based criteria:
                                    ReferenceBased

                                2) Exhaustion-based criteria
                                    MaxIterations, MaxFuncEval, MaxRunTime

                                3) Improvement-based criteria:
                                    ImprovementBestFuncEval, ImprovementAvgFuncEval

                                4) Movement-based criteria:
                                    MovementAvgFuncEval, MovementPositions

                                5) Distribution-based criteria
                                    MaxDistanceToBestDesignVariables, MaxPDistanceToBestDesignVariables, StandardDeviation, DifferenceBestWorstFuncEval
   
        """
        
        criteria = list(criteria)

        self._validate_criteria(criteria)
        self.criteria = criteria
        self.num_criteria = len(self.criteria)

        self._stop_per_criterion = np.array(self.num_criteria * [False])
        self._criteria_input_args = []
        self._get_criteria_input_args()

    def _validate_criteria(self, criteria):
        """
        Validates the selected criteria 
        
        Arguments:
            criteria {list} -- list of stopping criteria
        
        Raises:
            ValueError: when there is at least one invalid criterion object 
        """
        for i, criterion in enumerate(criteria):
            if not type(criterion) in StoppingCriteria.__CRITERIA:
                raise ValueError('Invalid stopping criterion "{}" at index {}'.format(criterion.__class__.__name__, i))
    
    def _get_criteria_input_args(self):
        """
        Gets the input arguments for every criterion in the criteria list.
        The arguments are stored at a list (self._criteria_input_args)
        """
        for criterion in self.criteria:
            self._criteria_input_args.append(inspect.getargspec(criterion.stop).args[1:])
            #self._criteria_input_args.append(inspect.getargs(criterion.stop).args[1:])

    def stop(self, by_all='True', **kwargs):
        """
        Boolean stopping indicator
        
        Keyword Arguments:
            by_all {bool} -- if True, the algorithm should stop when every criterion is fulfilled,
                             otherwise, it should stop when at least one criterion is fulfilled (default: {'True'})
        
        Raises:
            KeyError: when there is a missing argument for at least one criterion
            ValueError: when there is an invalid input argument for at least one criterion
        
        Returns:
            [bool] -- boolean stopping indicator
        """
        for i, criterion in enumerate(self.criteria):
            args = []
            for param in self._criteria_input_args[i]:
                try:
                    args.append(kwargs[param])
                except:
                    raise KeyError('Missing input argument "{}" for criterion "{}"'.format(param, criterion.__class__.__name__))
            try:
                self._stop_per_criterion[i] = criterion.stop(*args)
            except:
                raise ValueError('Invalid input arguments for criteria "{}"'.format(criterion.__class__.__name__))
        return np.all(self._stop_per_criterion) if by_all else np.any(self._stop_per_criterion)

    @property
    def info(self):
        msg = ''
        for criterion in self.criteria:
            msg += '  ' + criterion.info +'\t\t\t\t'
        return msg

    @property
    def report(self):
        msg = ''
        for criterion in self.criteria:
            msg += criterion.report + '\t\t\t\t'
        return msg


    @property
    def stop_per_criterion(self):
        return self._stop_per_criterion





