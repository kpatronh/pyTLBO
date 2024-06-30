import numpy as np
from itertools import permutations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.helpers.wrappers import ObjectiveFunction, Constraints


def clip_array(array, lower_bounds, upper_bounds, return_idxs=False):
    """
    Given an interval, values in the array outside the interval are clipped to the interval edges.
    For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.    

    See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html.


    Arguments:
        array {2D numpy array} -- array containing elements to clip
        lower_bounds {scalar or array_like or None} -- minimum value or values.
                                                       If None, clipping is not performed on lower interval edge

        upper_bounds {scalar or array_like or None} -- maximum value or values.
                                                       If None, clipping is not performed on upper interval edge

    
    Keyword Arguments:
        return_idxs {bool} -- if True, it returns the indexes of the rows of the arrays that have been clipped (default: {False})
    
    Returns:
        [2D numpy array, or tuple: (2D numpy array, 1D numpy array)] -- If ´return_idxs´ is True, it returns a tuple which contains
                                                                        the clipped array and the indexes of the rows of the array
                                                                        that have been clipped.
                                                                        Otherwise, only the clipped array is returned.

    """
    if return_idxs:
        array_copy = array.copy()
        array = np.clip(array, lower_bounds, upper_bounds)
        idxs = np.unique(np.where(np.not_equal(array, array_copy))[0])
        return array, idxs
    else:
        return np.clip(array, lower_bounds, upper_bounds)

class TeachingLearningOperators:
    """
    This class holds a set of (teaching-learning-based) methods that are used 
    to operate on the learners' grades, in order to find an optimal solution to
    an optimization problem.

    These methods are the core of the TLBO algorithm 'modus operandi'.
    """

    @staticmethod
    def teacher_phase(class_, function, *args, **kwargs):
        """
        This method allows the execution of the so-called "teacher phase" in the TLBO. 
        In this phase, the teacher interacts with the learners to increase the mean result
        of the class (i.e. the learners learn through the teacher).
        
        Positional and keyworded arguments can be entered for the evaluation of the function.
        
        Arguments:
            class_ {object of type Class} -- the class object, holding the teacher and the learners
                                                (check the class Class in backend/core/tlbo/objects.py )
            function {object of type Function} -- the (test/objective/fitness) function used to evaluate
                                                  the (individuals) learners 
                                                  (check the class Function in backend/core/helpers/wrappers.py)
    
        """

        # 1) computing the mean differences between the current learners' grades and the teacher's grades
        r = np.random.rand(class_.num_subjects)
        Tf = np.random.choice([1, 2])  # the teaching factor which decides the value of mean to be changed
        mean_grades = class_.grades.mean_values
        teacher_info = class_.min_evaluation # for minimization problems
        mean_differences = r * (teacher_info['grades'] - Tf * mean_grades)

        # 2) computing the new grades and their corresponding function evaluations
        new_grades = class_.grades.values + mean_differences
        new_grades = clip_array(new_grades, class_.grades.bounds.lower, class_.grades.bounds.upper, return_idxs=False) 
        new_evaluations = function.evaluate(new_grades, *args, **kwargs)

        # 3) if the new grades are better (in terms of function evaluation) than the current grades, then they are accepted
        idxs = np.where(new_evaluations < class_.evaluations)[0] # for minimization problems
        class_.grades.values[idxs] = new_grades[idxs]
        class_.evaluations[idxs] = new_evaluations[idxs]
        return class_

    @staticmethod
    def learner_phase(class_, function, *args, **kwargs):
        """
        This method allows the execution of the so-called "learner-phase" in the TLBO.
        In this phase, the learners randomly interact among each other for knowledge transfer.

        Positional and keyworded arguments can be entered for the evaluation of the function.
        
        Arguments:
            class_ {object of type Class} -- the class object holding the teacher and the learners
                                                  (check the class Function in backend/core/helpers/wrappers.py)
                                             (check the class Class in backend/core/tlbo/objects.py )
            function {object of type Function} -- the (test/objective/fitness) function used to evaluate
                                                  the (individuals) learners 
           
        """

        r = np.random.rand(class_.num_subjects)
        new_grades = np.empty_like(class_.grades.values)

        # 1) computing the new grades and their corresponding function evaluations

        # every learner has to interact with any other learner
        for p in range(class_.num_learners):  
            
            # selecting another learner q with a different function evaluation than p
            q = np.random.randint(0, class_.num_learners) 
            while p == q:
                q = np.random.randint(0, class_.num_learners)

            # updating the grades of learner p, based on the interaction between p and q
            if class_.evaluations[p] < class_.evaluations[q]:  # for minimization problems
                new_grades[p,:] = class_.grades.values[p,:] + r * (class_.grades.values[p,:] - class_.grades.values[q,:])
            else:
                new_grades[p,:] = class_.grades.values[p,:] + r * (class_.grades.values[q,:] - class_.grades.values[p,:])

        new_grades = clip_array(new_grades, class_.grades.bounds.lower, class_.grades.bounds.upper, return_idxs=False)
        new_evaluations = function.evaluate(new_grades, *args, **kwargs)
        
        # 2) if the new grades are better (in terms of function evaluation) than the current grades, then they are accepted
        idxs = np.where(new_evaluations < class_.evaluations)[0] # for minimization problems
        class_.grades.values[idxs] = new_grades[idxs]
        class_.evaluations[idxs] = new_evaluations[idxs]
        
        return class_


