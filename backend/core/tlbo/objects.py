import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from helpers.wrappers import InequalityConstraint, Constraints
from tlbo.operators import clip_array




class Bounds:
    """
    This class can be used to handle the bounds of the individuals' grades 
    """
    def __init__(self, lower_bounds, upper_bounds):
        """
        Initializes the bounds of the individuals' grades 
        
        Arguments:
            lower_bounds {1D numpy array/ list} -- lower bounds
            upper_bounds {1D numpy array/ list} -- upper bounds
        """

        self.lower = np.array(lower_bounds)
        self.upper = np.array(upper_bounds)
        self._validate()
        self.num_bounds = len(self.lower)

    def _validate(self):
        """
        Validates the input arguments
        
        Raises:
            ValueError: if shape of lower bounds is not equal to the shape of upper bounds
            ValueError: if not all upper bounds are greater than lower bounds
        """
        if self.lower.shape != self.upper.shape:
            raise ValueError('Lower bounds and upper bounds must have the same length')        
        if not np.all(self.upper > self.lower):
            raise ValueError('Upper bounds must be greater than lower bounds')

    def get_constraints(self):
        """
        Returns a list with the corresponding bounds constraints
        
        Returns:
            [list] --  bounds inequality constraints 
        """
        constraints = []
        for i in range(self.num_bounds):
            def gi_lower_bound(x):
                return self.lower[i] - x
            constraints.append(gi_lower_bound)

        for i in range(self.num_bounds):
            def gi_upper_bound(x):
                return x - self.upper[i]
            constraints.append(gi_upper_bound)
        return constraints

    @property
    def info(self):
        """
        Returns a description of the object
        
        Returns:
            str -- Description of the objects
        """
        s = 'BOUNDS\n'
        s += 'Lower bounds: \n{}\nUpper bounds: \n{}\n'.format(self.lower, self.upper)         
        return s


class ContinuousGrades:
    """
    This class can be used to handle the individuals' grades (or learners' grades) 
    in the TLBO algorithm, for problems with continuous design variables
    """
    def __init__(self, num_learners, lower_bounds, upper_bounds):
        """
        Initializes the individuals' grades
        
        Arguments:
            num_learners {int} -- the number of learners
            lower_bounds {1D numpy array/ list} -- lower bounds
            upper_bounds {1D numpy array/ list} -- upper bounds

        The values of the grades are initialized such that they lie within the bounds

        """
        self.num_learners = int(num_learners)
        self.bounds = Bounds(lower_bounds, upper_bounds)    
        self.num_subjects = self.bounds.num_bounds
        self._shape = (self.num_learners, self.num_subjects)
        self.values = np.random.uniform(self.bounds.lower, self.bounds.upper, (self.num_learners, self.num_subjects)).astype(float)
    
    @property
    def mean_values(self): 
        """
        Mean values of the individuals' grades for every subject
        
        Returns:
            [1D numpy array] -- means values of the individuals' grades for every subject
        """
        # mean values column-wise
        return np.mean(self.values, axis=0)

    def clip(self, return_idxs=False):
        """
        Clips the individuals' grades according to the bounds.
        See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html
        to check how the grades are clipped

        Keyword Arguments:
        return_idxs {bool} -- if True, it returns the indexes of the rows of the grades that have been clipped (default: {False})
        
        Returns:
        [2D numpy array, or tuple: (2D numpy array, 1D numpy array)] -- If ´return_idxs´ is True, it returns a tuple which contains
                                                                        the clipped grades and the indexes of the rows of the grades
                                                                        that have been clipped.
                                                                        Otherwise, only the clipped grades is returned.
        """
        return clip_array(self.values, self.bounds.lower, self.bounds.upper, return_idxs)

    def set_as(self, values):
        """
        Sets the individuals' grades as the given values
        
        Arguments:
            values {2D numpy array} -- new individuals' grades
        """
        if values.shape == self._shape:
            self.values = values
        else:
            raise ValueError('The new grades must be within an array of shape {}'.format(self._shape))

    def split(self, num_div):
        """
        Splits the individuals' grades in arrays with equal shapes,
        according to a given number of divisions.

        It may return a ValuError exception if the array split does not
        result in an equal division.
        
        Arguments:
            num_div {int} -- number of divisions
        
          Returns:
            [list] -- list of (numpy) arrays 
        """
        return np.split(self.values, int(num_div), axis=0)
    
    def reset_values(self):
        """
        Resets the values of the new individuals' grades.

        The grades are reset from uniform random distributions, according to the bounds.

        """
        self.values = np.random.uniform(self.bounds.lower, self.bounds.upper, (self.num_learners, self.num_subjects))
       
    def mutate(self, learners_ids, subjects_ids):
        """
        Set random values for the grades of some learners (indicated by their indexes) 
        at some specific subjects (also indicated by their indexes). These random
        values are within the corresponding bounds of the grades.
        
        Arguments:
            learners_ids {list of int} -- indexes of the learners whose grades are to be changed
            subjects_ids {list of int} -- indexes of the subjects to be changed
        """
        for this_learner in learners_ids:
            self.values[this_learner, subjects_ids] = np.random.uniform(self.bounds.lower[subjects_ids], self.bounds.upper[subjects_ids], (1, len(subjects_ids)))
        
    @property
    def info(self):
        """
        Returns a description of the object
        
        Returns:
            str -- Description of the objects
        """
        s = 'GRADES\n'
        s += 'Values: \n{}\n'.format(self.values)
        s += 'Mean values: \n{}\n'.format(self.mean_values)
        s += '\n{}\n'.format(self.bounds.info)
        return s
        

class Class:
    def __init__(self, num_learners, lower_bounds, upper_bounds): 
        """
        Initializes an object that represents a class.

        A class, in the context of TLBO, is basically an abstraction
        of a group of learners, each of these owning: a corresponding grade for
        every subject, and a resultant evaluation based on a given (test/objective)
        function
        
        Arguments:
            num_learners {int} -- number of learners in the class
            lower_bounds {1D numpy array/ list} -- lower bounds of grades
            upper_bounds {1D numpy array/ list} -- upper bounds of grades 
        """

        self.num_learners = int(num_learners)
        self.grades = ContinuousGrades(num_learners, lower_bounds, upper_bounds)
        self.num_subjects = self.grades.num_subjects 
        self.evaluations = None   # a 1D  numpy array (of size = self.num_learners) that holds the function evaluation for each learner, according to their grades

    def evaluate_learners(self, function, *args, **kwargs):
        """
        Evaluates the learners with respect to a (test/objective/fitness) function.
        Positional and keyworded arguments can be entered for the evaluation.
        
        Arguments:
            function {Function object} -- wrapper of the function 
                                          (Check the class ´Function´ in backend/core/helpers/wrappers.py)
        """
        self.evaluations = function.evaluate(self.grades.values, *args, **kwargs)

    def evaluate_some_learners(self, function, idxs, *args, **kwargs):
        """
        Evaluates some learners with respect to a (test/objective/fitness) function,
        based on the indexes (idxs) of the individuals that are to be evaluated.
        Positional and keyworded arguments can be entered for the evaluation.
        
        Arguments:
            function {Function object} -- wrapper of the function 
                                         (Check the class ´Function´ in backend/core/helpers/wrappers.py)

            idxs {list of int} -- indexes of the individuals that are to be evaluated 
        """
        self.evaluations = function.evaluate_some(self.grades.values, idxs, self.evaluations, *args, **kwargs)
    
    @property
    def min_evaluation(self):
        """
        Gets the index, grades, and evaluation corresponding to the individual with the 
        minimum function evaluation
        
        Returns:
            [dict] -- idx: an int which is the index of the individual learner with the minimum function evaluation
                      grades: grades of this individual, and 
                      evaluation: value of the corresponding evaluation for this individual
        """
        idx = np.argmin(self.evaluations)
        return None if self.evaluations is None else dict(idx=idx , grades=self.grades.values[idx], value=self.evaluations[idx])
    
    @property
    def max_evaluation(self):
        """
        Gets the index, grades, and evaluation corresponding to the individual with the 
        maximum function evaluation
        
        Returns:
            [dict] -- idx: an int which is the index of the individual learner with the maximum function evaluation
                      grades: grades of the this individual, and 
                      evaluation: value of the corresponding evaluation for this individual
        """
        idx = np.argmax(self.evaluations)
        return None if self.evaluations is None else dict(idx=idx , grades=self.grades.values[idx], value=self.evaluations[idx])

    @property
    def avg_evaluation(self):
        """
        Gets the average function evaluation among all of the learners' evaluations
        
        Returns:
            [float] -- average evaluation of the learners
        """
        return None if self.evaluations is None else np.mean(self.evaluations)

    @property
    def info(self):
        """
        Returns a description of the object
        
        Returns:
            str -- Returns a description of the object
        """
        s = 'CLASS\n'
        s += 'Number of learners: {}\nNumber of subjects: {}\n'.format(self.num_learners, self.num_subjects)
        s += '\nGrades:\n{}'.format(self.grades.info)
        s += '\nFunction evaluations:\n{}\nAverage evaluation value: {}\n'.format(self.evaluations, self.avg_evaluation)
        s += '\nLearner with max. evaluation: \n{}'.format(self.max_evaluation)
        s += '\nLearner with min. evaluation: \n{}'.format(self.min_evaluation)
        return s


class GroupOfClasses:
    """
    An object that represents a group of classes objects.
    Each class object in it might 'interact' with any other set of class objects.
    """
    def __init__(self):
        raise NotImplementedError('To be done')



    