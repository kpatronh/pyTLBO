import numpy as np

'''
Wrappers of some relevant objects for optimizers
------------------------------------------------
'''


class Function:
    """
    Wrapper of a mathematical function 
    """
    
    def __init__(self, f):
        """
        Initializes the function wrapper

        Arguments:
            f {function} -- the function to be wrapped
        
        Raises:
            ValueError: when the input argument is not a callable object (function or method)
        """
        if not hasattr(f, '__call__'):
            raise ValueError('The function must be a callable object')
            
        self.function = f
        self.name = self.function.__name__
        self._num_evals = 0   # the number of evaluations is counted
     
    def evaluate(self, x, *args, **kwargs):
        """
        Evaluates the function at a point or at a set of points, as defined by x.
        Positional and keyworded arguments can be used for the evaluation of the original function.

        
        Arguments:
            x {numpy array} -- the point or points of evaluation.
                               It can be a 1D numpy array to represent a point of evaluation,
                               or a 2D numpy array to represent a set of points of evaluation
                               (every row is a point).

        
        Returns:
            float or 1D numpy array -- the function evaluated at the point (float) or 
                                       at every point (1D numpy array)
        """
        if x.ndim == 2:
            self._num_evals += x.shape[0]
            return np.array([self.function(xi, *args, **kwargs) for xi in x]).flatten()

        elif x.ndim == 1:         
            self._num_evals += 1
            return self.function(x, *args, **kwargs)

        else:
            raise ValueError('The point/points of evaluation must be a 1D/2D numpy array')
        
    def evaluate_some(self, x, idxs, outputs, *args, **kwargs):
        """
        Evaluates the function at a subset of points in x (according to the indexes idxs),
        and stores the evaluation in an existing array (outputs) at the indexes idxs 
        
        Arguments:
            x {2D numpy array} -- the points of evaluation (the first point is the first row, and so on)
            idxs {1D array-like} -- indexes (integers) of the points of evaluations in x        
            outputs {1D array-like} -- values of the function evaluated at the desired points in x
        
        Returns:
            [1D array-like] -- the function evaluated at the desired subset of x
        """
        n = len(idxs)
        if n > 0:
            self._num_evals += n        
            for idx in idxs:
                outputs[idx] = self.function(x[idx], *args, **kwargs)
        return outputs

    @property
    def info(self):
        """ 
        A description of the function 
        
        Returns:
            str -- description of the function
        """
        s = 'FUNCTION\n'
        s += 'name: {}\nnumber of evaluations: {}\n'.format(self.name, self._num_evals)
        return s


class ObjectiveFunction(Function):
    """
    Wrapper of a mathematical objective function for a single objective optimization problem
    """
    def __init__(self, f):
        """
        Initializes the objective function wrapper
        
        Arguments:
            f {function} -- the (callable) objective function to be wrapped
        """

        Function.__init__(self, f)

    @property
    def info(self):
        """ 
        A description of the objective function 
        
        Returns:
            str -- description of the objective function
        """
        s = 'OBJECTIVE FUNCTION\n'
        s += 'name: {}\nnumber of evaluations: {}\n'.format(self.name, self._num_evals)
        return s


class InequalityConstraint(Function):
    """
    Wrapper of a mathematical inequality constraint for an optimization problem.
    It defines the left side (g) of an inequality of type g(x) <= 0
    """
    
    def __init__(self, g):
        """
        Initializes the inequality constraint wrapper
        
        Arguments:
            g {function} -- the (callable) function
        """
        Function.__init__(self, g)
        self._type = 'inequality'
    
    @property
    def info(self):
        """ 
        A description of the constraint
        
        Returns:
            str -- description of the constraint
        """
        s = 'INEQUALITY CONSTRAINT\n'
        s += 'name: {}\ntype: {}\nnumber of evaluations: {}\n'.format(self.name, self._type, self._num_evals)
        return s


class EqualityConstraint(Function):
    """
    Wrapper of a mathematical equality constraint for an optimization problem.
    It defines the left side (h) of an inequality of type h(x) = 0.
    This equality is converted into an equivalent inequality constraint: abs(h(x)) - tol <= 0

    """
    def __init__(self, h, tol=1e-4):
        """
        Initializes the equality constraint wrapper
        
        Arguments:
            h {function} -- the (callable) function
        
        Keyword Arguments:
            tol {float} -- tolerance for the equivalent inequality constraint (default: {1e-4})
        
        """
        
        self.tol = tol
        
        def hmod(x):
            """
            Evaluates the equality constraint at a given set of points using its equivalent inequality constraint
            
            Arguments:
                x {2D numpy array} -- the points of evaluation
            
            Returns:
                1D numpy array -- the constraint evaluated at every point of evaluation
            """
            return abs(h(x)) - self.tol


        Function.__init__(self, hmod)
        self._type = 'equality'

    @property
    def info(self):
        """ 
        A description of the constraint
        
        Returns:
            str -- description of the constraint
        """
        s = 'EQUALITY CONSTRAINT\n'
        s += 'name: {}\ntype: {}\nnumber of evaluations: {}\n'.format(self.name, self._type, self._num_evals)
        return s


class Constraints:  
    """
    Wrapper of a complete set of constraints for an optimization problem.

    As for the inequality constraints, each function represent the left side (g) of an inequality of type g(x) <= 0.
    
    As for the equality constraints, each function represent the left side (h) of an inequality of type h(x) = 0.
    Each equality constraint is converted into an equivalent inequality constraint: abs(h(x)) - tol <= 0

    Positional and keyworded arguments can be entered for the evaluation of the constraint functions.

    """

    def __init__(self, inequality_constraints, equality_constraints, tol=1e-4):
        """
        Initializes the wrapper of the constraints object
        
        Arguments:
            inequality_constraints {list} -- list of the mathematical functions (callable objects) 
                                             that represent the inequality constraints of the optimization problem.
                                             Each function represent the left side (g) of an inequality of type g(x) <= 0.

            equality_constraints {list} -- list of the mathematical functions (callable objects) 
                                           that represent the equality constraints of the optimization problem
                                           Each function represent the left side (h) of an inequality of type h(x) = 0.

        Keyword Arguments:
            tol {float} -- tolerance for the equivalent inequality constraint (default: {1e-4})
        """
        
        self.num_equality_constraints = len(equality_constraints)
        self.num_inequality_constraints = len(inequality_constraints)
        self.num_constraints = self.num_equality_constraints + self.num_inequality_constraints
        self.tol = float(tol)
        self.constraints = [InequalityConstraint(g) for g in inequality_constraints]  +  [EqualityConstraint(h, self.tol) for h in equality_constraints]


    def evaluate(self, x, *args, **kwargs):
        """
        Evaluates the constraints at a given set of points.

        Positional and keyworded arguments can be entered for the evaluation of the constraint functions.

        
        Arguments:
            x {2D numpy array} -- the points of evaluation (the first point is the first row of the array, and so on)
        
        Returns:
            dict -- evaluations: a 2D numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                 it contains the evaluation of every constraint at every point of evaluation 
                                 (i.e., the component i,j has the evaluation of the jth constraint at the ith evaluation point)

                    violations: a 2D numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                 it contains the constraint violation of every (infeasible) point of evaluation for every constraint

                    feasible: a 1D numpy array of length n. It indicates the feasibility of the evaluation points
                              (i.e., the ith component of the array is a bool (either False or True) that indicates whether the ith point
                              of evaluation is feasible (i.e., has constraint violation equal 0. for every constraint) or not
                    
                    feasible_ratio: float; ratio of the number of feasible points out of the total number of points
                                 
        """ 
        
        evaluations = np.array([c.evaluate(x, *args, **kwargs) for c in self.constraints]).T
        violations = np.maximum(0., evaluations)
        feasible = np.all(evaluations <= 0.0, axis=1)
        feasible_ratio = feasible.sum()/x.shape[0]

        return dict(evaluations=evaluations, violations=violations, feasible=feasible, feasible_ratio=feasible_ratio)

    def evaluate_some(self, x, idxs, constr_evals, *args, **kwargs):
        """
        Evaluates the constraints at a subset of a given set of points x, as defined by idxs.

        Positional and keyworded arguments can be entered for the evaluation of the constraint functions.

        Arguments:
            x {2D numpy array} -- the points of evaluation (the first point is the first row of the array, and so on)

            idxs {list} -- the (integer) indexes of the points of evaluations in x

            constr_evals {2D numpy array} -- a numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                             it contains the evaluation of every constraint at every point of evaluation 
                                             (i.e., the component i,j has the evaluation of the jth constraint at the ith evaluation point)


        Returns:
            dict -- evaluations: a 2D numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                 it contains the evaluation of every constraint at every point of evaluation 
                                 (i.e., the component i,j has the evaluation of the jth constraint at the ith evaluation point)

                    violations: a 2D numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                 it contains the constraint violation of every (infeasible) point of evaluation for every constraint

                    feasible: a 1D numpy array of length n. It indicates the feasibility of the evaluation points
                              (i.e., the ith component of the array is a bool (either False or True) that indicates whether the ith point
                              of evaluation is feasible (i.e., has constraint violation equal 0. for every constraint) or not
                    
                    feasible_ratio: float; ratio of the number of feasible points out of the total number of points                                 
        """ 


        for i, c in enumerate(self.constraints):
            constr_evals[:, i] = c.evaluate_some(x, idxs, constr_evals[:, i], *args, **kwargs)

        violations = np.maximum(0., constr_evals)
        feasible = np.all(constr_evals <= 0.0, axis=1)
        feasible_ratio = feasible.sum()/x.shape[0]

        return dict(evaluations=constr_evals, violations=violations, feasible=feasible, feasible_ratio=feasible_ratio)

    def add_inequality_constraints(self, inequality_constraints):
        """
        Adds a new set of inequality constraints to the constraints object
        
        Arguments:
            inequality_constraints {list} -- list of the mathematical functions (callable objects) 
                                             that represent new inequality constraints for the optimization problem        """

        self.constraints += [InequalityConstraint(g) for g in inequality_constraints]
        n = len(inequality_constraints)
        self.num_inequality_constraints += n
        self.num_constraints += n

    def add_equality_constraints(self, equality_constraints):
        """
        Adds a new set of equality constraints to the constraints object
        
        Arguments:
            equality_constraints {list} -- list of the mathematical functions (callable objects) 
                                           that represent new equality constraints for the optimization problem

        """
        self.constraints += [EqualityConstraint(h) for h in equality_constraints]
        n = len(equality_constraints)
        self.num_equality_constraints += n
        self.num_constraints += n

    @property
    def info(self):
        s = 'CONSTRAINTS\n'
        s += 'Number of inequality constraints: {}\n'.format(self.num_inequality_constraints)
        s += 'Number of equality constraints: {}\n'.format(self.num_equality_constraints)
        s += 'Number of total constraints: {}\n'.format(self.num_constraints)
        s += '\nDescription of constraints:\n\n'
        for c in self.constraints:
            s += c.info + '\n'
        return s


class SelfAdaptivePenaltyFunction:
    """
    This class represents a self adaptive penalty function, used for solving constrained optimization problem, 
    as described in: 

    Tessema, B. and Yen, G. (2006). A Self Adaptive Penalty Function Based Algorithm for Constrained Optimization,
    IEEE Congress on Evolutionary Computation. Vancouver, Canada
    
    """


    def __init__(self, objective_function, inequality_constraints, equality_constraints, tol=1e-4):
        """
        Initializes the self adaptive penalty function
        
        Arguments:
            objective_function {function} -- the objective function to be minimized

            inequality_constraints {list} -- list of the mathematical functions (callable objects) 
                                             that represent the inequality constraints of the optimization problem.
                                             Each function represent the left side (g) of an inequality of type g(x) <= 0.

            equality_constraints {list} -- list of the mathematical functions (callable objects) 
                                           that represent the equality constraints of the optimization problem
                                           Each function represent the left side (h) of an inequality of type h(x) = 0.
        
        Keyword Arguments:
            tol {float} -- tolerance for the equivalent inequality constraint (default: {1e-4})
        """
        self._objective_function = ObjectiveFunction(objective_function)
        self._constraints = Constraints(inequality_constraints, equality_constraints, float(tol))
        
        self.name = self._objective_function.name
        self._num_evals = self._objective_function._num_evals


    def evaluate(self, x, *args, **kwargs):
        """
        Evaluates the self adaptive penalty function at a set of given points.
        
        Positional and keyworded arguments can be entered for the evaluation of the objective function and constraints.

        
        Arguments:
            x {2D numpy array} -- the points of evaluation (the first point is the first row of the array, and so on)
        
        Returns:
            [1D numpy array] -- the evaluation of the self adaptive penalty function at the given points
        """

        # objective function and constraints evaluated at the learners' grades
        func_eval = self._objective_function.evaluate(x, *args, **kwargs)
        constr_eval = self._constraints.evaluate(x, *args, **kwargs)
    
        # constraint violation function evaluated at the (infeasible) learners' grades
        n = x.shape[0]
        m = self._constraints.num_constraints
        v = np.array([0. if constr_eval['feasible'][i] else (1/m)*(constr_eval['violations'][i].sum()/constr_eval['violations'][i].max()) for i in range(n)])                      

        # normalized function evaluations
        fmin = func_eval.min()
        fn = (func_eval - fmin)/(func_eval.max() - fmin)

        # distance values d(x)
        rf = constr_eval['feasible_ratio']
        d = v if rf == 0. else np.sqrt(v**2 + fn**2)      # np.sqrt(v**2 + fn**2) if rf > 0 else v
        
        # penalties values p(x)
        X = np.zeros(n) if rf == 0. else v   # v if rf > 0 else np.zeros(n)
        Y = np.array([0. if constr_eval['feasible'][i] else fn[i] for i in range(n)])
        p = (1 - rf) * X   +  rf * Y

        return d + p
        
    def evaluate_some(self, x, idxs, func_eval, constr_eval, *args, **kwargs):
        """

        Evaluates the self adaptive penalty function at a subset of a given set of points x, as defined by idxs.

        Positional and keyworded arguments can be entered for the evaluation of the constraint functions.


        Arguments:
            x {2D numpy array} -- the points of evaluation (the first point is the first row of the array, and so on)
            idxs {list} -- the (integer) indexes of the points of evaluations in x
            func_eval {1D array-like} -- values of the function evaluated at the desired points in x
            constr_eval {2D numpy array} -- a numpy array with dimensions n x m (n: number of evaluations points, m: number of total constraints)
                                             it contains the evaluation of every constraint at every point of evaluation 
                                             (i.e., the component i,j has the evaluation of the jth constraint at the ith evaluation point)

        """

        # objective function and constraints evaluated at the learners' grades
        func_eval = self._objective_function.evaluate_some(x, idxs, func_eval, *args, **kwargs)
        constr_eval = self._constraints.evaluate_some(x, idxs, constr_eval, *args, **kwargs)
    
        # constraint violation function evaluated at the (infeasible) learners' grades
        n = x.shape[0]
        m = self._constraints.num_constraints
        v = np.array([0. if constr_eval['feasible'][i] else (1/m)*(constr_eval['violations'][i].sum()/constr_eval['violations'][i].max()) for i in range(n)])                      

        # normalized function evaluations
        fmin = func_eval.min()
        fn = (func_eval - fmin)/(func_eval.max() - fmin)

        # distance values d(x)
        rf = constr_eval['feasible_ratio']
        d = v if rf == 0. else np.sqrt(v**2 + fn**2)      # np.sqrt(v**2 + fn**2) if rf > 0 else v
        
        # penalties values p(x)
        X = np.zeros(n) if rf == 0. else v   # v if rf > 0 else np.zeros(n)
        Y = np.array([0. if constr_eval['is_feasible'][i] else fn[i] for i in range(n)])
        p = (1 - rf) * X   +  rf * Y

        return d + p

