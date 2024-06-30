import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.helpers.wrappers import Function, ObjectiveFunction, InequalityConstraint, EqualityConstraint, Constraints

def test1():
    
    def func(x):
        return x ** 2

    f = Function(func)
    print(f.info)

def test2():
    
    def func(x):
        return np.sum(x)
    
    x = np.random.rand(10,2)
    print('x = \n{}'.format(x))
    
    f = Function(func)
    print('f(x) = \n{}'.format(f.evaluate(x)))
    print(f.info)

def test3():

    def func(x):
        return np.sum(x)

    x = np.random.rand(10,2)
    print('x = \n{}'.format(x))
    
    f = Function(func)
    outputs = f.evaluate(x)
    print('outputs = \n{}'.format(outputs))

    def func2(x):
        return 0.

    f2 = Function(func2)
    print('evaluate some = \n{}'.format(f2.evaluate_some(x, [0,1,-1], outputs)))

def test4():

    def g1(x):
        return x[0] - 3

    def g2(x):
        return x[1] - 3

    def h1(x):
        return x[1] - 1

    c = Constraints(inequality_constraints=[g1, g2], equality_constraints=[h1])
    x = np.random.rand(4, 2)*3.5
    x[0,1] = 1.0
    print('x = \n{}'.format(x))
    constr_eval = c.evaluate(x)
    print('constraints evaluation = \n{}'.format(constr_eval))

    x[0, 0] = 1.0
    x[0, 1] = 1.0
    print('')
    print('')
    print('x = \n{}'.format(x))
    print('')
    print('constraints evaluation after modifying only the first individual')
    print('')
    print('')
    constr_eval = c.evaluate_some(x, [0], constr_eval['evaluations'])
    print('constraints evaluation = \n{}'.format(constr_eval))
    print('')
    print('')
    print('constraints info: \n{}'.format(c.info))

    

if __name__ == "__main__":
    test4()