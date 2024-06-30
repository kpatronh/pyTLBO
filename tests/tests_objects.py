import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.tlbo.objects import Bounds, ContinuousGrades, Class
from backend.core.helpers.wrappers import Constraints, Function


def test1():
    b = Bounds(lower_bounds=[0,0,0],
               upper_bounds=[1,1,1])
    
    print(b.info)
    c = b.get_constraints()
    
    for ci in c:
        print(ci.info)

def test2():
    cg = ContinuousGrades(num_learners=10, lower_bounds=[0,0], upper_bounds=[0.5,0.5])
    print(cg.info)
    cg.set_as(values=np.random.rand(10,2)*2)
    print(cg.info)
    cg.clip()
    print(cg.info)
    print(cg.split(num_div=2))

def test3():
    lower_bounds  = [0,0]
    upper_bounds = [1,1]
    cl = Class(num_learners=3, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    print(cl.info)

    def myfunc(x):
        return np.sum(x)

    cl.evaluate_learners(Function(myfunc))
    print('\n\nAfter evaluation of the learners...\n\n')
    print(cl.info)

if __name__ == "__main__":
    test3()     