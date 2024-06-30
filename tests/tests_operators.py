import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.tlbo.operators import TeachingLearningOperators, clip_array
from backend.core.tlbo.objects import Class, ContinuousGrades, Bounds
from backend.core.helpers.wrappers import Function

def test1():

    def func(x):
        return np.sum(x)
    myfunc = Function(func)

    b = Bounds(lower_bounds=[0,0], upper_bounds=[1,1])
    c =  Class(num_learners=5, grades_bounds=b)
    c.evaluate_learners(myfunc)
    print(c.info)
    print('--------------------------------')

    operators = TeachingLearningOperators()
    
    c = operators.teacher_learners_interaction(c, myfunc)
    print(c.info)
    print('--------------------------------')


def test2():

    def func(x):
        return np.sum(x)
    myfunc = Function(func)

    b = Bounds(lower_bounds=[0,0], upper_bounds=[1,1])
    c =  Class(num_learners=5, grades_bounds=b)
    c.evaluate_learners(myfunc)
    print(c.info)
    print('--------------------------------')

    operators = TeachingLearningOperators()
    c = operators.random_learners_interaction(c, myfunc)
    print(c.info)
    print('--------------------------------')


def test3():

    def func(x):
        return np.sum(x)
    myfunc = Function(func)

    b = Bounds(lower_bounds=[0,0], upper_bounds=[1,1])
    c =  Class(num_learners=5, grades_bounds=b)
    c.evaluate_learners(myfunc)
    print(c.info)
    print('--------------------------------')

    operators = TeachingLearningOperators()
    c = operators.full_learners_interaction(c, myfunc)
    print(c.info)
    print('--------------------------------')


def test4():
    '''
    Example taken from: 
    Venkata Rao (2016), Review of applications of TLBO algorithm and a tutorial for begginers to solve 
    the unconstrained and constrained optimization problems. Decision Science Letters, Growing Science

    '''

    c = Class(num_learners=5, lower_bounds=[-5, -5], upper_bounds=[5, 5])
    X = np.array([[3.22, 0.403],
                  [0.191, 2.289],
                  [3.182, 0.335],
                  [1.66, 4.593],
                  [2.214, 0.867]])
    c.grades.set_as(X)
    
    def himmelbau(x):
        x1 = x[0]
        x2 = x[1]
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

    c.evaluate_learners(function=Function(himmelbau))
    print(c.info)

    print('\n\nHERE THE TEACHER PHASE STARTS...\n\n')

    r = np.array([0.25, 0.43])
    Tf = 1
    teacher_info = c.min_evaluation
    mean_diff = r * (teacher_info['grades'] - Tf*c.grades.mean_values)
    print('\nMean differences:\n{}\n'.format(mean_diff))

    new_grades = c.grades.values + mean_diff
    print('\nNew grades:\n{}\n'.format(new_grades))
    
    new_evaluations = Function(himmelbau).evaluate(new_grades)
    print('\nNew evaluations:\n{}\n'.format(new_evaluations))


    idxs = np.where(new_evaluations < c.evaluations)[0] # for minimization problems
    print('\nIndexes:\n{}\n'.format(idxs))
    c.grades.values[idxs] = new_grades[idxs]
    c.evaluations[idxs] = new_evaluations[idxs]
    print(c.info)


    print('\n\nHERE THE LEARNER PHASE STARTS...\n\n')
    r = np.array([0.47, 0.33])

    new_grades = np.zeros((5,2))

    p = 0
    q = 1
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)


    p = 1
    q = 4
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    p = 2
    q = 0
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)


    p = 3
    q = 4
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    p = 4
    q = 3
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    print('\ngrades after clipping...\n\n')
    new_grades = clip_array(new_grades, [-100, -100], [100, 100])
    print(new_grades)


    print('\nevaluations corresponding to the new grades\n')
    new_evaluations = Function(himmelbau).evaluate(new_grades)

    idxs = np.where(new_evaluations < c.evaluations)[0] # for minimization problems
    c.grades.values[idxs] = new_grades[idxs]
    c.evaluations[idxs] = new_evaluations[idxs]

    print('idxs = ', idxs)
    print('\nclass grades after learner phase\n\n')
    print(c.info)



def test5():
    c = Class(num_learners=5, lower_bounds=[-100, -100], upper_bounds=[100, 100])
    X = np.array([[-55., 36],
                  [0, 41],
                  [96, -86],
                  [-64, 31],
                  [-18, -27]])
    c.grades.set_as(X)
    
    def sphere(x):
        n = len(x)
        sum_ = 0
        for i in range(n):
            sum_ += x[i]**2
        return sum_

    c.evaluate_learners(function=Function(sphere))
    print(c.info)

    print('\n\nHERE THE TEACHER PHASE STARTS...\n\n')

    r = np.array([0.58, 0.49])
    Tf = 1
    teacher_info = c.min_evaluation
    mean_diff = r * (teacher_info['grades'] - Tf*c.grades.mean_values)
    print('\nMean differences:\n{}\n'.format(mean_diff))

    new_grades = c.grades.values + mean_diff
    print('\nNew grades:\n{}\n'.format(new_grades))
    
    new_evaluations = Function(sphere).evaluate(new_grades)
    print('\nNew evaluations:\n{}\n'.format(new_evaluations))


    idxs = np.where(new_evaluations < c.evaluations)[0] # for minimization problems
    print('\nIndexes:\n{}\n'.format(idxs))
    c.grades.values[idxs] = new_grades[idxs]
    c.evaluations[idxs] = new_evaluations[idxs]
    print(c.info)

    # Here the teacher phase ends, and the learner phase starts

    print('\n\nHERE THE LEARNER PHASE STARTS...\n\n')
    r = np.array([0.81, 0.92])

    new_grades = np.zeros((5,2))

    p = 0
    q = 1
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)


    p = 1
    q = 3
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    p = 2
    q = 4
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)


    p = 3
    q = 0
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    p = 4
    q = 2
    print('\np = {}, q = {}\n'.format(p,q))
    if c.evaluations[p] < c.evaluations[q]:  # for minimization problems
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[p,:] - c.grades.values[q,:])
    else:
        new_grades[p,:] = c.grades.values[p,:] + r * (c.grades.values[q,:] - c.grades.values[p,:])
    print(new_grades)

    print('\ngrades after clipping...\n\n')
    new_grades = clip_array(new_grades, [-100, -100], [100, 100])
    print(new_grades)


    print('\nevaluations corresponding to the new grades\n')
    new_evaluations = Function(sphere).evaluate(new_grades)

    idxs = np.where(new_evaluations < c.evaluations)[0] # for minimization problems
    c.grades.values[idxs] = new_grades[idxs]
    c.evaluations[idxs] = new_evaluations[idxs]

    print('idxs = ', idxs)
    print('\nclass grades after learner phase\n\n')
    print(c.info)


    


if __name__ == "__main__":
    test4()