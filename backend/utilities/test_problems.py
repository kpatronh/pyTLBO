import numpy as np

'''
Test problems for constrained global optimization




Reference:
http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page422.htm

'''


'''
G1
'''

def f1(x):
    # x assumed as a numpy array
    x1 = x[0:4]
    x2 = x[4:]
    return 5*(x1.sum()) - 5*((x1**2).sum()) - x2.sum()

def g1_1(x):
    return 2*x[0] + 2*x[1] + x[9] + x[10] - 10

def g2_1(x):
    return 2*x[0] + 2*x[2] + x[9] + x[11] - 10

def g3_1(x):
    return 2*x[1] + 2*x[2] + x[10] + x[11] - 10

def g4_1(x):
    return -8*x[0] + x[9]

def g5_1(x):
    return -8*x[1] + x[10]

def g6_1(x):
    return -8*x[2] + x[11]

def g7_1(x):
    return -2*x[3] - x[4] + x[9]

def g8_1(x):
    return -2*x[5] - x[6] + x[10]

def g9_1(x):
    return -2*x[7] - x[8] + x[11]

    




'''
G2
'''





'''
G3
'''





'''
G4
'''







