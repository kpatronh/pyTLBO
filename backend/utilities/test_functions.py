import numpy as np

'''
Benchmark functions for testing optimization algorithms

References:

1) https://www.sfu.ca/~ssurjano/optimization.html

2) http://benchmarkfcns.xyz/fcns

3) https://en.wikipedia.org/wiki/Test_functions_for_optimization

4) http://infinity77.net/global_optimization/test_functions.html

5) https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/index.html

common benchmark functions for metaheuristic evaluation: a review
a literature survey of benchmark functions for global optimization

'''




'''
FUNCTIONS AT REFERENCE 1)
-------------------------
'''

'''
Many local minima
'''

def ackley(xx):
    d = len(xx)
    c = 2*np.pi
    b = 0.2
    a = 20

    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 = sum1 + xi**2
        sum2 = sum2 + np.cos(c*xi)
    

    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)

    return term1 + term2 + a + np.exp(1)

def bukin6(xx):

    x = xx[0]
    y = xx[1]

    term1 = 100 * np.sqrt(np.abs(y - 0.01*x**2))
    term2 = 0.01 * np.abs(x+10)

    return term1 + term2

def cross_in_tray(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1)*np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1**2+x2**2)/np.pi))

    return -0.0001 * (np.abs(fact1*fact2)+1)**0.1

def drop_wave(xx):
    x1 = xx[0]
    x2 = xx[1]


    frac1 = 1 + np.cos(12*np.sqrt(x1**2+x2**2))
    frac2 = 0.5*(x1**2+x2**2) + 2

    return -frac1/frac2

def eggholder(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = -(x2+47) * np.sin(np.sqrt(np.abs(x2+x1/2+47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1-(x2+47))))

    return term1 + term2

def gramacylee(x):
    term1 = np.sin(10*np.pi*x) / (2*x)
    term2 = (x-1)**4

    return term1 + term2    

def griewank(xx):
    d = len(xx)
    sum = 0
    prod = 1

    for ii in range(d):
        xi = xx[ii]
        sum = sum + xi**2/4000
        prod = prod * np.cos(xi/np.sqrt(ii+1))

    y = sum - prod + 1

    return y

def holder_table(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1)*np.cos(x2)
    fact2 = np.exp(np.abs(1. - np.sqrt(x1**2+x2**2)/np.pi))

    y = -abs(fact1*fact2)
    return y 

def levy(xx):
    d = len(xx)

    w = np.empty(d)

    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1)/4
    
    term1 = (np.sin(np.pi*w[0]))**2
    term3 = (w[d-1]-1)**2 * (1+(np.sin(2*np.pi*w[d-1]))**2)

    sum_ = 0
    for ii in range(0, d-1):
        wi = w[ii]
        new = (wi-1)**2 * (1+10*(np.sin(np.pi*wi+1))**2)
        sum_ = sum_ + new

    y = term1 + sum_ + term3

    return y

def levy13(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (np.sin(3*np.pi*x1))**2
    term2 = (x1-1)**2 * (1+(np.sin(3*np.pi*x2))**2)
    term3 = (x2-1)**2 * (1+(np.sin(2*np.pi*x2))**2) 

    return term1 + term2 + term3

def rastrigin(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum = sum + (xi**2 - 10*np.cos(2*np.pi*xi))
    return 10*d + sum

def schaffer2(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = (np.sin(x1**2-x2**2))**2 - 0.5
    fact2 = (1 + 0.001*(x1**2+x2**2))**2
    return 0.5 + fact1/fact2

def schwefel(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum = sum + xi*np.sin(np.sqrt(np.abs(xi)))
    return 418.9829*d - sum

def shubert(xx):
    x1 = xx[0]
    x2 = xx[1]
    sum1 = 0
    sum2 = 0

    for ii in range(1,6):
        new1 = ii * np.cos((ii+1)*x1+ii)
        new2 = ii * np.cos((ii+1)*x2+ii)
        sum1 = sum1 + new1
        sum2 = sum2 + new2

    return sum1 * sum2


'''
Bowl-shaped
'''

def bohachevksy1(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = x1**2
    term2 = 2*x2**2
    term3 = -0.3 * np.cos(3*np.pi*x1)
    term4 = -0.4 * np.cos(4*np.pi*x2)

    return term1 + term2 + term3 + term4 + 0.7

def sphere(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum = sum + xi**2
    return sum

def sum_different_powers(xx):
    d = len(xx)
    sum = 0

    for ii in range(d):
        xi = xx[ii]
        new = (np.abs(xi))**(ii+1)
        sum = sum + new

    return sum

def sum_squares(xx):
    d = len(xx)
    sum = 0
    for ii in range(1,d+1):
        xi = xx[ii-1]
        sum = sum + ii*xi**2
    return sum


'''
Plate-shaped
'''

def booth(xx):
    x1 = xx[0]
    x2 = xx[1]
    term1 = (x1 + 2*x2 - 7)**2
    term2 = (2*x1 + x2 - 5)**2
    return term1 + term2

def matyas(xx):
    return 0.26*(xx[0]**2 + xx[1]**2) - 0.48*xx[0]*xx[1]

def mccormick(xx):
    x1 = xx[0]
    x2 = xx[1]
    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2)**2
    term3 = -1.5*x1
    term4 = 2.5*x2
    return term1 + term2 + term3 + term4 + 1

def zakharov(xx):
    d = len(xx)
    sum1 = 0
    sum2 = 0
    for ii in range(1,d+1):
        xi = xx[ii-1]
        sum1 = sum1 + xi**2
        sum2 = sum2 + 0.5*ii*xi
    return sum1 + sum2**2 + sum2**4;



'''
Valley-shaped
'''

def three_hump_camel(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = 2*x1**2
    term2 = -1.05*x1**4
    term3 = (x1**6) / 6
    term4 = x1*x2  
    term5 = x2**2
    return term1 + term2 + term3 + term4 + term5

def six_hump_camel(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3

def rosenbrock(xx):
    d = len(xx)
    sum = 0
    for ii in range(1,d):
        xi = xx[ii-1]
        xnext = xx[ii]
        new = 100*(xnext-xi**2)**2 + (xi-1)**2
        sum = sum + new
    return sum



'''
Steep ridges/drops
'''

def easom(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = -np.cos(x1)*np.cos(x2)
    fact2 = np.exp(-(x1-np.pi)**2-(x2-np.pi)**2)

    return fact1*fact2

def michalewicz(xx, m=10):
    d = len(xx)
    sum = 0
    for ii in range(1, d+1):
        xi = xx[ii-1]
        new = np.sin(xi) * (np.sin(ii*xi**2/np.pi))**(2*m)
        sum  = sum + new
    return - 1*sum



'''
Other
'''

def beale(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*x2**2)**2
    term3 = (2.625 - x1 + x1*x2**3)**2

    return term1 + term2 + term3

def branin(xx, t=1/(8*np.pi), s=10, r=6, c=5/np.pi, b = 5.1/(4*np.pi**2), a=1):

    x1 = xx[0]
    x2 = xx[1]

    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)

    return term1 + term2 + s

def colville(xx):

    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    x4 = xx[3]

    term1 = 100 * (x1**2-x2)**2
    term2 = (x1-1)**2
    term3 = (x3-1)**2
    term4 = 90 * (x3**2-x4)**2
    term5 = 10.1 * ((x2-1)**2 + (x4-1)**2)
    term6 = 19.8*(x2-1)*(x4-1)

    return term1 + term2 + term3 + term4 + term5 + term6

def forrester(x):
    fact1 = (6*x - 2)**2
    fact2 = np.sin(12*x - 4)

    return fact1 * fact2

def goldstein_price(xx):

    x1 = xx[0]
    x2 = xx[1]

    fact1a = (x1 + x2 + 1)**2
    fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    fact1 = 1 + fact1a*fact1b

    fact2a = (2*x1 - 3*x2)**2
    fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    fact2 = 30 + fact2a*fact2b

    return fact1*fact2







'''
1 Dimensional
'''

def gramacyleefcn(x):
    n = x.ndim

    if n != 1:
        raise ValueError('Gramacy & Lee function is only defined on a 1-D space')

    return (np.sin(10 * np.pi * x) / (2 * x)) + ((x - 1) ** 4)



'''
2 Dimensional
'''

def ackleyn2fcn(x):
    n = x.ndim

    if n != 2:
        raise ValueError('Ackley N. 2 function is only defined on a 2D space')

    X = x[0]
    Y = x[1]
    
    return -200 * np.exp(-0.02 * np.sqrt((X**2) + (Y**2)))

def himmelblau(x):
    n = x.ndim
    if n != 2:
        raise ValueError("Himmelblau's function is only defined on a 2-D space")
    X = x[0]
    Y = x[1]
    return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

'''
3 Dimensional
'''
