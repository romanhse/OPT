import numpy as np
import numpy.linalg as ln
import scipy as sp


def function(x):
    n = len(x)
    res = 0
    for i in range(n - 1):
        res += (x[i]**2 - 2)**2
    prom = 0
    for i in range(n):
        prom += x[i]**2
    prom -= 0.5
    res += prom ** 2
    return res


# Derivative
def derivative(x):
    n = len(x)
    res = [0] * n
    for i in range(n - 1):
        res[i] += 2*(x[i]**2 - 2)*2*x[i]
    prom = 0
    for i in range(n):
        prom += x[i] ** 2
    prom -= 0.5
    for i in range(n):
        res[i] += 2*prom*2*x[i]
    return np.array(res)


def bfgs_method(f, f_der, x0, epsi=0.001):

    k = 0
    gfk = f_der(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0

    while ln.norm(gfk) > epsi:

        pk = -np.dot(Hk, gfk)

        line_search = sp.optimize.line_search(f, derivative, xk, pk)
        alpha_k = line_search[0]

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = f_der(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1

        k += 1

        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    return (xk, k)


result, k = bfgs_method(function, derivative, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


print(f'Final Result (best point): {result}')
print(f'Iteration Count: {k}')
print(f'Value of the function at the point: {function(result)}')

'''
The output of the program:

Result of BFGS method:
Final Result (best point): [-5.00001514e-01 -5.00001514e-01 -5.00001514e-01 -5.00001514e-01
 -5.00001514e-01 -5.00001514e-01 -5.00001514e-01 -5.00001514e-01
 -5.00001514e-01 -1.43849011e-06]
Iteration Count: 7
Value of the function at the point: 30.625000000213618
'''

