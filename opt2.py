import numpy as np


def distance(vec):
    res = 0
    for v in vec:
        res += v * v
    return res ** 0.5


def goldstein_function(func, grad, x_k, d, max_alpha=1, rho=1e-4, t=2):
    phi_0 = func(x_k)
    dphi_0 = np.dot(grad(x_k), d)
    a = 0
    b = max_alpha
    k = 0
    np.random.seed(42)
    alpha = np.random.rand() * max_alpha
    max_iter = 1000
    while k < max_iter:
        phi = func(x_k + d * alpha)
        if phi_0 + rho * alpha * dphi_0 >= phi:
            if phi_0 + (1 - rho) * alpha * dphi_0 <= phi:
                break
            else:
                a = alpha
                if b >= max_alpha:
                    alpha = t * alpha
                    k += 1
                    continue
        else:
            b = alpha
        alpha = 0.5 * (a + b)
        k += 1
    return alpha


def fletcher_reeves(func, grad, x0, epsilon):
    k = 0
    p, x, alpha, beta = [-grad(x0)], [x0], [], ['_']
    while distance(grad(x[k])) > epsilon:
        print(k, distance(grad(x[k])))
        # alpha += [linesearch(...)]
        alpha += [goldstein_function(func, grad, x[k], p[k])]
        x += [x[k] + alpha[k] * p[k]]
        beta += [distance(grad(x[k + 1])) ** 2 / distance(grad(x[k])) ** 2]
        p += [-grad(x[k + 1]) + beta[k + 1] * p[k]]
        k += 1
    return {
        "x": x[k],
        "f(x)": func(x[k]),
        "iterations": k
    }


def function(x):
    n = len(x)
    res = 0
    for i in range(n - 1):
        res += (x[i] + x[i + 1] - 3) ** 2
    for i in range(1, n - 1):
        res += (x[i] - x[i - 1] + 1) ** 4
    return res


def gradient(x):
    n = len(x)
    res = np.array([0] * n)
    for i in range(n - 1):
        res[i] += 2 * (x[i] + x[i + 1] - 3)
        res[i + 1] += 2 * (x[i] + x[i + 1] - 3)

    for i in range(1, n - 1):
        res[i - 1] += -4 * (x[i] - x[i - 1] + 1) ** 3
        res[i] += 4 * ((x[i] - x[i - 1] + 1) ** 3)
    return np.array(res)


x0 = np.array([1] * 10)
results = fletcher_reeves(function, gradient, x0, epsilon=0.001)
print(results)

'''
Output of the program

Iterations:
0 12.328828005937952
1 4.358898943540674
2 4.358898943540674
3 5.0990195135927845
4 4.47213595499958
5 3.1622776601683795
6 5.916079783099616
7 5.916079783099616
8 4.358898943540674
9 6.164414002968976
10 4.358898943540674
11 2.6457513110645907
12 3.0
13 2.449489742783178
14 1.4142135623730951


{'x': array([2.06835186, 1.85501325, 1.71338237, 1.6219772 , 1.61978841,
       1.60039845, 1.42368632, 1.22239809, 0.68220607, 2.46982356]), 
'f(x)': 7.178294827939238
'iterations': 15}
'''