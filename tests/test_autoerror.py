from autoerror.autoerror import *

x = as_dual(3)
sigmas = [0.01]
print(uncert(3*x+1, sigmas))

def expr(x,y):
    return x**2 + dual.cos(y)
xs = as_dual([2,3.14])
sigmas = [0.2, 0.1]
print(uncert(expr(*xs), sigmas))

def entropy(probs):
    s = 0
    for p in probs:
        s -= p * dual.log(p)
    return s
ps = as_dual([2/3, 2/9, 1/9])
sigmas = [0.1, 0.12, 0.08]
print(uncert(entropy(ps), sigmas))