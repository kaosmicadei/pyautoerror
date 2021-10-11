from numbers import Real
import numpy as np

class Dual:
    def __init__(self, value, gradient):
        if isinstance(value, Real) and all([isinstance(x, Real) for x in gradient]):
            self.value = value
            self.gradient = np.array(gradient)
        else:
            raise ValueError

    def __str__(self):
        return "{} + {}".format(self.value, list(self.gradient))
    
    # ADDITION
    def __add__(self, x):
        if isinstance(x, Dual):
            return Dual(self.value + x.value, self.gradient + x.gradient)
        return Dual(x, [0]).__add__(self)
    
    def __radd__(self, x):
        return Dual(x, [0]).__add__(self)

    # SUBTRACTION
    def __sub__(self, x):
        if isinstance(x, Dual):
            return Dual(self.value - x.value, self.gradient - x.gradient)
        return Dual(x, [0]).__sub__(self)
    
    def __rsub__(self, x):
        return Dual(x, [0]).__sub__(self)

    def __neg__(self):
        return Dual(-self.value, -self.gradient)

    # MULTIPLICATION
    def __mul__(self, x):
        if isinstance(x, Dual):
            return Dual(self.value * x.value, self.gradient * x.value + self.value * x.gradient)
        return Dual(x, [0]).__mul__(self)

    def __rmul__(self, x):
        return Dual(x, [0]).__mul__(self)

    def __pow__(self, x):
        if isinstance(x, Dual):
            return Dual(self.value**x.value, x.value * self.value**(x.value-1) * self.gradient + self.value**x.value * np.log(self.value) * x.gradient)
        return self.__pow__(Dual(x, [0]))
    
    def __rpow__(self, x):
        return Dual(x, [0]).__pow__(self)

    # DIVISION
    def __truediv__(self, x):
        if isinstance(x, Dual):
            return Dual(self.value / x.value, (self.gradient * x.value - self.value * x.gradient) / x.value**2)
        return self.__truediv__(Dual(x, [0]))

    def __rtruediv__(self, x):
        return Dual(x, [0]).__truediv__(self)


def as_dual(x):
    '''Create and array of dual number variables.'''
    if isinstance(x, list):
        basis = np.identity(len(x))
        variables = []
        for i, value in enumerate(x):
            variables.append(Dual(value, basis[:,i]))
        return variables
    return Dual(x,[1])

# Basic functions
def sqrt(x):
    if isinstance(x, Dual):
        return Dual(np.sqrt(x.value), x.gradient / np.sqrt(x.value))
    return np.sqrt(x)

def exp(x):
    if isinstance(x, Dual):
        return Dual(np.exp(x.value), np.exp(x.value) * x.gradient)
    return np.exp(x)

def log(x):
    if isinstance(x, Dual):
        return Dual(np.log(x.value), x.gradient / x.value)
    return np.log(x)

def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.value), np.cos(x.value) * x.gradient)
    return np.sin(x)

def sinh(x):
    if isinstance(x, Dual):
        return Dual(np.sinh(x.value), np.cosh(x.value) * x.gradient)
    return np.sinh(x)

def cos(x):
    if isinstance(x, Dual):
        return Dual(np.cos(x.value), -np.sin(x.value) * x.gradient)
    return np.cos(x)

def cosh(x):
    if isinstance(x, Dual):
        return Dual(np.cosh(x.value), np.sinh(x.value) * x.gradient)
    return np.cosh(x)

def tan(x):
    if isinstance(x, Dual):
        return Dual(np.tan(x.value), x.gradient / np.cos(x.value)**2)
    return np.tan(x)

def tanh(x):
    if isinstance(x, Dual):
        return Dual(np.tanh(x.value), x.gradient / np.cosh(x.value)**2)
    return np.tanh(x)