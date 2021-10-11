import autoerror.dualnumber as dual
from autoerror.dualnumber import as_dual

def uncert(f, sigmas):
    x = f.gradient * sigmas
    return f.value, dual.sqrt(x.dot(x))