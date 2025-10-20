import numpy as np

def and_test(a, b):
    return a & b


def or_test(a, b):
    return a | b


def not_test(a):
    return not a


def ifelse(a, b, c):
    return a if c else b

def promote_type(a, b) -> type:
    a = type(a) if not isinstance(a, type) else a
    b = type(b) if not isinstance(b, type) else b
    if issubclass(a, np.generic) or issubclass(b, np.generic):
        return np.promote_types(a, b).type
    return type(a(False) + b(False))

def promote_min(a, b):
    cast = promote_type(a, b)
    return cast(min(a, b))


def promote_max(a, b):
    cast = promote_type(a, b)
    return max(cast(a), cast(b))


def conjugate(x):
    """
    Computes the complex conjugate of the input number

    Parameters
    ----------
    x: Any
        The input number to compute the complex conjugate of.

    Returns
    ----------
    Any
        The complex conjugate of the input number. If the input is not a complex number,
        it returns the input unchanged.
    """
    if hasattr(x, "conjugate"):
        return x.conjugate()
    return x

class InitWrite:
    """
    InitWrite may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, x, y):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y



def overwrite(x, y):
    """
    overwrite(x, y) returns y always.
    """
    return y



def first_arg(*args):
    """
    Returns the first argument passed to it.
    """
    return args[0] if args else None



def identity(x):
    """
    Returns the input value unchanged.
    """
    return x