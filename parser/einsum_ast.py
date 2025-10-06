from abc import abstractmethod, ABC
from dataclasses import dataclass

# all credit to Willow Ahrens

class EinsumExpr(ABC):
    @abstractmethod
    def get_loops(self) -> set[str]:
        pass

    @abstractmethod
    def run(self, xp, loops, kwargs):
        pass


@dataclass
class Access(EinsumExpr):
    tns: str
    idxs: list[str]

    def get_loops(self) -> set[str]:
        return set(self.idxs)

    def run(self, xp, loops, kwargs):
        assert len(self.idxs) == len(set(self.idxs))
        perm = [self.idxs.index(idx) for idx in loops if idx in self.idxs]
        tns = kwargs[self.tns]
        tns = xp.transpose(tns, perm)
        return xp.expand_dims(
            tns, [i for i in range(len(loops)) if loops[i] not in self.idxs]
        )


@dataclass
class Literal(EinsumExpr):
    value: bool | int | float | complex

    def get_loops(self) -> set[str]:
        return set()

    def run(self, xp, loops, kwargs):
        # Create a scalar array with the same shape as needed
        shape = [1] * len(loops)
        return xp.full(shape, self.value)


@dataclass
class Call(EinsumExpr):
    func: str
    args: list[EinsumExpr]

    def get_loops(self) -> set[str]:
        return set().union(*[arg.get_loops() for arg in self.args])

    def run(self, xp, loops, kwargs):
        if len(self.args) == 1:
            func = getattr(xp, unary_ops[self.func])
        else:
            func = getattr(xp, nary_ops[self.func])
        vals = [arg.run(xp, loops, kwargs) for arg in self.args]
        return func(*vals)


@dataclass
class Einsum:
    arg: EinsumExpr
    op: str | None
    tns: str
    idxs: list[str]

    def run(self, xp, kwargs):
        # This is the main entry point for einsum execution
        loops = self.arg.get_loops()
        assert set(self.idxs).issubset(loops)
        loops = sorted(loops)
        arg = self.arg.run(xp, loops, kwargs)
        axis = tuple(i for i in range(len(loops)) if loops[i] not in self.idxs)
        if self.op is not None:
            op = getattr(xp, reduction_ops.get(self.op, None))
            val = op(arg, axis=axis)
        else:
            assert set(self.idxs) == set(loops)
            val = arg
        dropped = [idx for idx in loops if idx in self.idxs]
        axis = [dropped.index(idx) for idx in self.idxs]
        return xp.transpose(val, axis)

nary_ops = {
    "+": "add",
    "add": "add",
    "-": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "*": "multiply",
    "mul": "multiply",
    "multiply": "multiply",
    "/": "divide",
    "div": "divide",
    "divide": "divide",
    "//": "floor_divide",
    "fld": "floor_divide",
    "floor_divide": "floor_divide",
    "%": "remainder",
    "mod": "remainder",
    "remainder": "remainder",
    "**": "power",
    "pow": "power",
    "power": "power",
    "==": "equal",
    "eq": "equal",
    "equal": "equal",
    "!=": "not_equal",
    "ne": "not_equal",
    "not_equal": "not_equal",
    "<": "less",
    "lt": "less",
    "less": "less",
    "<=": "less_equal",
    "le": "less_equal",
    "less_equal": "less_equal",
    ">": "greater",
    "gt": "greater",
    "greater": "greater",
    ">=": "greater_equal",
    "ge": "greater_equal",
    "greater_equal": "greater_equal",
    "&": "bitwise_and",
    "bitwise_and": "bitwise_and",
    "|": "bitwise_or",
    "bitwise_or": "bitwise_or",
    "^": "bitwise_xor",
    "bitwise_xor": "bitwise_xor",
    "<<": "bitwise_left_shift",
    "lshift": "bitwise_left_shift",
    "bitwise_left_shift": "bitwise_left_shift",
    ">>": "bitwise_right_shift",
    "rshift": "bitwise_right_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    "and": "logical_and",
    "or": "logical_or",
    "not": "logical_not",
    "min": "minimum",
    "max": "maximum",
    "logaddexp": "logaddexp",
}


unary_ops = {
    "+": "positive",
    "pos": "positive",
    "positive": "positive",
    "-": "negative",
    "neg": "negative",
    "negative": "negative",
    "~": "bitwise_invert",
    "invert": "bitwise_invert",
    "bitwise_invert": "bitwise_invert",
    "not": "logical_not",
    "logical_not": "logical_not",
    "abs": "absolute",
    "absolute": "absolute",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log1p": "log1p",
    "log10": "log10",
    "log2": "log2",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
}


reduction_ops = {
    "+": "sum",
    "add": "sum",
    "sum": "sum",
    "*": "prod",
    "mul": "prod",
    "prod": "prod",
    "and": "all",
    "or": "any",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "mean": "mean",
    "std": "std",
    "var": "var",
    "count_nonzero": "count_nonzero",
    # "&": "bitwise_and",
    # "|": "bitwise_or",
    # "^": "bitwise_xor",
}