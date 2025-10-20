import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, cast

from ..operators import (
    overwrite,
    promote_max,
    promote_min,
)
from ..symbolic import Context, Term, TermTree


class SetBuilderNode(Term):
    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *children: Term) -> Self:
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        return cls(*children)

    def __str__(self):
        """Returns a string representation of the node."""
        ctx = SetBuilderPrinterContext()
        res = ctx(self)
        return res if res is not None else ctx.emit()


class SetBuilderTree(SetBuilderNode, TermTree):
    """
    SetBuilderExpr

    Represents a pointwise expression in the SetBuilder IR
    """

    @property
    @abstractmethod
    def children(self) -> list[SetBuilderNode]:  # type: ignore[override]
        ...


class SetBuilderExpr(SetBuilderNode, ABC):
    pass

@dataclass(eq=True, frozen=True)
class Literal(SetBuilderExpr):
    """
    Literal
    """

    val: Any

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.val == other.val

    def get_idxs(self) -> set["Index"]:
        return set()


@dataclass(eq=True, frozen=True)
class Index(SetBuilderExpr):
    """
    Represents a  AST expression for an index named `name`.

    Attributes:
        name: The name of the index.
    """

    name: str

    def get_idxs(self) -> set["Index"]:
        return {self}


@dataclass(eq=True, frozen=True)
class CoordSet(SetBuilderExpr, SetBuilderTree):
    """
    CoordSet

    {(idxs...) | pred}

    Attributes:
        idxs: The indices at which to access the tensor.
        pred: The predicate for which indices are in the set.
    """
    idxs: tuple[Index, ...]  # (Field('i'), Field('j'))
    pred: SetBuilderExpr

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is tns, rest are indices
        if len(children) < 1:
            raise ValueError("Access expects at least 1 child")
        idxs = cast(tuple[SetBuilderExpr, ...], children[:-1])
        pred = cast(SetBuilderExpr, children[-1])
        return cls(idxs, pred)

    @property
    def children(self):
        return [*self.idxs, self.pred]


@dataclass(eq=True, frozen=True)
class Variable(SetBuilderExpr):
    """
    Represents a  AST expression for a variable named `name`.

    Attributes:
        name: The name of the variable.
    """

    name: str

    def get_idxs(self) -> set["Index"]:
        return {self}

@dataclass(eq=True, frozen=True)
class LessThan(SetBuilderExpr, SetBuilderTree):
    """
    LessThan

    Return true if x < y

    Attributes:
        x: The first variable to compare.
        y: The second variable to compare.
    """

    x: SetBuilderExpr
    y: SetBuilderExpr

    @property
    def children(self):
        return [self.x, self.y]

@dataclass(eq=True, frozen=True)
class GreaterThan(SetBuilderExpr, SetBuilderTree):
    """
    GreaterThan

    Return true if x > y

    Attributes:
        x: The first value to compare.
        y: The second value to compare.
    """

    x: SetBuilderExpr
    y: SetBuilderExpr

    @property
    def children(self):
        return [self.x, self.y]

@dataclass(eq=True, frozen=True)
class And(SetBuilderExpr, SetBuilderTree):
    """
    And

    Return true if x and y

    Attributes:
        x: The first value to consider.
        y: The second value to consider.
    """

    x: SetBuilderExpr
    y: SetBuilderExpr

    @property
    def children(self):
        return [self.x, self.y]

@dataclass(eq=True, frozen=True)
class Or(SetBuilderExpr, SetBuilderTree):
    """
    Or

    Return true if x or y

    Attributes:
        x: The first value to consider.
        y: The second value to consider.
    """

    x: SetBuilderExpr
    y: SetBuilderExpr

    @property
    def children(self):
        return [self.x, self.y]

@dataclass(eq=True, frozen=True)
class Not(SetBuilderExpr, SetBuilderTree):
    """
    Not

    Return true if not x

    Attributes:
        x: The value to consider.
    """

    x: SetBuilderExpr

    @property
    def children(self):
        return [self.x]

@dataclass(eq=True, frozen=True)
class IsNonFill(SetBuilderExpr, SetBuilderTree):
    """
    IsNonFill

    Return true if a[i, j] is non-fill

    Attributes:
        tensor: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    tns: SetBuilderExpr
    idxs: tuple[SetBuilderExpr, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is tns, rest are indices
        if len(children) < 1:
            raise ValueError("Access expects at least 1 child")
        tns = cast(SetBuilderExpr, children[0])
        idxs = cast(tuple[SetBuilderExpr, ...], children[1:])
        return cls(tns, tuple(idxs))

    @property
    def children(self):
        return [self.tns, *self.idxs]

@dataclass(eq=True, frozen=True)
class Access(SetBuilderExpr, SetBuilderTree):
    """
    Access

    Return the value of `tns[idxs]`

    Attributes:
        tns: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    tns: SetBuilderExpr
    idxs: tuple[SetBuilderExpr, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is tns, rest are indices
        if len(children) < 1:
            raise ValueError("Access expects at least 1 child")
        tns = cast(SetBuilderExpr, children[0])
        idxs = cast(tuple[SetBuilderExpr, ...], children[1:])
        return cls(tns, tuple(idxs))

    @property
    def children(self):
        return [self.tns, *self.idxs]

@dataclass(eq=True, frozen=True)
class Union(SetBuilderExpr, SetBuilderTree):
    """
    Union

    Attributes:
        left: The left set.
        right: The right set.
    """

    left: SetBuilderExpr
    right: SetBuilderExpr

    @property
    def children(self):
        return [self.left, self.right]

@dataclass(eq=True, frozen=True)
class Intersect(SetBuilderExpr, SetBuilderTree):
    """
    Intersect

    Attributes:
        left: The left set.
        right: The right set.
    """

    left: SetBuilderExpr
    right: SetBuilderExpr

    @property
    def children(self):
        return [self.left, self.right]

@dataclass(eq=True, frozen=True)
class SetDiff(SetBuilderExpr, SetBuilderTree):
    """
    SetDiff

    Attributes:
        left: The left set.
        right: The right set.
    """

    left: SetBuilderExpr
    right: SetBuilderExpr

    @property
    def children(self):
        return [self.left, self.right]

class ForAll(SetBuilderExpr, SetBuilderTree):
    """
    ForAll

    Attributes:
        idx: The index to quantify over.
        body: The body of the quantifier.
    """

    idx: Index
    body: SetBuilderExpr

    @property
    def children(self):
        return [self.idx, self.body]

class Plus(SetBuilderExpr, SetBuilderTree):
    """
    Plus

    Attributes:
        left: The left expression.
        right: The right expression.
    """

    left: SetBuilderExpr
    right: SetBuilderExpr

    @property
    def children(self):
        return [self.left, self.right]

@dataclass(eq=True, frozen=True)
class Exists(SetBuilderExpr, SetBuilderTree):
    """
    Exists

    Attributes:
        idx: The index to quantify over.
        body: The body of the quantifier.
    """
    
    idx: Index
    body: SetBuilderExpr

    @property
    def children(self):
        return [self.idx, self.body]


class Dimension(SetBuilderExpr, SetBuilderTree):
    """
    Dimension

    Represents the range of feasible values for an index.

    Attributes:
        idx: The index whose dimension to return.
    """
    
    idx: Any

    @property
    def children(self):
        return [self.idx]

@dataclass(eq=True, frozen=True)
class Cardinality(SetBuilderExpr, SetBuilderTree):
    """
    Cardinality

    Attributes:
        set_expr: The set expression to compute the cardinality of.
    """
    
    set_expr: SetBuilderExpr

    @property
    def children(self):
        return [self.set_expr]

class SetBuilderPrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self):
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: SetBuilderNode):
        feed = self.feed
        match prgm:
            case Literal(val):
                return str(val)
            case Index(name):
                return str(name)
            case Variable(name):
                return str(name)
            case CoordSet(idxs, pred):
                idx_str = ', '.join(self(idx) for idx in idxs)
                return f"{{({idx_str}) | {self(pred)}}}"
            case LessThan(x, y):
                return f"({self(x)} < {self(y)})"
            case GreaterThan(x, y):
                return f"({self(x)} > {self(y)})"
            case And(x, y):
                return f"({self(x)} ∧ {self(y)})"
            case Or(x, y):
                return f"({self(x)} ∨ {self(y)})"
            case Not(x):
                return f"¬({self(x)})"
            case IsNonFill(tns, idxs):
                idx_str = ', '.join(self(idx) for idx in idxs)
                return f"{self(tns)}[[{idx_str}]]"
            case Access(tns, idxs):
                idx_str = ', '.join(self(idx) for idx in idxs)
                return f"{self(tns)}[{idx_str}]"
            case Union(left, right):
                return f"({self(left)} ∪ {self(right)})"
            case Intersect(left, right):
                return f"({self(left)} ∩ {self(right)})"
            case SetDiff(left, right):
                return f"({self(left)} \\ {self(right)})"
            case ForAll(idx, body):
                return f"∀ {self(idx)}. ({self(body)})"
            case Exists(idx, body):
                return f"∃ {self(idx)}. ({self(body)})"
            case _:
                raise ValueError(f"Unknown expression type: {type(prgm)}")
