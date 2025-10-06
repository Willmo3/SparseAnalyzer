from lark import Lark, Tree

# all credit to Willow Ahrens

from parser.einsum_ast import EinsumExpr, Einsum, Access, Literal, Call

lark_parser = Lark("""
    %import common.CNAME
    %import common.SIGNED_INT
    %import common.SIGNED_FLOAT
    %ignore " "           // Disregard spaces in text

    start: increment | assign
    increment: access (OP | FUNC_NAME) "=" expr
    assign: access "=" expr

    // Python operator precedence (lowest to highest)
    expr: or_expr
    or_expr: and_expr (OR and_expr)*
    and_expr: not_expr (AND not_expr)*
    not_expr: NOT not_expr | comparison_expr
    comparison_expr: bitwise_or_expr ((EQ | NE | LT | LE | GT | GE) bitwise_or_expr)*
    bitwise_or_expr: bitwise_xor_expr (PIPE bitwise_xor_expr)*
    bitwise_xor_expr: bitwise_and_expr (CARET bitwise_and_expr)*
    bitwise_and_expr: shift_expr (AMPERSAND shift_expr)*
    shift_expr: add_expr ((LSHIFT | RSHIFT) add_expr)*
    add_expr: mul_expr ((PLUS | MINUS) mul_expr)*
    mul_expr: unary_expr ((MUL | DIV | FLOORDIV | MOD) unary_expr)*
    unary_expr: (PLUS | MINUS | TILDE) unary_expr | power_expr
    power_expr: primary (POW unary_expr)?
    primary: call_func | access | literal | "(" expr ")"

    OR: "or"
    AND: "and"
    NOT: "not"
    EQ: "=="
    NE: "!="
    LT: "<"
    LE: "<="
    GT: ">"
    GE: ">="
    PIPE: "|"
    CARET: "^"
    AMPERSAND: "&"
    LSHIFT: "<<"
    RSHIFT: ">>"
    PLUS: "+"
    MINUS: "-"
    MUL: "*"
    DIV: "/"
    FLOORDIV: "//"
    MOD: "%"
    POW: "**"
    TILDE: "~"

    OP: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>"
          | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="

    access: TNS "[" (IDX ",")* IDX? "]"
    call_func: (FUNC_NAME "(" (expr ",")* expr?  ")")
    literal: bool_literal | complex_literal | float_literal | int_literal
    bool_literal: BOOL
    int_literal: SIGNED_INT
    float_literal: SIGNED_FLOAT
    complex_literal: COMPLEX

    BOOL: "True" | "False"
    COMPLEX: (SIGNED_FLOAT | SIGNED_INT) ("j" | "J")
    IDX: CNAME
    TNS: CNAME
    FUNC_NAME: CNAME
""")


def _parse_einsum_expr(t: Tree) -> EinsumExpr:
    match t:
        case Tree(
            "start"
            | "expr"
            | "or_expr"
            | "and_expr"
            | "not_expr"
            | "comparison_expr"
            | "bitwise_or_expr"
            | "bitwise_xor_expr"
            | "bitwise_and_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr"
            | "unary_expr"
            | "power_expr"
            | "primary"
            | "literal",
            [child],
        ):
            return _parse_einsum_expr(child)
        case Tree(
            "or_expr"
            | "and_expr"
            | "bitwise_or_expr"
            | "bitwise_and_expr"
            | "bitwise_xor_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr",
            args,
        ) if len(args) > 1:
            expr = _parse_einsum_expr(args[0])
            for i in range(1, len(args), 2):
                arg = _parse_einsum_expr(args[i + 1])
                expr = Call(args[i].value, [expr, arg])  # type: ignore[union-attr]
            return expr
        case Tree("comparison_expr", args) if len(args) > 1:
            # Handle Python's comparison chaining: a < b < c becomes (a < b) and (b < c)
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            expr = Call(args[1].value, [left, right])  # type: ignore[union-attr]
            for i in range(2, len(args) - 2, 2):
                left = _parse_einsum_expr(args[i])
                right = _parse_einsum_expr(args[i + 2])
                expr = Call("and", [expr, Call(args[i + 1].value, [left, right])])  # type: ignore[union-attr]
            return expr
        case Tree("power_expr", args) if len(args) > 1:
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            return Call(args[1].value, [left, right])  # type: ignore[union-attr]
        case Tree("unary_expr" | "not_expr", [op, arg]):
            return Call(op.value, [_parse_einsum_expr(arg)])  # type: ignore[union-attr]
        case Tree("access", [tns, *idxs]):
            return Access(tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]
        case Tree("bool_literal", [val]):
            return Literal(val.value == "True")  # type: ignore[union-attr]
        case Tree("int_literal", [val]):
            return Literal(int(val.value))  # type: ignore[union-attr]
        case Tree("float_literal", [val]):
            return Literal(float(val.value))  # type: ignore[union-attr]
        case Tree("complex_literal", [val]):
            return Literal(complex(val.value))  # type: ignore[union-attr]
        case Tree("call_func", [func, *args]):
            return Call(func.value, [_parse_einsum_expr(arg) for arg in args])  # type: ignore[union-attr]
        case _:
            raise ValueError(f"Unknown tree structure: {t}")


def parse_einsum(expr: str) -> Einsum:
    tree = lark_parser.parse(expr)
    print(f"Parsed tree: {tree.pretty()}")

    match tree:
        case Tree(
            "start", [Tree("increment", [Tree("access", [tns, *idxs]), op, expr_node])]
        ):
            input_expr = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            return Einsum(input_expr, op.value, tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]

        case Tree("start", [Tree("assign", [Tree("access", [tns, *idxs]), expr_node])]):
            input_expr = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            return Einsum(input_expr, None, tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]

        case _:
            raise ValueError(
                f"Expected top-level assignment or increment, got {tree.data}"
            )