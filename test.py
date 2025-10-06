from einsum_analyzer import CountAssignVisitor
from einsum_parser import parse_einsum

"""
Examples considered:

Matrix addition with transpose:
C[i,j] = A[i,j] + B[j,i]")

Matrix multiplication
D[i,j] += A[i,k] * B[k,j]")

Min-Plus multiplication with shift
E[i] min= A[i,k] + D[k,j] << 1
"""

# -- Shared State -- #

# Dimensions mapped to their size.
env = {
    "i": 2,
    "k": 3,
    "j": 4,
}
visitor = CountAssignVisitor(env)

# -- Test functions -- #

def test_report_reads():
    visitor.reset()
    tree = parse_einsum("E[i] min= A[i,k] + D[k,j] << 1")
    visitor.visit(tree)
    cost = visitor.total_reads()
    assert cost == 48

    visitor.reset()
    tree = parse_einsum("C[i,j] = A[i,j] + B[j,i]")
    visitor.visit(tree)
    cost = visitor.total_reads()
    assert cost == 32

    visitor.reset()
    tree = parse_einsum("D[i,j] += A[i,k] * B[k,j]")
    visitor.visit(tree)
    cost = visitor.total_reads()
    assert cost == 48

def test_report_writes():
    visitor.reset()
    tree = parse_einsum("E[i] min= A[i,k] + D[k,j] << 1")
    visitor.visit(tree)
    cost = visitor.total_writes()
    assert cost == 2

    visitor.reset()
    tree = parse_einsum("C[i,j] = A[i,j] + B[j,i]")
    visitor.visit(tree)
    cost = visitor.total_writes()
    assert cost == 8

    visitor.reset()
    tree = parse_einsum("D[i,j] += A[i,k] * B[k,j]")
    visitor.visit(tree)
    cost = visitor.total_writes()
    assert cost == 8

# -- Execution -- #

test_report_reads()
test_report_writes()