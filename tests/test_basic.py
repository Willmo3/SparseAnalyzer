from sparseanalyzer import CountOpsVisitor, parse_einop, RowDistributionVisitor, einsum

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
count_env = {
    einsum.Index("i"): 2,
    einsum.Index("k"): 3,
    einsum.Index("j"): 4,
}
count_visitor = CountOpsVisitor(count_env)

# -- Test functions -- #

def test_report_reads():
    count_visitor.reset()
    tree = parse_einop("E[i] min= A[i,k] + D[k,j] << 1")
    count_visitor.visit(tree)
    cost = count_visitor.total_reads()
    assert cost == 48

    count_visitor.reset()
    tree = parse_einop("C[i,j] = A[i,j] + B[j,i]")
    count_visitor.visit(tree)
    cost = count_visitor.total_reads()
    assert cost == 32

    count_visitor.reset()
    tree = parse_einop("D[i,j] += A[i,k] * B[k,j]")
    count_visitor.visit(tree)
    cost = count_visitor.total_reads()
    assert cost == 48

def test_report_writes():
    count_visitor.reset()
    tree = parse_einop("E[i] min= A[i,k] + D[k,j] << 1")
    count_visitor.visit(tree)
    cost = count_visitor.total_writes()
    assert cost == 2

    count_visitor.reset()
    tree = parse_einop("C[i,j] = A[i,j] + B[j,i]")
    count_visitor.visit(tree)
    cost = count_visitor.total_writes()
    assert cost == 8

    count_visitor.reset()
    tree = parse_einop("D[i,j] += A[i,k] * B[k,j]")
    count_visitor.visit(tree)
    cost = count_visitor.total_writes()
    assert cost == 8

def test_ownership_dict():
    env = {einsum.Index("i"): 8, einsum.Index("j"): 8, einsum.Index("k"): 8}
    visitor = RowDistributionVisitor(env, 8)
    visitor.reset()

    tree = parse_einop("C[i,k] = A[i,j] + B[j,k]")
    visitor.visit(tree)
    print(visitor.ownership_dictionary())
    print(visitor._total_comms)

def generate_report():
    # Einsum program to multiply a 4x4 matrix w/ 4x4 matrix
    program = "C[i, k] = A[i, j] * B[j, k]"
    size = 32
    env = {einsum.Index("i"): size, einsum.Index("j"): size, einsum.Index("k"): size}

    # Construct analyzer for distribution over two processors.
    visitor = RowDistributionVisitor(env, 4)

    tree = parse_einop(program)
    visitor.visit(tree)
    visitor.report()

# -- Execution -- #
generate_report()