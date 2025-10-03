from einsum_analyzer import CountAssignVisitor
from einsum_parser import parse_einsum

# Matrix addition with transpose
C = parse_einsum("C[i,j] = A[i,j] + B[j,i]")
# Matrix multiplication
D = parse_einsum("D[i,j] += A[i,k] * B[k,j]")
# Min-Plus multiplication with shift
E = parse_einsum("E[i] min= A[i,k] + D[k,j] << 1")

visitor = CountAssignVisitor()
visitor.visit(E)
visitor.generate_iter_order()