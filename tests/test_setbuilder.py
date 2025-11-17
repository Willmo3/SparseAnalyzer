from sparseanalyzer import CountOpsVisitor, parse_einop, RowDistributionVisitor, einsum
from sparseanalyzer import setbuilder as sbn
import pytest

def test_setbuilder():
    A = sbn.Variable("A")
    B = sbn.Variable("B")
    i = sbn.Index("i")
    j = sbn.Index("j")
    k = sbn.Index("k")
    I = sbn.Variable("I")
    J = sbn.Variable("J")
    K = sbn.Variable("K")

    expr = sbn.Union(
        sbn.CoordSet((i, j, k), sbn.IsNonFill(A, (i, j, k))),
        sbn.CoordSet((i, k, j), sbn.And(sbn.IsNonFill(B, (i, k)), sbn.In((j,), sbn.Dimension(j)))),
    )

    simplified = sbn.simplify(expr)

    print("Original expression:")
    print(expr)
    print("Simplified expression:")
    print(simplified)

test_setbuilder()

def test_partition():
    data_partition = sbn.Variable("Π")
    work_partition = sbn.Variable("Φ")
    processor = sbn.Variable("p")

    A = sbn.Variable("A")
    B = sbn.Variable("B")
    C = sbn.Variable("C")

    i = sbn.Index("i")
    j = sbn.Index("j")
    k = sbn.Index("k")

    #C[i, j] = A[i, k] * B[k, j] where A is partitioned with Π over i, computation is partitioned with Φ over i

    has_coords = sbn.CoordSet((i, j), sbn.And(
        sbn.IsNonFill(A, (i, j)),
        sbn.In((i,), sbn.Access(data_partition, (processor,))
    )))

    work_coords = sbn.CoordSet((i, j, k),
        sbn.In((i,), sbn.Access(work_partition, (processor,))
    ))

    A_coords = sbn.CoordSet((i, j), sbn.IsNonFill(A, (i, j)))
    need_coords = sbn.Intersect(
        sbn.Project((i, j), work_coords), A_coords)

    comm_coords = sbn.SetDiff(need_coords, has_coords)

    expr = comm_coords

    simplified = sbn.simplify(expr)

    print()
    print(has_coords)
    print(work_coords)
    print(need_coords)

    # print("Original expression:")
    # print(expr)
    # print("Simplified expression:")
    # print(simplified)

test_partition()