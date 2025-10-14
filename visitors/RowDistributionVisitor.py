from visitors.EinsumVisitor import EinsumVisitor

class RowDistributionVisitor(EinsumVisitor):
    """
    Track the no of data transfers in a distributed einsum application.
    For now, we assume that num nodes = num dims in split.


    """
    def __init__(self, env, k):
        self._env = env
        self._ownership_dictionary = []
        self._k = k

    def reset(self):
        self.__init__(self._env, self._split_dim)

    def apply_literal(self, node):
        pass

    def apply_einsum(self, node):
        # For now, not marking writes. This will change later.
        pass

    def apply_call(self, node):
        if self._left_subtree is None or self._right_subtree is None:
            raise Exception("Illegal state -- both subtrees must have been assigned!")

        left_cols = self._env[self._left_subtree[1]]
        right_rows = self._env[self._right_subtree[0]]
        assert left_cols == right_rows

        left_transfers = 1
        for dim in self._left_subtree:
            if dim == self._split_dim:
                # Each processor does not need to transfer no. elems - 1 data.
                left_transfers = left_transfers * (self._env[dim] - 1)
            else:
                left_transfers = left_transfers * self._env[dim]
        self._total_transfers += left_transfers

        right_transfers = 1
        for dim in self._right_subtree:
            if dim == self._split_dim:
                # Each processor does not need to transfer no. elems - 1 data.
                right_transfers = right_transfers * (self._env[dim] - 1)
            else:
                right_transfers = right_transfers * self._env[dim]
        self._total_transfers += right_transfers

        self._left_subtree = [self._left_subtree[0], self._right_subtree[1]]
        self._right_subtree = None

    def apply_access(self, node):
        if self._left_subtree is None:
            self._left_subtree = node.idxs
        elif self._right_subtree is None:
            self._right_subtree = node.idxs
        else:
            raise Exception("Illegal state -- one subtree should be unassigned!")

        # Check whether we've already distributed the data from this node.
        # This allows us to repeatedly apply operations to the same node.
        if node.tns in self._cached_tensors:
            return
        self._cached_tensors.add(node.tns)

        # If this element includes the split index, its elements were all stored a-priori.
        if self._split_dim in node.idxs:
            starter_size = 1
            for idx in node.idxs:
                starter_size = starter_size * self._env[idx]
            self._total_transfers += starter_size

    def total_transfers(self):
        return self._total_transfers