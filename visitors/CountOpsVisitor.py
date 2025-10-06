from visitors.EinsumVisitor import EinsumVisitor

class CountOpsVisitor(EinsumVisitor):
    """
    :param env Mapping of dimension titles to their size.
    Tensors with dimensions of matching titles are assumed to match in size.
    """
    def __init__(self, env):
        self._env = env

        self._read_dims = set()
        self._read_tensor_index_map = dict()

        # Since we will only write to a single dimension, unnecessary to track more than just the dimensions we use.
        self._write_dims = set()

    def reset(self):
        self.__init__(self._env)

    # Visitor methods

    def apply_literal(self, node):
        pass
    def apply_call(self, node):
        pass

    def apply_access(self, node):
        for dim in node.idxs:
            self._read_dims.add(dim)

        if node.tns not in self._read_tensor_index_map:
            self._read_tensor_index_map[node.tns] = set()

        for idx in node.idxs:
            self._read_tensor_index_map[node.tns].add(idx)

    def apply_einsum(self, node):
        for dim in node.idxs:
            self._write_dims.add(dim)


    # Post-traversal analysis

    def report_traversals(self):
        for tns in self._read_tensor_index_map:
            print(tns, self._read_tensor_index_map[tns])

    def report_example_iter_order(self):
        for dim in self._read_dims:
            print(dim)

    """
    Report the total reads a matrix operation will require in a given env.
    
    This is more complex than simply multiplying the size of the dimensions, because some dimensions are iterated over multiple times.
    """
    def total_reads(self):
        factors = {dim: 0 for dim in self._read_dims}
        for tns in self._read_tensor_index_map:
            for idx in self._read_tensor_index_map[tns]:
                factors[idx] += 1

        cost = 1
        for dim in self._read_dims:
            cost = cost * factors[dim] * self._env[dim]

        return cost

    """
    Report the total writes a matrix operation will require in a given env.
    
    Since einsum requires only a single matrix be written to, we can report this by multiplying width * height of dim.
    """
    def total_writes(self):
        cost = 1
        for dim in self._write_dims:
            cost = cost * self._env[dim]

        return cost

    # Data transfer analysis




        pass
