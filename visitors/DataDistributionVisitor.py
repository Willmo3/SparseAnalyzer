from visitors.EinsumVisitor import EinsumVisitor

class RowDistributionVisitor(EinsumVisitor):
    """
    Track the no of data transfers in a distributed einsum application.
    For now, we assume that num nodes = num dims in split.


    """
    def __init__(self, env, k):
        self._env = env
        self._total_comms = 0
        # Which values of which array are owned by each?
        self._ownership_dictionary = dict()
        # On which dimensions do we perform the split?
        self._split_dims = dict()
        # Which arrays have already been totally distributed in memory?
        self._cached_arrs = set()
        self._k = k

    def reset(self):
        self.__init__(self._env, self._k)

    def apply_literal(self, node):
        pass

    def apply_einsum(self, node):
        # For now, not marking writes. This will change later.
        pass

    def apply_call(self, node):
        # For now, only apply to binary functions.
        if not len(node.args) == 2:
            return

        # Binary operations will be performed by the portion of the lhs that this node already owns to the rhs.
        # hence, we only need to distribute the rhs
        access_to_distribute = node.args[1]

        # Only incur a data transfer cost if we haven't already shared this array between all processors.
        if access_to_distribute.tns not in self._cached_arrs:
            self._cached_arrs.add(access_to_distribute.tns)
            split_dim = self._split_dims[access_to_distribute.tns]

            # Number of values allocated to each processor.
            values_per_split = self._env[split_dim] // self._k
            for dim in access_to_distribute.idxs:
                # Processors do not need to transfer the dimensions they already own.
                if not dim == split_dim:
                    values_per_split *= self._env[dim]

            # Each processor must receive each split that it doesn't already own.
            # i.e. k-1 splits.
            transfers_per_processor = values_per_split * (self._k - 1)
            # This communication cost is incurred for each processor.
            self._total_comms += transfers_per_processor * self._k

    def apply_access(self, node):
        # Once ownership has been distributed, we assume needed results are cached.
        if node.tns not in self._ownership_dictionary:
            self._ownership_dictionary[node.tns] = dict()
            # Split each on the first dimension.
            split_dim = node.idxs[0]
            self._split_dims[node.tns] = split_dim

            # Split a single dimension between the processors.
            if not self._env[split_dim] % self._k == 0:
                raise Exception("Not supported!")

            dims_per_node = self._env[split_dim] // self._k
            for i in range(0, self._k):
                # Inclusive lower and upper bounds.
                self._ownership_dictionary[node.tns][i] = range(i * dims_per_node, i * dims_per_node + dims_per_node - 1)


    def ownership_dictionary(self):
        return self._ownership_dictionary