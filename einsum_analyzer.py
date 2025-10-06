from abc import ABC, abstractmethod

class EinsumVisitor(ABC):
    def visit(self, node):
        match type(node).__name__.lower():
            case 'call':
                for arg in node.args:
                    self.visit(arg)
                self.apply_call(node)
            case 'einsum':
                self.visit(node.arg)
                self.apply_einsum(node)
            case 'access':
                self.apply_access(node)
            case 'literal':
                self.apply_literal(node)

    @abstractmethod
    def apply_literal(self, node):
        pass
    @abstractmethod
    def apply_access(self, node):
        pass
    @abstractmethod
    def apply_einsum(self, node):
        pass
    @abstractmethod
    def apply_call(self, node):
        pass

class CountAssignVisitor(EinsumVisitor):
    def __init__(self):
        self._dims = set()
        self._tensor_index_map = dict()

    def apply_literal(self, node):
        pass

    def apply_access(self, node):
        for dim in node.idxs:
            self._dims.add(dim)

        if node.tns not in self._tensor_index_map:
            self._tensor_index_map[node.tns] = set()
        self._tensor_index_map[node.tns].add(tuple(node.idxs))

    def apply_einsum(self, node):
        pass

    def apply_call(self, node):
        pass

    # Post-traversal analysis

    def report_traversals(self):
        for tns in self._tensor_index_map:
            print(tns, self._tensor_index_map[tns])

    def generate_iter_order(self):
        for dim in self._dims:
            print(dim)