from abc import ABC, abstractmethod
from .. import einsum as ein

class EinsumVisitor(ABC):
    def visit(self, node: ein.EinsumNode):
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
            case 'index':
                self.apply_index(node)
            case 'alias':
                self.apply_alias(node)
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
    @abstractmethod
    def apply_index(self, node):
        pass
    @abstractmethod
    def apply_alias(self, node):
        pass