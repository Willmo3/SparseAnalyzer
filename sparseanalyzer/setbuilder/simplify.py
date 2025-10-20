from . import nodes as sbn
from .nodes import SetBuilderExpr, SetBuilderNode
from ..symbolic import PostWalk, Fixpoint

def simplify_node(expr: SetBuilderNode):
    match expr:
        case sbn.Intersect(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == (idxs2):
            rename_dict = dict(zip(idxs2, idxs1))
            def rename(ex):
                match ex:
                    case sbn.Index(name) if name in idxs1:
                        return sbn.Index(rename_dict[name])
                    case _:
                        return None
            pred3 = PostWalk(rename)(pred2)
            return sbn.CoordSet(idxs1, sbn.And(pred1, pred3))
        case sbn.SetDiff(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == (idxs2):
            rename_dict = dict(zip(idxs2, idxs1))
            def rename(ex):
                match ex:
                    case sbn.Index(name) if name in idxs1:
                        return sbn.Index(rename_dict[name])
                    case _:
                        return None
            pred3 = PostWalk(rename)(pred2)
            return sbn.CoordSet(idxs1, sbn.And(pred1, sbn.Not(pred3)))
        case sbn.Union(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == (idxs2):
            rename_dict = dict(zip(idxs2, idxs1))
            def rename(ex):
                match ex:
                    case sbn.Index(name) if name in idxs1:
                        return sbn.Index(rename_dict[name])
                    case _:
                        return None
            pred3 = PostWalk(rename)(pred2)
            return sbn.CoordSet(idxs1, sbn.Or(pred1, pred3))
        case _:
            return None

def simplify(prgm: SetBuilderNode) -> SetBuilderNode:
    rw = Fixpoint(PostWalk(simplify_node))
    return rw(prgm)
