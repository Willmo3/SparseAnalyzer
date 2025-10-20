from . import nodes as sbn
from .nodes import SetBuilderExpr, SetBuilderNode
from ..symbolic import PostWalk, Fixpoint

def simplify_node(expr: SetBuilderNode):
    def renamer(idxs1, idxs2, pred):
        rename_dict = dict(zip(idxs2, idxs1))
        backward_dict = dict(zip(idxs1, idxs2))
        def rename(ex):
            match ex:
                case sbn.Index(_) as idx if idx in idxs1:
                    return sbn.Index(rename_dict[idx])
                case _:
                    return ex
        def rename_back(ex):
            match ex:
                case sbn.Dimension(sbn.Index(_) as idx) if idx in idxs2:
                    return sbn.Index(backward_dict[idx])
                case _:
                    return ex
        pred2 = PostWalk(rename)(pred)
        pred2 = PostWalk(rename_back)(pred2)
        return pred2
    match expr:
        case sbn.Intersect(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == len(idxs2):
            pred3 = renamer(idxs1, idxs2, pred2)
            return sbn.CoordSet(idxs1, sbn.And(pred1, pred3))
        case sbn.SetDiff(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == len(idxs2):
            pred3 = renamer(idxs1, idxs2, pred2)
            return sbn.CoordSet(idxs1, sbn.And(pred1, sbn.Not(pred3)))
        case sbn.Union(sbn.CoordSet(idxs1, pred1), sbn.CoordSet(idxs2, pred2)) if len(idxs1) == len(idxs2):
            pred3 = renamer(idxs1, idxs2, pred2)
            return sbn.CoordSet(idxs1, sbn.Or(pred1, pred3))
        case _:
            return expr

def simplify(prgm: SetBuilderNode) -> SetBuilderNode:
    rw = Fixpoint(PostWalk(simplify_node))
    return rw(prgm)
