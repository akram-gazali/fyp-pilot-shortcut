"""Small typed program DSL (CLEVR-style).

Types: ObjSet (a set of objects), Int, Bool.
  Scene()                      -> ObjSet
  FilterColor(ObjSet, colour)  -> ObjSet
  FilterShape(ObjSet, shape)   -> ObjSet
  Count(ObjSet)                -> Int
  Exists(ObjSet)               -> Bool
  GreaterThan(Int, Int)        -> Bool
"""
from dataclasses import dataclass

OBJSET, INT, BOOL = "ObjSet", "Int", "Bool"


class ProgramTypeError(TypeError):
    pass


@dataclass(frozen=True)
class Scene:
    def out_type(self):
        return OBJSET


@dataclass(frozen=True)
class FilterColor:
    child: object
    color: str

    def out_type(self):
        _expect(self.child, OBJSET)
        return OBJSET


@dataclass(frozen=True)
class FilterShape:
    child: object
    shape: str

    def out_type(self):
        _expect(self.child, OBJSET)
        return OBJSET


@dataclass(frozen=True)
class Count:
    child: object

    def out_type(self):
        _expect(self.child, OBJSET)
        return INT


@dataclass(frozen=True)
class Exists:
    child: object

    def out_type(self):
        _expect(self.child, OBJSET)
        return BOOL


@dataclass(frozen=True)
class GreaterThan:
    left: object
    right: object

    def out_type(self):
        _expect(self.left, INT)
        _expect(self.right, INT)
        return BOOL


def _expect(node, typ):
    got = node.out_type()
    if got != typ:
        raise ProgramTypeError(f"{type(node).__name__} returns {got}, expected {typ}")


def length(p):
    """Program length = number of nodes (a simple difficulty measure)."""
    kids = [getattr(p, a) for a in ("child", "left", "right") if hasattr(p, a)]
    return 1 + sum(length(k) for k in kids)


def to_dict(p):
    d = {"op": type(p).__name__}
    for a, v in vars(p).items():
        d[a] = to_dict(v) if hasattr(v, "out_type") else v
    return d


_OPS = {c.__name__: c for c in (Scene, FilterColor, FilterShape, Count, Exists, GreaterThan)}


def from_dict(d):
    kwargs = {k: from_dict(v) if isinstance(v, dict) else v for k, v in d.items() if k != "op"}
    p = _OPS[d["op"]](**kwargs)
    p.out_type()  # type-check on load
    return p


def to_str(p):
    """Readable form, e.g. count(filter_color[blue](filter_shape[square](scene)))."""
    if isinstance(p, Scene):
        return "scene"
    if isinstance(p, FilterColor):
        return f"filter_color[{p.color}]({to_str(p.child)})"
    if isinstance(p, FilterShape):
        return f"filter_shape[{p.shape}]({to_str(p.child)})"
    if isinstance(p, Count):
        return f"count({to_str(p.child)})"
    if isinstance(p, Exists):
        return f"exists({to_str(p.child)})"
    return f"greater_than({to_str(p.left)}, {to_str(p.right)})"
