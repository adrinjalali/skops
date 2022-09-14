import inspect
import json

from sklearn.covariance._graph_lasso import _DictWithDeprecatedKeys
from sklearn.linear_model._sgd_fast import (
    EpsilonInsensitive,
    Hinge,
    Huber,
    Log,
    LossFunction,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
    SquaredLoss,
)
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
    BrayCurtisDistance,
    CanberraDistance,
    ChebyshevDistance,
    DiceDistance,
    EuclideanDistance,
    HammingDistance,
    HaversineDistance,
    JaccardDistance,
    KulsinskiDistance,
    MahalanobisDistance,
    ManhattanDistance,
    MatchingDistance,
    MinkowskiDistance,
    PyFuncDistance,
    RogersTanimotoDistance,
    RussellRaoDistance,
    SEuclideanDistance,
    SokalMichenerDistance,
    SokalSneathDistance,
    WMinkowskiDistance,
)
from sklearn.neighbors import BallTree, KDTree
from sklearn.tree._tree import Tree
from sklearn.utils import Bunch

from ._general import dict_get_instance, dict_get_state
from ._utils import (
    _get_instance,
    _get_state,
    get_instance,
    get_module,
    get_state,
    gettype,
)

ALLOWED_SGD_LOSSES = {
    ModifiedHuber,
    Hinge,
    SquaredHinge,
    Log,
    SquaredLoss,
    Huber,
    EpsilonInsensitive,
    SquaredEpsilonInsensitive,
}
ALLOWED_BINARY_TREES = {BallTree, KDTree}
ALLOWED_DIST_METRICS = {
    BrayCurtisDistance,
    CanberraDistance,
    ChebyshevDistance,
    ChebyshevDistance,
    DiceDistance,
    EuclideanDistance,
    EuclideanDistance,
    HammingDistance,
    HaversineDistance,
    JaccardDistance,
    KulsinskiDistance,
    MahalanobisDistance,
    ManhattanDistance,
    ManhattanDistance,
    ManhattanDistance,
    MatchingDistance,
    MinkowskiDistance,
    MinkowskiDistance,
    PyFuncDistance,
    RogersTanimotoDistance,
    RussellRaoDistance,
    SEuclideanDistance,
    SokalMichenerDistance,
    SokalSneathDistance,
    WMinkowskiDistance,
}


def generic_get_state(obj, dst):
    # This method is for objects which can either be persisted with json, or
    # the ones for which we can get/set attributes through
    # __getstate__/__setstate__ or reading/writing to __dict__.
    try:
        # if we can simply use json, then we're done.
        return json.dumps(obj)
    except Exception:
        pass

    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }

    # __getstate__ takes priority over __dict__, and if non exist, we only save
    # the type of the object, and loading would mean instantiating the object.
    if hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return res

    content = {}
    for key, value in attrs.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        content[key] = _get_state(value, dst)

    res["content"] = content

    return res


def generic_get_instance(state, src):
    try:
        return json.loads(state)
    except Exception:
        pass

    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")

    # Instead of simply constructing the instance, we use __new__, which
    # bypasses the __init__, and then we set the attributes. This solves
    # the issue of required init arguments.
    instance = cls.__new__(cls)

    content = state.get("content", {})
    if not len(content):
        return instance

    attrs = {}
    for key, value in content.items():
        attrs[key] = _get_instance(value, src)

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def reduce_get_state(obj, dst):
    # This method is for objects for which we have to use the __reduce__
    # method to get the state.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }

    # We get the output of __reduce__ and use it to reconstruct the object.
    # For security reasons, we don't save the constructor object returned by
    # __reduce__, and instead use the pre-defined constructor for the object
    # that we know. This avoids having a function such as `eval()` as the
    # "constructor", abused by attackers.
    #
    # We can/should also look into removing __reduce__ from scikit-learn,
    # and that is not impossible. Most objects which use this don't really
    # need it.
    #
    # More info on __reduce__:
    # https://docs.python.org/3/library/pickle.html#object.__reduce__
    #
    # As a good example, this makes Tree object to be serializable.
    reduce = obj.__reduce__()
    res["__reduce__"] = {}
    res["__reduce__"]["args"] = get_state(reduce[1], dst)

    if len(reduce) == 3:
        # reduce includes what's needed for __getstate__ and we don't need to
        # call __getstate__ directly.
        attrs = reduce[2]
    elif hasattr(obj, "__getstate__"):
        attrs = obj.__getstate__()
    elif hasattr(obj, "__dict__"):
        attrs = obj.__dict__
    else:
        return res

    content = {}
    for key, value in attrs.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        content[key] = _get_state(value, dst)

    res["content"] = content

    return res


def reduce_get_instance(state, src, constructor):
    state.pop("__class__")
    state.pop("__module__")

    reduce = state.pop("__reduce__")
    args = get_instance(reduce["args"], src)
    instance = constructor(*args)

    if "content" not in state:
        return instance

    content = state["content"]
    attrs = {}
    for key, value in content.items():
        attrs[key] = _get_instance(value, src)

    if hasattr(instance, "__setstate__"):
        instance.__setstate__(attrs)
    else:
        instance.__dict__.update(attrs)

    return instance


def Tree_get_instance(state, src):
    return reduce_get_instance(state, src, constructor=Tree)


def sgd_loss_get_instance(state, src):
    cls = gettype(state)
    if cls not in ALLOWED_SGD_LOSSES:
        raise TypeError(f"Expected LossFunction, got {cls}")
    return reduce_get_instance(state, src, constructor=cls)


def reduce_based_on_tuple_get_state(obj, dst):
    # For more details, take a look at reduce_get_state. This extra function is
    # needed because __getstate__ of certain sklearn classes returns a tuple
    # instead of a dict, which we thus need to treat differently.
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    reduce = obj.__reduce__()
    res["__reduce__"] = {"state": get_state(reduce[2], dst)}
    return res


def reduce_based_on_tuple_get_instance(state, src, constructor):
    # For more details, take a look at reduce_get_instance. This extra function
    # is needed because __setstate__ of certain sklearn classes require a tuple
    # instead of a dict, which we thus need to treat differently.
    instance = constructor.__new__(constructor)
    reduce = state.pop("__reduce__")
    args = get_instance(reduce["state"], src)
    instance.__setstate__(args)
    return instance


def binary_tree_get_instance(state, src):
    cls = gettype(state)
    if cls not in ALLOWED_BINARY_TREES:
        raise TypeError(f"Expected BinaryTree, got {cls}")
    return reduce_based_on_tuple_get_instance(state, src, constructor=cls)


def distance_metric_get_instance(state, src):
    cls = gettype(state)
    if cls not in ALLOWED_DIST_METRICS:
        raise TypeError(f"Expected DistanceMetric, got {cls}")
    return reduce_based_on_tuple_get_instance(state, src, constructor=cls)


def bunch_get_instance(state, src):
    # Bunch is just a wrapper for dict
    content = dict_get_instance(state, src)
    return Bunch(**content)


def _DictWithDeprecatedKeys_get_state(obj, dst):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": get_module(type(obj)),
    }
    content = {}
    content["main"] = dict_get_state(obj, dst)
    content["_deprecated_key_to_new_key"] = dict_get_state(
        obj._deprecated_key_to_new_key, dst
    )
    res["content"] = content
    return res


def _DictWithDeprecatedKeys_get_instance(state, src):
    # _DictWithDeprecatedKeys is just a wrapper for dict
    content = dict_get_instance(state["content"]["main"], src)
    deprecated_key_to_new_key = dict_get_instance(
        state["content"]["_deprecated_key_to_new_key"], src
    )
    res = _DictWithDeprecatedKeys(**content)
    res._deprecated_key_to_new_key = deprecated_key_to_new_key
    return res


# tuples of type and function that gets the state of that type
GET_STATE_DISPATCH_FUNCTIONS = [
    (LossFunction, reduce_get_state),
    (Tree, reduce_get_state),
    (BallTree, reduce_based_on_tuple_get_state),
    (KDTree, reduce_based_on_tuple_get_state),
    (DistanceMetric, reduce_based_on_tuple_get_state),
    (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_state),
    (object, generic_get_state),
]
# tuples of type and function that creates the instance of that type
GET_INSTANCE_DISPATCH_FUNCTIONS = [
    (LossFunction, sgd_loss_get_instance),
    (Tree, Tree_get_instance),
    (BallTree, binary_tree_get_instance),
    (KDTree, binary_tree_get_instance),
    (DistanceMetric, distance_metric_get_instance),
    (Bunch, bunch_get_instance),
    (_DictWithDeprecatedKeys, _DictWithDeprecatedKeys_get_instance),
    (object, generic_get_instance),
]
