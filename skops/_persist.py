from __future__ import annotations

import importlib
import inspect
import io
import json
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

import skops


def _import_obj(module, cls_or_func):
    return getattr(importlib.import_module(module), cls_or_func)


def gettype(state):
    if "__module__" in state and "__class__" in state:
        return _import_obj(state["__module__"], state["__class__"])
    return None


def BaseEstimator_get_state(obj, dst, methods):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    for key, value in obj.__dict__.items():
        if isinstance(getattr(type(obj), key, None), property):
            continue
        try:
            res[key] = get_state_method(value, methods)(value, dst, methods)
        except TypeError:
            res[key] = json.dumps(value)

    return res


def BaseEstimator_get_instance(state, src, methods):
    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")
    instance = cls()
    for key, value in state.items():
        if isinstance(value, dict):
            setattr(
                instance, key, get_instance_method(value, methods)(value, src, methods)
            )
        else:
            setattr(instance, key, json.loads(value))
    return instance


def Pipeline_get_instance(state, src, methods):
    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")
    steps = state.pop("steps")
    steps = get_instance_method(steps, methods)(steps, src, methods)
    instance = cls(steps)
    for key, value in state.items():
        if isinstance(value, dict):
            setattr(
                instance, key, get_instance_method(value, methods)(value, src, methods)
            )
        else:
            setattr(instance, key, json.loads(value))
    return instance


def FeatureUnion_get_instance(state, src, methods):
    cls = gettype(state)
    state.pop("__class__")
    state.pop("__module__")
    transformer_list = state.pop("transformer_list")
    steps = get_instance_method(transformer_list, methods)(
        transformer_list, src, methods
    )
    instance = cls(steps)
    for key, value in state.items():
        if isinstance(value, dict):
            setattr(
                instance, key, get_instance_method(value, methods)(value, src, methods)
            )
        else:
            setattr(instance, key, json.loads(value))
    return instance


def ndarray_get_state(obj, dst, methods):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }

    try:
        f_name = f"{uuid4()}.npy"
        with open(Path(dst) / f_name, "wb") as f:
            np.save(f, obj, allow_pickle=False)
            res["type"] = "numpy"
            res["file"] = f_name
    except ValueError:
        # object arrays cannot be saved with allow_pickle=False, therefore we
        # convert them to a list and store them as a json file.
        f_name = f"{uuid4()}.json"
        with open(Path(dst) / f_name, "w") as f:
            f.write(json.dumps(obj.tolist()))
            res["type"] = "json"
            res["file"] = f_name

    return res


def ndarray_get_instance(state, src, methods):
    if state["type"] == "numpy":
        return np.load(io.BytesIO(src.read(state["file"])), allow_pickle=False)
    return np.array(json.loads(src.read(state["file"])))


def dict_get_state(obj, dst, methods):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = {}
    for key, value in obj.items():
        if np.isscalar(key) and hasattr(key, "item"):
            # convert numpy value to python object
            key = key.item()
        try:
            content[key] = get_state_method(value, methods)(value, dst, methods)
        except TypeError:
            content[key] = json.dumps(value)
    res["content"] = content
    return res


def dict_get_instance(state, src, methods):
    state.pop("__class__")
    state.pop("__module__")
    content = {}
    for key, value in state["content"].items():
        if isinstance(value, dict):
            content[key] = get_instance_method(value, methods)(value, src, methods)
        else:
            content[key] = json.loads(value)
    return content


def list_get_state(obj, dst, methods):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = []
    for value in obj:
        try:
            content.append(get_state_method(value, methods)(value, dst, methods))
        except TypeError:
            content.append(json.dumps(value))
    res["content"] = content
    return res


def list_get_instance(state, src, methods):
    state.pop("__class__")
    state.pop("__module__")
    content = []
    for value in state["content"]:
        if gettype(value):
            content.append(get_instance_method(value, methods)(value, src, methods))
        else:
            content.append(json.loads(value))
    return content


def tuple_get_state(obj, dst, methods):
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
    }
    content = ()
    for value in obj:
        try:
            content += (get_state_method(value, methods)(value, dst, methods),)
        except TypeError:
            content += (json.dumps(value),)
    res["content"] = content
    return res


def tuple_get_instance(state, src, methods):
    state.pop("__class__")
    state.pop("__module__")
    content = ()
    for value in state["content"]:
        if gettype(value):
            content += (get_instance_method(value, methods)(value, src, methods),)
        else:
            content += (json.loads(value),)
    return content


def function_get_state(obj, dst, methods):
    if isinstance(obj, partial):
        raise TypeError("partial function are not supported yet")
    res = {
        "__class__": obj.__class__.__name__,
        "__module__": inspect.getmodule(type(obj)).__name__,
        "__content__": obj.__name__,
    }
    return res


def function_get_instance(obj, src, methods):
    loaded = _import_obj(obj["__module__"], obj["__content__"])
    return loaded


# A dictionary mapping types to their corresponding persistance method.
GET_STATE_METHODS = {
    BaseEstimator: BaseEstimator_get_state,
    FunctionType: function_get_state,
    np.ufunc: function_get_state,
    np.ndarray: ndarray_get_state,
    np.generic: ndarray_get_state,
    dict: dict_get_state,
    list: list_get_state,
    tuple: tuple_get_state,
}

SET_STATE_METHODS = {
    Pipeline: Pipeline_get_instance,
    FeatureUnion: FeatureUnion_get_instance,
    BaseEstimator: BaseEstimator_get_instance,
    FunctionType: function_get_instance,
    np.ufunc: function_get_instance,
    np.ndarray: ndarray_get_instance,
    np.generic: ndarray_get_instance,
    dict: dict_get_instance,
    list: list_get_instance,
    tuple: tuple_get_instance,
}


def get_state_method(obj, methods):
    # we go through the MRO and find the first class for which we have a method
    # to save the object. For instance, we might have a function for
    # BaseEstimator used for most classes, but a specialized one for Pipeline.
    for cls in type(obj).mro():
        if cls in methods:
            return methods[cls]

    raise TypeError(f"Can't serialize {type(obj)}")


def get_instance_method(state, methods):
    cls_ = gettype(state)
    for cls in cls_.mro():
        if cls in methods:
            return methods[cls]

    raise TypeError(f"Can't deserialize {type(state)}")


def save(obj, file, protocol=0, sklearn_version=None):
    persister = Persister(protocol=protocol, sklearn_version=sklearn_version)
    persister.save(obj, file)


def load(file, protocol=0):
    persister = Persister(protocol=protocol)
    return persister.load(file)


class Persister:
    """Wrapper class that calls save and load functions

    Makes it possible to inject extrra get and set methods.

    """

    def __init__(
        self,
        protocol: int = 0,
        sklearn_version: str | None = None,
        extra_get_state_methods: dict[str, Any] | None = None,
        extra_set_state_methods: dict[str, Any] | None = None,
    ) -> None:
        # this is not being used at the moment
        self.protocol = protocol
        if sklearn_version is None:
            self.sklearn_version = sklearn.__version__
        else:
            self.sklearn_version = sklearn_version
        self.skops_version = skops.__version__

        self.extra_get_state_methods = extra_get_state_methods or {}
        self.extra_set_state_methods = extra_set_state_methods or {}

    def save(self, obj: Any, file: str) -> None:
        methods = self.extra_get_state_methods.copy()
        methods.update(GET_STATE_METHODS)
        with tempfile.TemporaryDirectory() as dst:
            with open(Path(dst) / "schema.json", "w") as f:
                json.dump(get_state_method(obj, methods)(obj, dst, methods), f)

            # we use the zip format since tarfile can be exploited to create files
            # outside of the destination directory:
            # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
            shutil.make_archive(file, format="zip", root_dir=dst)
            shutil.move(f"{file}.zip", file)

    def load(self, file: str) -> Any:
        methods = self.extra_set_state_methods.copy()
        methods.update(SET_STATE_METHODS)
        input_zip = ZipFile(file)
        schema = json.loads(input_zip.read("schema.json"))
        return get_instance_method(schema, methods)(schema, input_zip, methods)
