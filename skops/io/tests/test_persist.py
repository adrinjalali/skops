import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse, special
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.exceptions import SkipTestWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedGroupKFold, check_cv
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.utils import all_estimators, check_random_state
from sklearn.utils._testing import (
    SkipTest,
    assert_allclose_dense_sparse,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _construct_instance,
    _enforce_estimator_tags_y,
    _get_check_estimator_ids,
)

from skops.io import load, save
from skops.utils.fixes import path_unlink

# list of estimators for which we need to write tests since we can't
# automatically create an instance of them.
EXPLICIT_TESTS = [
    "ColumnTransformer",
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "RandomizedSearchCV",
    "SparseCoder",
]

# These estimators fail in our tests, we should fix them one by one, by
# removing them from this list, and fixing the error.
#
# This list can be generated generated by pasting the last part of the pytest
# log into a 1.log file, and then running:
#
# cat /tmp/1.log | sed "s/.*fitted\[//g" | sed "s/(.*//g"
ESTIMATORS_TO_IGNORE = []  # type: ignore


def save_load_round(estimator):
    # save and then load the model, and return the loaded model.
    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)
    path_unlink(Path(f_name))
    return loaded


def _tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        try:
            # suppress warnings here for skipped estimators.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=SkipTestWarning,
                    message="Can't instantiate estimator",
                )
                estimator = _construct_instance(Estimator)
        except SkipTest:
            continue

        yield estimator

    # nested Pipeline & FeatureUnion
    # fmt: off
    yield Pipeline([
        ("features", FeatureUnion([
            ("scaler", StandardScaler()),
            ("scaled-poly", Pipeline([
                ("polys", FeatureUnion([
                    ("poly1", PolynomialFeatures()),
                    ("poly2", PolynomialFeatures(degree=3, include_bias=False))
                ])),
                ("scale", MinMaxScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(random_state=0, solver="liblinear")),
    ])
    # fmt: on

    # FunctionTransformer with numpy functions
    yield FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.square,
    )

    # FunctionTransformer with scipy functions - problem is that they look like
    # numpy ufuncs
    yield FunctionTransformer(
        func=special.erf,
        inverse_func=special.erfinv,
    )


def _is_steps_like(obj):
    # helper function to check if an object is something like Pipeline.steps,
    # i.e. a list of tuples of names and estimators
    if not isinstance(obj, list):  # must be a list
        return False

    if not obj:  # must not be empty
        return False

    if not isinstance(obj[0], tuple):  # must be list of tuples
        return False

    lens = set(map(len, obj))
    if not lens == {2}:  # all elements must be length 2 tuples
        return False

    keys, vals = list(zip(*obj))

    if len(keys) != len(set(keys)):  # keys must be unique
        return False

    if not all(map(lambda x: isinstance(x, (type(None), BaseEstimator)), vals)):
        # values must be BaseEstimators or None
        return False

    return True


def _assert_vals_equal(val1, val2):
    if hasattr(val1, "__getstate__"):
        # This includes BaseEstimator since they implement __getstate__ and
        # that returns the parameters as well.
        assert_params_equal(val1.__getstate__(), val2.__getstate__())
    elif sparse.issparse(val1):
        assert sparse.issparse(val2) and ((val1 - val2).nnz == 0)
    elif isinstance(val1, (np.ndarray, np.generic)):
        if len(val1.dtype) == 0:
            # simple comparison of arrays with simple dtypes, almost all arrays
            # are of this sort.
            assert np.allclose(val1, val2)
        elif len(val1.shape) == 1:
            # comparing arrays with structured dtypes, but they have to be 1D
            # arrays. This is what we get from the Tree's state.
            assert np.all([x == y for x, y in zip(val1, val2)])
        else:
            # we don't know what to do with these values, for now.
            assert False
    elif isinstance(val1, (tuple, list)):
        assert len(val1) == len(val2)
        for subval1, subval2 in zip(val1, val2):
            _assert_vals_equal(subval1, subval2)
    elif isinstance(val1, float) and np.isnan(val1):
        assert np.isnan(val2)
    elif isinstance(val1, dict):
        # dictionaries are compared by comparing their values recursively.
        assert set(val1.keys()) == set(val2.keys())
        for key in val1:
            _assert_vals_equal(val1[key], val2[key])
    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        _assert_vals_equal(val1.__dict__, val2.__dict__)
    else:
        assert val1 == val2


def assert_params_equal(params1, params2):
    # helper function to compare estimator dictionaries of parameters
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)

        if _is_steps_like(val1):
            # Deal with Pipeline.steps, FeatureUnion.transformer_list, etc.
            assert _is_steps_like(val2)
            val1, val2 = dict(val1), dict(val2)

        if isinstance(val1, (tuple, list)):
            assert len(val1) == len(val2)
            for subval1, subval2 in zip(val1, val2):
                _assert_vals_equal(subval1, subval2)
        elif isinstance(val1, dict):
            assert_params_equal(val1, val2)
        else:
            _assert_vals_equal(val1, val2)


def _get_learned_attrs(estimator):
    # Find the learned attributes like "coefs_"
    attrs = {}
    for key in estimator.__dict__:
        if key.startswith("_") or not key.endswith("_"):
            continue

        val = getattr(estimator, key)
        if isinstance(val, property):
            continue
        attrs[key] = val
    return attrs


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_non_fitted(estimator):
    """Check that non-fitted estimators can be persisted."""
    if estimator.__class__.__name__ in ESTIMATORS_TO_IGNORE:
        pytest.skip()

    loaded = save_load_round(estimator)
    assert_params_equal(estimator.get_params(), loaded.get_params())


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_can_persist_fitted(estimator):
    """Check that fitted estimators can be persisted and return the right results."""
    if estimator.__class__.__name__ in ESTIMATORS_TO_IGNORE:
        pytest.skip()

    set_random_state(estimator, random_state=0)

    # TODO: make this a parameter and test with sparse data
    # TODO: try with pandas.DataFrame as well
    # This data can be used for a regression model as well.
    X, y = make_classification(n_samples=50)
    # Some models require positive X
    X = np.abs(X)
    y = _enforce_estimator_tags_y(estimator, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")
        estimator.fit(X, y)

    loaded = save_load_round(estimator)
    # check that params and learned attributes are equal
    assert_params_equal(estimator.get_params(), loaded.get_params())
    attrs_est = _get_learned_attrs(estimator)
    attrs_loaded = _get_learned_attrs(loaded)
    assert_params_equal(attrs_est, attrs_loaded)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
        "predict_log_proba",
    ]:
        err_msg = (
            f"{estimator.__class__.__name__}.{method}() doesn't produce the same"
            " results after loading the persisted model."
        )
        if hasattr(estimator, method):
            X_pred1 = getattr(estimator, method)(X)
            X_pred2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)


class RandomStateEstimator(BaseEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        if isinstance(self.random_state, np.random.Generator):
            # forwards compatibility with np.random.Generator
            self.random_state_ = self.random_state
        else:
            self.random_state_ = check_random_state(self.random_state)
        return self


@pytest.mark.parametrize(
    "random_state",
    [
        None,
        0,
        np.random.RandomState(42),
        np.random.default_rng(),
        np.random.Generator(np.random.PCG64DXSM(seed=123)),
    ],
)
def test_random_state(random_state):
    # Numpy random Generators
    # (https://numpy.org/doc/stable/reference/random/generator.html) are not
    # supported by sklearn yet but will be in the future, thus they're tested
    # here
    est = RandomStateEstimator(random_state=random_state).fit(None, None)
    est.random_state_.random(123)  # move RNG forwards

    loaded = save_load_round(est)
    rand_floats_expected = est.random_state_.random(100)
    rand_floats_loaded = loaded.random_state_.random(100)
    np.testing.assert_equal(rand_floats_loaded, rand_floats_expected)


class CVEstimator(BaseEstimator):
    def __init__(self, cv=None):
        self.cv = cv

    def fit(self, X, y, **fit_params):
        self.cv_ = check_cv(self.cv)
        return self

    def split(self, X, **kwargs):
        return list(self.cv_.split(X, **kwargs))


@pytest.mark.parametrize(
    "cv",
    [
        None,
        3,
        KFold(4),
        StratifiedGroupKFold(5, shuffle=True, random_state=42),
        ShuffleSplit(6, random_state=np.random.RandomState(123)),
    ],
)
def test_cross_validator(cv):
    est = CVEstimator(cv=cv).fit(None, None)
    loaded = save_load_round(est)
    X, y = make_classification(n_samples=50)

    kwargs = {}
    name = est.cv_.__class__.__name__.lower()
    if "stratified" in name:
        kwargs["y"] = y
    if "group" in name:
        kwargs["groups"] = np.random.randint(0, 5, size=len(y))

    splits_est = est.split(X, **kwargs)
    splits_loaded = loaded.split(X, **kwargs)
    assert len(splits_est) == len(splits_loaded)
    for split_est, split_loaded in zip(splits_est, splits_loaded):
        np.testing.assert_equal(split_est, split_loaded)


# TODO: remove this, Adrin uses this for debugging.
if __name__ == "__main__":
    from sklearn.cross_decomposition import PLSRegression

    SINGLE_CLASS = PLSRegression

    estimator = _construct_instance(SINGLE_CLASS)
    loaded = save_load_round(estimator)
    assert_params_equal(estimator.get_params(), loaded.get_params())

    set_random_state(estimator, random_state=0)

    # TODO: make this a parameter and test with sparse data
    # TODO: try with pandas.DataFrame as well
    # This data can be used for a regression model as well.
    X, y = make_classification(n_samples=50)
    # Some models require positive X
    X = np.abs(X)
    y = _enforce_estimator_tags_y(estimator, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="sklearn")
        estimator.fit(X, y)

    loaded = save_load_round(estimator)
    # check that params and learned attributes are equal
    assert_params_equal(estimator.get_params(), loaded.get_params())
    attrs_est = _get_learned_attrs(estimator)
    attrs_loaded = _get_learned_attrs(loaded)
    assert_params_equal(attrs_est, attrs_loaded)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
        "predict_log_proba",
    ]:
        err_msg = (
            f"{estimator.__class__.__name__}.{method}() doesn't produce the same"
            " results after loading the persisted model."
        )
        if hasattr(estimator, method):
            X_pred1 = getattr(estimator, method)(X)
            X_pred2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)
