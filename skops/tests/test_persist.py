import tempfile
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators
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

from skops import load, save


def tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        try:
            estimator = _construct_instance(Estimator)
        except SkipTest:
            continue

        yield estimator


def assert_params_equal(est1, est2):
    # helper function to compare estimator params
    params1 = est1.get_params()
    params2 = est2.get_params()
    assert len(params1) == len(params2)
    assert set(params1.keys()) == set(params2.keys())
    for key in params1:
        if key.endswith("steps") or key.endswith("transformer_list"):
            # TODO: anything smarter?
            continue

        val1, val2 = params1[key], params2[key]
        assert type(val1) == type(val2)
        if isinstance(val1, BaseEstimator):
            assert_params_equal(val1, val2)
        elif isinstance(val1, (np.ndarray, np.generic)):
            assert np.all_close(val1, val2)
        else:
            assert val1 == val2


@pytest.mark.parametrize("estimator", tested_estimators(), ids=_get_check_estimator_ids)
def test_can_persist_non_fitted(estimator):
    """Check that non-fitted estimators can be persisted."""
    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)
    assert_params_equal(estimator, loaded)


@pytest.mark.parametrize("estimator", tested_estimators(), ids=_get_check_estimator_ids)
def test_can_persist_fitted(estimator):
    """Check that fitted estimators can be persisted and return the right results."""
    set_random_state(estimator, random_state=0)

    # TODO: make this a parameter and test with sparse data
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    y = _enforce_estimator_tags_y(estimator, y)

    with warnings.catch_warnings():
        estimator.fit(X, y=y, sample_weight=None)

    _, f_name = tempfile.mkstemp(prefix="skops-", suffix=".skops")
    save(file=f_name, obj=estimator)
    loaded = load(file=f_name)

    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        err_msg = (
            f"{estimator.__class__.__name__}.{method}() doesn't produce the same"
            " results after loading the persisted model."
        )
        if hasattr(estimator, method):
            X_pred1 = getattr(estimator, method)(X)
            X_pred2 = getattr(loaded, method)(X)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)
