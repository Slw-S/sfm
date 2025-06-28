# tests/test_skfmm_contiguity.py
import numpy as np
import pytest
import skfmm
import functools
from inspect import signature

#
# --- monkey-patch skfmm.distance to enforce C-contiguity ---
#
def enforce_c_contiguous(func):
    """
    Decorator that checks all ndarray arguments and
    raises ValueError if any are not C-contiguous.
    """
    sig = signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        for name, val in bound.arguments.items():
            if isinstance(val, np.ndarray) and not val.flags['C_CONTIGUOUS']:
                raise ValueError(
                    f"Argument '{name}' to {func.__name__!r} must be C-contiguous, "
                    f"but got flags {val.flags!r}"
                )
        return func(*args, **kwargs)

    return wrapper

# apply the patch
skfmm.distance = enforce_c_contiguous(skfmm.distance)


#
# --- tests ---
#

def test_c_contiguous_2d_works():
    # random C-contiguous array
    phi = np.random.randn(50, 40)
    assert phi.flags.c_contiguous
    dist = skfmm.distance(phi)
    # shape preserved
    assert dist.shape == phi.shape
    # distances are non-negative
    assert (dist >= 0).all()


@pytest.mark.parametrize("order", ["C", "F"])
def test_fortran_order_raises_if_not_c(order):
    # build a signed-distance-like field
    phi = np.random.randn(30, 20)
    arr = np.array(phi, order=order)
    expect_c = arr.flags.c_contiguous

    if not expect_c:
        with pytest.raises(ValueError) as exc:
            skfmm.distance(arr)
        assert "must be C-contiguous" in str(exc.value)
    else:
        # sanity check: C-order should not raise
        _ = skfmm.distance(arr)


def test_non_contiguous_slices_raise():
    # a bigger array, then take a non-contiguous slice
    big = np.random.randn(100,100)
    sl = big[::2, ::3]  # takes every 2nd row, every 3rd col -> non-contiguous
    assert not sl.flags.c_contiguous
    with pytest.raises(ValueError):
        skfmm.distance(sl)


def test_mask_argument_enforcement():
    # skfmm.distance(phi, dx=1.0, mask=mask_array)
    phi = np.random.randn(60, 60)
    mask_f = np.asfortranarray(np.zeros_like(phi, dtype=bool))
    assert not mask_f.flags.c_contiguous

    # C-contiguous phi is fine, but mask is not
    with pytest.raises(ValueError):
        skfmm.distance(phi, dx=1.0, mask=mask_f)

    # if we fix mask to C, everything works
    mask_c = np.ascontiguousarray(mask_f)
    dist = skfmm.distance(phi, dx=1.0, mask=mask_c)
    assert dist.shape == phi.shape


def test_1d_analytic_distance():
    # simple 1D: phi(x) = x - 0.5 on [0,1] => true distance = |phi|
    N = 200
    x = np.linspace(0, 1, N)
    phi = x - 0.5
    # ensure C order
    phi = np.ascontiguousarray(phi)
    d = skfmm.distance(phi, dx=x[1]-x[0])
    # compare to analytic |phi|
    assert np.allclose(d, np.abs(phi), atol=1e-3)


# helper: run a small battery of random-shaped tests
@pytest.mark.parametrize("shape", [(5,), (10,10), (6,7,8)])
def test_random_shapes_consistency(shape):
    phi = np.random.randn(*shape)
    assert phi.flags.c_contiguous
    out1 = skfmm.distance(phi)
    # repeat call to ensure idempotence
    out2 = skfmm.distance(phi.copy())
    assert np.allclose(out1, out2, atol=1e-12)
