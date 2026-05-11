"""Unit tests for the KV-cache rolling-slice guard in
``models.wan.causal_model``.

These tests pin down the bug that previously took down rank 0 (and with
NCCL, the whole 4-GPU pipeline) in long streaming sessions: the rolling
window could advance into negative ``local_start_index`` territory while
the ``end - start == num_new_tokens`` check still evaluated True, so the
code happily did a zero-width slice write of a non-zero-width RHS and
broadcast-crashed.

We only exercise the pure guard helpers, so the tests run on CPU without
needing flash-attn / flex-attention / distributed init. If the module
cannot be imported at all (e.g. torch missing) the whole module is
skipped. The file runs under either ``pytest`` or the stdlib runner::

    python -m unittest tests.test_causal_model_kvcache_guard
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

# Make ``models.wan.causal_model`` importable when launched from either
# the repo root or the ``tests/`` directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from models.wan import causal_model as _causal_model
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover - env-dependent
    _causal_model = None
    _IMPORT_ERR = e

_skip_if_no_module = unittest.skipIf(
    _causal_model is None,
    f"models.wan.causal_model unavailable: {_IMPORT_ERR!r}",
)

try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - env-dependent
    _torch = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# _kvcache_slice_ok: 8 edge cases mapped 1-1 to real failure modes seen in
# production streaming sessions.
# ---------------------------------------------------------------------------

@_skip_if_no_module
class TestKvcacheSliceOk(unittest.TestCase):
    """The 8 parametrised edge cases.

    Each ``test_case_N_xxx`` is one row of the original parametrised
    table; kept as individual test methods so a failure report names
    the exact case.
    """

    def setUp(self):
        self.fn = _causal_model._kvcache_slice_ok

    # ---- happy paths --------------------------------------------------

    def test_case_1_exact_fit_at_head(self):
        self.assertTrue(self.fn(0, 5, 5, 100))

    def test_case_2_exact_fit_middle(self):
        self.assertTrue(self.fn(40, 45, 5, 100))

    def test_case_3_exact_fit_tail(self):
        self.assertTrue(self.fn(95, 100, 5, 100))

    # ---- the bug rows -------------------------------------------------

    def test_case_4_negative_start_zero_end(self):
        # Python slice semantics would turn k[:, -5:0] into an empty
        # view on the LHS while the RHS has width 5 -> broadcast crash.
        # Guard must reject.
        self.assertFalse(self.fn(-5, 0, 5, 100))

    def test_case_5_negative_start_positive_end(self):
        # Also from rolling-window wrap; width check alone would pass
        # (3 - (-2) == 5) but guard must still reject.
        self.assertFalse(self.fn(-2, 3, 5, 100))

    def test_case_6_zero_width_window(self):
        # start == end with num_new > 0.
        self.assertFalse(self.fn(7, 7, 5, 100))

    def test_case_7_overrun_past_tail(self):
        # end > cache_size.
        self.assertFalse(self.fn(98, 103, 5, 100))

    def test_case_8_width_mismatch(self):
        # window inside cache but not num_new tokens wide.
        self.assertFalse(self.fn(10, 14, 5, 100))

    # ---- type-coercion sanity ----------------------------------------

    @unittest.skipUnless(_HAS_TORCH, "torch not available")
    def test_accepts_tensor_scalar_indices(self):
        """At the real call site ``local_end_index`` is often a 0-dim
        torch tensor (``kv_cache["local_end_index"][i]``). The guard
        must coerce tensor/int inputs to a canonical int; otherwise a
        mixed-type comparison could produce a surprising ``True`` with
        the wrong width.
        """
        start_t = _torch.tensor(10)
        end_t = _torch.tensor(15)
        self.assertTrue(self.fn(start_t, end_t, 5, 100))
        self.assertTrue(self.fn(start_t, 15, 5, 100))
        self.assertFalse(self.fn(start_t, end_t, 4, 100))

    def test_unconvertible_input_returns_false(self):
        # Should fail closed (False) rather than raise.
        self.assertFalse(self.fn("bad", 5, 5, 100))


# ---------------------------------------------------------------------------
# _kvcache_warn: rate-limit behaviour. Without rate-limiting a misbehaving
# streaming session floods the log at ~30 msgs/sec/rank.
# ---------------------------------------------------------------------------

@_skip_if_no_module
class TestKvcacheWarnRateLimit(unittest.TestCase):

    def setUp(self):
        self.state = _causal_model._KVCACHE_WARN_STATE
        self.state["count"] = 0
        self.state["stride"] = 1
        self.warn = _causal_model._kvcache_warn

    def tearDown(self):
        self.state["count"] = 0
        self.state["stride"] = 1

    def _fire(self, n):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(n):
                self.warn("test-violation")
        return [str(w.message) for w in caught if "kvcache-guard" in str(w.message)]

    def test_first_call_always_fires(self):
        msgs = self._fire(1)
        self.assertEqual(len(msgs), 1)
        self.assertIn("[n=1]", msgs[0])

    def test_rate_limit_progression(self):
        """Expected firing schedule over 200 consecutive violations:

          * calls  1..10  (stride=1)   -> every call fires  -> 10 msgs
          * calls 11..100 (stride=10)  -> fires at 20,30,...,100 -> 9 msgs
          * calls 101..200(stride=100) -> fires at 200 -> 1 msg

        Total: 20 msgs.
        """
        msgs = self._fire(200)
        ns = [int(m.split("[n=")[1].split("]")[0]) for m in msgs]
        expected = list(range(1, 11)) + list(range(20, 101, 10)) + [200]
        self.assertEqual(
            ns, expected,
            msg=f"rate-limit schedule changed: got {ns}, expected {expected}",
        )

    def test_does_not_raise_without_distributed(self):
        # dist is not initialised in the test process; must not raise.
        self.warn("no-dist")
        self.assertEqual(self.state["count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
