import numpy as np

def reference(q, k, v):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    qk = k @ q                         # one raw score per token
    scores = qk / np.sqrt(q.size)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return qk, weights, weights @ v

# Hand-checkable case: equal scores give equal weights.
q = np.array([1, 0], dtype=np.float16)
k = np.array([[2, 1], [2, -1]], dtype=np.float16)
v = np.array([[1, 2], [3, 6]], dtype=np.float16)
qk, weights, out = reference(q, k, v)
np.testing.assert_allclose(qk, [2, 2])
np.testing.assert_allclose(weights, [0.5, 0.5])
np.testing.assert_allclose(out, [2, 4])
print("Hand-check passed: output =", out)

# First kernel test: 16 tokens, one head, head dimension 64.
rng = np.random.default_rng(42)
L, d = 16, 64
q, k, v = (rng.standard_normal(shape).astype(np.float16)
           for shape in [(d,), (L, d), (L, d)])
qk, weights, out = reference(q, k, v)
np.testing.assert_allclose(weights.sum(), 1.0)
assert out.shape == (d,) and np.isfinite(out).all()
np.savez("case_L16_d64.npz", q=q, k=k, v=v,
         qk_ref=qk, weights_ref=weights, out_ref=out)
print("Saved case_L16_d64.npz")
print("QK / weights / output shapes:", qk.shape, weights.shape, out.shape)
