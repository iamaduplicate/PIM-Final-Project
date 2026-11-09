import numpy as np

with np.load("case_L16_d64.npz") as case:
    q = case["q"].astype(np.float64)
    k = case["k"].astype(np.float64)
    expected = case["qk_ref"]

L, d = k.shape
lanes = 16
assert q.shape == (d,) and d % lanes == 0

# Each row is one token; each column is one lane's partial sum.
partials = np.zeros((L, lanes), dtype=np.float64)
for lane in range(lanes):
    partials[:, lane] = k[:, lane::lanes] @ q[lane::lanes]

# Combine lanes within each score, never across different tokens.
reconstructed = partials.sum(axis=1)
np.testing.assert_allclose(reconstructed, expected, rtol=1e-12, atol=1e-12)

print("QK layout check passed")
print("Partial sums (tokens, lanes):", partials.shape)
print("Products per lane per score:", d // lanes)
print("Maximum absolute QK error:", np.max(np.abs(reconstructed - expected)))
