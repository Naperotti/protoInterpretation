# Pairwise Distance Computation Locations

This document identifies where pairwise distances are computed in the protoInterpretation codebase.

## Summary

Pairwise distances are computed in **two main files**:
1. **`src/protoInterpretation/analysis.py`** - Primary location for chain analysis
2. **`src/protoInterpretation/projections.py`** - Used for dimensionality reduction diagnostics

---

## 1. Primary Location: `analysis.py`

### Function: `_pairwise_distances_for_step()`
**Location:** Lines 83-94

This is the **core helper function** that wraps sklearn's `pairwise_distances`:

```python
def _pairwise_distances_for_step(
    embeddings_step: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise distances between chains at a single step.

    embeddings_step: [N, D]
    Returns:
        dist: [N, N]
    """
    return pairwise_distances(embeddings_step, metric=metric)
```

**Purpose:** Compute distance matrix between N chains at a single timestep

**Default metric:** Cosine distance

---

### Usage 1: Horizon Width Computation
**Function:** `compute_horizon_width_curve()`  
**Location:** Lines 97-137

Computes the geometric "width" of the horizon over time by calculating:
- Max pairwise distance at each step
- Mean pairwise distance at each step (upper triangle only)
- 95th percentile of pairwise distances

```python
for t in range(T):
    emb_t = embeddings[:, t, :]  # [N, D]
    dist_mat = _pairwise_distances_for_step(emb_t, metric=metric)  # [N, N]
    
    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    vals = dist_mat[triu_indices]
    
    max_dist[t] = float(vals.max())
    mean_dist[t] = float(vals.mean())
    p95_dist[t] = float(np.percentile(vals, 95.0))
```

**Call site:** Line 118

---

### Usage 2: Line Fit Analysis
**Function:** `compute_line_fit_curve()`  
**Location:** Lines 200-228

Finds the most distant pair of chains and computes how well a 1D line explains the embedding cloud:

```python
for t in range(T):
    emb_t = embeddings[:, t, :]  # [N, D]
    dist_mat = _pairwise_distances_for_step(emb_t, metric=metric)
    i, j = _extreme_pair_indices(dist_mat)  # Find most distant pair
    # ... compute R² for line fit
```

**Call site:** Line 217

---

### Usage 3: KL Divergence Between Extremes
**Function:** `compute_kl_between_extremes()`  
**Location:** Lines 253-303

Finds extreme sequences (most distant pair) at the final step:

```python
if line_fit is None:
    emb_final = embeddings[:, -1, :]  # [N, D]
    dist_mat = _pairwise_distances_for_step(emb_final, metric="cosine")
    i_extreme, j_extreme = _extreme_pair_indices(dist_mat)
```

**Call site:** Line 279

---

## 2. Secondary Location: `projections.py`

### Function: `_compute_dr_diagnostics()`
**Location:** Lines 49-94

Computes dimensionality reduction quality metrics by comparing pairwise distances in high-dimensional and low-dimensional spaces:

```python
# Distance correlation (Pearson correlation between flattened distance matrices)
dist_high = pairwise_distances(base_embeddings, metric="euclidean")
dist_low = pairwise_distances(viz_embeddings, metric="euclidean")

# Use upper triangle to avoid double-counting / diagonals
triu = np.triu_indices(dist_high.shape[0], k=1)
dh = dist_high[triu].ravel()
dl = dist_low[triu].ravel()

# Compute Pearson correlation
# ... correlation computation
```

**Call sites:** Lines 73-74

**Purpose:** Evaluate how well the projection preserves pairwise distances

**Metric:** Euclidean distance (unlike analysis.py which uses cosine by default)

---

## Import Statement

Both files import from sklearn:

```python
from sklearn.metrics import pairwise_distances
```

- **analysis.py:** Line 6
- **projections.py:** Line 8

---

## Key Differences

| Aspect | `analysis.py` | `projections.py` |
|--------|---------------|------------------|
| **Primary metric** | Cosine distance | Euclidean distance |
| **Purpose** | Chain diversity analysis | DR quality diagnostics |
| **Wrapper function** | Yes (`_pairwise_distances_for_step`) | No (direct call) |
| **Frequency** | Called in loop over time | Called once per projection |

---

## Visualizations

Pairwise distance metrics are visualized in `viz.py`:

- **Line 59:** Plots mean pairwise distance over time
- **Line 60:** Plots max pairwise distance over time
- **Line 155:** Y-axis label for mean pairwise distance (cosine)

---

## High-Level API

The main entry point that triggers all pairwise distance computations:

**Function:** `compute_horizon_metrics()`  
**Location:** `analysis.py`, Lines 397-443

This function orchestrates:
1. Entropy computation (no pairwise distances)
2. **Horizon width** → calls `compute_horizon_width_curve()` → **uses pairwise distances**
3. **Line fit** → calls `compute_line_fit_curve()` → **uses pairwise distances**
4. **KL divergence** → calls `compute_kl_between_extremes()` → **uses pairwise distances**
5. Clustering (uses embeddings directly, not pairwise distances)

---

## Summary

**Main computation point:** `_pairwise_distances_for_step()` in `analysis.py` (line 83-94)

**Usage count:** 3 times in analysis.py, 2 times in projections.py

**Default behavior:** Cosine distance for chain analysis, Euclidean for projection diagnostics
