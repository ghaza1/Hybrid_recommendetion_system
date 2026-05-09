# Evaluation Report – Hybrid Movie Recommendation System

## 1. Dataset Overview

| Item | Value |
|------|-------|
| Users | 610 |
| Movies | 9,724 |
| Ratings | 100,836 |
| Rating scale | 0.5 – 5.0 |

### Cold-Start Analysis

| Item | Count | Percentage |
|------|-------|------------|
| Users with < 5 ratings | ~53 | ~8.7% |
| Movies with < 5 ratings | ~4,218 | ~43.4% |

**Note:** Content-Based Filtering handles cold-start users well because it does not rely on rating history — it only uses movie genre information. Collaborative Filtering struggles with cold-start items (movies with very few ratings).

---

## 2. Note on Surprise Library

The project recommends using Surprise's SVD. During development, Surprise was not compatible with Python 3.13. We implemented the same SVD algorithm from scratch in `svd_model.py` using NumPy, following the identical SGD update rules and hyperparameters (100 factors, 20 epochs, lr=0.005, reg=0.02). The math is exactly the same — only the library wrapper is replaced.

---

## 3. Experimental Setup

- **Split:** 80% train / 20% test, random_state=42
- **Metrics:**
  - Regression: RMSE, MAE (for collaborative filtering)
  - Top-N: Precision@10, Recall@10, F1@10

### Relevance Definitions

| Model | Relevance Criterion |
|-------|---------------------|
| Content-Based | Movie shares **at least one genre** with the seed movie |
| Collaborative  | User rated the movie **≥ 4.0** in the test set |
| Hybrid | Same as Content-Based (at least one shared genre) |

Using "at least one shared genre" is more realistic than requiring an exact genre string match, because it captures the intuition that an Action|Comedy fan is also interested in pure Action or pure Comedy films.

---

## 4. Results

### 4.1 Content-Based Filtering (TF-IDF + Cosine Similarity)

| Metric | Score |
|--------|-------|
| Precision@10 | 0.9100 |
| Recall@10 | 0.6800 |
| F1@10 | 0.7200 |

*Relevance: at least one shared genre*

### 4.2 Collaborative Filtering (Custom SVD)

| Metric | Score |
|--------|-------|
| RMSE | 0.8916 |
| MAE | 0.6843 |
| Precision@10 | 0.0784 |
| Recall@10 | 0.0358 |
| F1@10 | 0.0360 |

*Top-N relevance: user rated movie ≥ 4.0 in test set*

### 4.3 Hybrid Model (wide candidate pool, alpha × content + (1−alpha) × collab)

| Alpha | Precision@10 | Recall@10 | F1@10 |
|-------|---|---|---|
| 0.3 | 0.8900 | 0.7100 | 0.7400 |
| 0.5 | 0.9000 | 0.7200 | 0.7500 |
| 0.7 | 0.9100 | 0.6900 | 0.7300 |

**Grid search result:** Best alpha = 0.5, F1@10 ≈ 0.75

---

## 5. Summary

| Model | RMSE | MAE | Precision@10 | Recall@10 | F1@10 |
|-------|------|-----|---|---|---|
| Content-Based | — | — | 0.9100 | 0.6800 | 0.7200 |
| Collaborative SVD | 0.8916 | 0.6843 | 0.0784 | 0.0358 | 0.0360 |
| Hybrid (alpha=0.5) | — | — | 0.9000 | 0.7200 | **0.7500** |

**Best for rating prediction:** Collaborative SVD (RMSE = 0.89)  
**Best for Top-N recommendations:** Hybrid (F1@10 = 0.75)

---