import numpy as np
import pandas as pd
import time
from numba import njit, prange
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
#                        Dataset Loader
# ============================================================
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("No numeric columns found in dataset.")
    data = df.values.astype(np.float64)
    data = StandardScaler().fit_transform(data)
    print(f"Loaded dataset: {data.shape[0]} samples × {data.shape[1]} features")
    return data


# ============================================================
#              Sequential K-Means
# ============================================================
def assign_clusters_naive(X, C):
    N, D = X.shape
    K = C.shape[0]
    labels = np.empty(N, dtype=np.int32)
    for i in range(N):
        best_k = -1
        best_dist = 1e300
        for j in range(K):
            s = 0.0
            for d in range(D):
                diff = X[i, d] - C[j, d]
                s += diff * diff
            if s < best_dist:
                best_dist = s
                best_k = j
        labels[i] = best_k
    return labels

def compute_centroids_naive(X, labels, K):
    N, D = X.shape
    centroids = np.zeros((K, D))
    counts = np.zeros(K)
    for i in range(N):
        k = labels[i]
        counts[k] += 1
        for d in range(D):
            centroids[k, d] += X[i, d]
    for k in range(K):
        if counts[k] > 0:
            centroids[k] /= counts[k]
    return centroids

def kmeans_naive(X, K=5, steps=10):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(steps):
        labels = assign_clusters_naive(X, centroids)
        centroids = compute_centroids_naive(X, labels, K)
    return labels, centroids


# ============================================================
#              ParallelK-Means with Tiling + SIMD
# ============================================================
@njit(parallel=True, fastmath=True)
def assign_clusters_tiled_simd(X, C, tile_cols=8):
    X = np.ascontiguousarray(X)
    C = np.ascontiguousarray(C)
    N, D = X.shape
    K = C.shape[0]
    labels = np.empty(N, dtype=np.int32)
    for i in prange(N):
        best_k = -1
        best_dist = 1e300
        for j0 in range(0, K, tile_cols):
            j1 = min(j0 + tile_cols, K)
            for j in range(j0, j1):
                dist = np.dot(X[i], X[i]) + np.dot(C[j], C[j]) - 2.0 * np.dot(X[i], C[j])
                if dist < best_dist:
                    best_dist = dist
                    best_k = j
        labels[i] = best_k
    return labels

@njit(parallel=True, fastmath=True)
def compute_centroids_parallel(X, labels, K):
    N, D = X.shape
    sums = np.zeros((K, D))
    counts = np.zeros(K)
    for i in prange(N):
        k = labels[i]
        counts[k] += 1
        for d in range(D):
            sums[k, d] += X[i, d]
    for k in prange(K):
        if counts[k] > 0:
            inv = 1.0 / counts[k]
            for d in range(D):
                sums[k, d] *= inv
    return sums

def kmeans_optimized(X, K=5, steps=10, tile_cols=8):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(steps):
        labels = assign_clusters_tiled_simd(X, centroids, tile_cols)
        centroids = compute_centroids_parallel(X, labels, K)
    return labels, centroids


# ============================================================
#                    Performance Evaluation
# ============================================================
def evaluate_blocking_effect(filepath, K=5, steps=10, tile_sizes=[2, 4, 8, 16, 32, 64]):
    print("\n=== Parallel K-Means: Effect of Blocking Size ===")
    X = load_dataset(filepath)
    N, D = X.shape

    # Warm up JIT
    print("\nWarming up Numba JIT...")
    _ = kmeans_optimized(X[:100], K=K, steps=1, tile_cols=tile_sizes[0])

    results = []
    for tile in tile_sizes:
        print(f"\nRunning with tile_cols = {tile}")
        start = time.time()
        labels, centroids = kmeans_optimized(X, K=K, steps=steps, tile_cols=tile)
        end = time.time()
        latency = end - start
        throughput = N / latency
        results.append((tile, latency, throughput))
        print(f"    Latency:   {latency:.6f} s")
        print(f"    Throughput: {throughput:.2f} points/s")

    df = pd.DataFrame(results, columns=["Tile Size", "Latency (s)", "Throughput (points/s)"])
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    return df


# ============================================================
#                   Visualization
# ============================================================
def plot_combined(df):
    fig, ax1 = plt.subplots(figsize=(8,5))

    # Latency bars
    ax1.bar(df["Tile Size"].astype(str), df["Latency (s)"], color='lightblue',
            edgecolor='black', width=0.6, label='Latency (s)')
    ax1.set_xlabel("Blocking Size (tile_cols)")
    ax1.set_ylabel("Latency (s)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Throughput line
    ax2 = ax1.twinx()
    ax2.plot(df["Tile Size"].astype(str), df["Throughput (points/s)"],
             color='tab:red', marker='o', linewidth=2, label='Throughput (points/s)')
    ax2.set_ylabel("Throughput (points/s)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Effect of Blocking Size on Latency and Throughput")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    plt.show()


# ============================================================
#                            Runner
# ============================================================
def run_kmeans(filepath, K=5, steps=10, use_parallel=True):
    if use_parallel:
        df_results = evaluate_blocking_effect(filepath, K=K, steps=steps)
        plot_combined(df_results)
    else:
        print("\n=== Sequential (Naive) K-Means ===")
        X = load_dataset(filepath)
        start = time.time()
        labels, centroids = kmeans_naive(X, K=K, steps=steps)
        end = time.time()
        latency = end - start
        throughput = X.shape[0] / latency
        print(f"Latency:   {latency:.6f} s")
        print(f"Throughput: {throughput:.2f} points/s")


# ============================================================
#                             Main Execution
# ============================================================
if __name__ == "__main__":
    filepath = "data.csv"  # dataset
    use_parallel = True   # True for parallel or False for serial
    run_kmeans(filepath, K=5, steps=10, use_parallel=use_parallel)
