import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import time
from numba import jit, prange

# Numba's Just-In-Time (JIT) compiler is used here to accelerate the function.
@jit(nopython=True, parallel=True)
def matvec_M_final_accurate(k, v):
    # Computes the matrix-vector product and uses float64
    y_a1, y_b1, y_c1 = v.astype(np.float64), v.astype(np.float64), v.astype(np.float64)
    y_a2, y_b2, y_c2 = np.empty_like(v, dtype=np.float64), np.empty_like(v, dtype=np.float64), np.empty_like(v, dtype=np.float64)
    size = v.shape[0]

    for i in range(1, k + 1):
        if i % 2 == 1:
            src_a, src_b, src_c = y_a1, y_b1, y_c1
            dst_a, dst_b, dst_c = y_a2, y_b2, y_c2
        else:
            src_a, src_b, src_c = y_a2, y_b2, y_c2
            dst_a, dst_b, dst_c = y_a1, y_b1, y_c1

        block_size = 2**i
        half_block = 2**(i - 1)

        for j in prange(size // block_size):
            start = j * block_size
            mid = start + half_block
            end = start + block_size

            a1, a2 = src_a[start:mid], src_a[mid:end]
            b1, b2 = src_b[start:mid], src_b[mid:end]
            c1, c2 = src_c[start:mid], src_c[mid:end]

            dst_a[start:mid] = c2
            dst_a[mid:end] = c1

            dst_b[start:mid] = a1
            dst_b[mid:end] = b2

            dst_c[start:mid] = b1
            dst_c[mid:end] = a2

    if k % 2 == 1:
        return y_a2 + y_b2 + y_c2
    else:
        return y_a1 + y_b1 + y_c1

if __name__ == "__main__":
    print("--- Calculating Spectral Gap")
    limit_value = 2 * np.sqrt(2)
    print(f"Target comparison value (2*sqrt(2)): {limit_value:.30f}\n")

    for k in range(2, 19):
        size = 2**k
        print(f"Processing Level k={k}, Matrix Size={size}x{size}...")

        if k == 2:
            dummy_v = np.ones(size, dtype=np.float64)
            _ = matvec_M_final_accurate(k, dummy_v)

        start_time = time.time()
        M_k_operator = LinearOperator(
            (size, size),
            matvec=lambda v, k=k: matvec_M_final_accurate(k, v),
            dtype=np.float64
        )

        try:
            eigenvalues = eigsh(
                M_k_operator, k=2, which='LA', tol=1e-12,
                maxiter=size, return_eigenvectors=False
            )
            elapsed = time.time() - start_time
            eigenvalues = np.sort(eigenvalues)[::-1]
            lambda2 = eigenvalues[1]

            print(f"  \033[92m> Success!\033[0m (Took {elapsed:.2f} seconds)")
            print(f"  > Largest Eigenvalue (λ1):          {eigenvalues[0]:.30f}")
            print(f"  > Second-Largest Eigenvalue (λ2): {lambda2:.30f}")

            if lambda2 < limit_value:
                print(f"  \033[96m> Verification: λ2 is LESS than 2*sqrt(2).\033[0m\n")
            else:
                print(f"  \033[91m> Verification: λ2 is GREATER than or EQUAL to 2*sqrt(2).\033[0m\n")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  \033[91m> Failed after {elapsed:.2f} seconds. Error: {e}\033[0m\n")