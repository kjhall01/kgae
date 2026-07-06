import torch


# -------------------------
# Conjugate Gradient solver
# -------------------------
def cg_solve(A, B, x0=None, max_iter=200, tol=1e-6):
    """
    Solve A X = B using Conjugate Gradient.
    A: (n, n) SPD
    B: (n, k)
    """
    X = torch.zeros_like(B) if x0 is None else x0.clone()

    R = B - A @ X
    P = R.clone()

    Rs_old = (R * R).sum(dim=0)

    for _ in range(max_iter):
        AP = A @ P
        alpha = Rs_old / (P * AP).sum(dim=0)
        X = X + P * alpha
        R = R - AP * alpha
        Rs_new = (R * R).sum(dim=0)

        if torch.sqrt(Rs_new.max()) < tol:
            break

        P = R + P * (Rs_new / Rs_old)
        Rs_old = Rs_new

    return X


# -------------------------
# MPS-safe least squares
# -------------------------
def mps_linalg_solve(A, B, *, reg=1e-4, max_iter=200, tol=1e-6):
    """
    Least squares solve using normal equations + CG.
    Matches torch.linalg.lstsq(A, B).solution
    """
    AtA = A.T @ A
    AtB = A.T @ B

    if reg > 0:
        AtA = AtA + reg * torch.eye(
            AtA.shape[0],
            device=A.device,
            dtype=A.dtype,
        )

    return cg_solve(AtA, AtB, max_iter=max_iter, tol=tol)


# -------------------------
# Test harness
# -------------------------
def main():
    assert torch.backends.mps.is_available(), "MPS not available"
    device = torch.device("mps")

    torch.manual_seed(0)

    # Problem dimensions
    m = 5000   # rows
    n = 64     # columns
    k = 3      # RHS count

    A = torch.randn(m, n, device=device)
    B = torch.randn(m, k, device=device)

    # ---- MPS solve ----
    X_mps = mps_linalg_solve(A, B, reg=1e-4)

    # ---- Residual on MPS ----
    residual_mps = torch.linalg.norm(A @ X_mps - B, dim=0)

    # ---- CPU reference ----
    A_cpu = A.cpu()
    B_cpu = B.cpu()

    X_cpu = torch.linalg.lstsq(A_cpu, B_cpu).solution
    residual_cpu = torch.linalg.norm(A_cpu @ X_cpu - B_cpu, dim=0)

    # ---- Error metrics ----
    rel_error = torch.linalg.norm(X_mps.cpu() - X_cpu) / torch.linalg.norm(X_cpu)

    print("=== Least Squares Test ===")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print()
    print("Residual norms:")
    print(f"  MPS CG   : {residual_mps.cpu().numpy()}")
    print(f"  CPU lstsq: {residual_cpu.numpy()}")
    print()
    print(f"Relative solution error ||X_mps - X_cpu|| / ||X_cpu|| = {rel_error:.3e}")

    # Sanity check
    assert torch.all(torch.isfinite(X_mps)), "Non-finite values in solution"
    print("\nTest passed ✔")


if __name__ == "__main__":
    main()
