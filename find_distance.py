import multiprocessing as mp
import numpy as np
from mip import Model, xsum, BINARY, INTEGER, OptimizationStatus
from bposd.css import css_code
from math import gcd, inf
import itertools


def l1_area(u, v):
    area = 0
    d = len(u)
    for i in range(d):
        for j in range(i + 1, d):
            area += abs(u[i] * v[j] - u[j] * v[i])
    return area


def primitive(vec):
    g = 0
    for x in vec:
        g = gcd(g, abs(int(x)))
    return g == 1


def toric4d_distance_fast(L, max_coord=3, use_transpose=False):
    L = np.array(L, dtype=int)
    if use_transpose:
        L = L.T

    candidates = []
    for coords in itertools.product(range(-max_coord, max_coord + 1), repeat=4):
        if all(c == 0 for c in coords):
            continue
        if not primitive(coords):
            continue
        vec = L @ np.array(coords, dtype=int)
        candidates.append(vec)
    candidates = np.array(candidates, dtype=int)

    best = inf
    n_cand = len(candidates)

    for i in range(n_cand):
        u = candidates[i]
        if np.sum(np.abs(u)) > best:
            continue
        for j in range(i + 1, n_cand):
            v = candidates[j]
            if np.linalg.matrix_rank(np.stack([u, v])) < 2:
                continue
            area = l1_area(u, v)
            if 0 < area < best:
                best = area
    return int(best)


def gf2_row_echelon(A):
    A = A.copy().astype(np.uint8) % 2
    m, n = A.shape
    row = 0
    pivots = []
    for col in range(n):
        # find pivot
        sel = None
        for r in range(row, m):
            if A[r, col]:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        # eliminate others
        for r in range(m):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break
    return A, pivots


def gf2_nullspace_basis(H):
    H = H.copy().astype(np.uint8) % 2
    m, n = H.shape
    R, pivots = gf2_row_echelon(H)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis = []
    for free in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        # solve for pivot vars: R[row]*vec = 0
        # find pivot rows in order
        for row_idx, pc in enumerate(pivots):
            # dot product row with vec (only free columns contribute)
            s = (R[row_idx] & vec).sum() % 2
            # since pivot at pc is 1, set vec[pc] = s to make the row sum 0
            vec[pc] = s
        basis.append(vec)
    return basis  # list of numpy uint8 vectors


def find_v_with_odd_overlap(lx, null_basis):
    if len(null_basis) == 0:
        return None  # only zero vector in kernel
    # build matrix M whose columns are basis vectors (shape n x t)
    B = np.column_stack(null_basis).astype(np.uint8)  # n x t
    # compute s = lx . B (1 x t)  (over GF(2))
    s = (lx.astype(np.uint8) @ B) % 2  # length t
    # we need c (length t) such that s·c = 1 (mod 2)
    # This is a single linear equation: if s is all zeros => no solution
    if np.all(s == 0):
        return None
    # otherwise choose any c with s·c = 1; simplest: pick index j with s[j]==1 and set c[j]=1
    j = np.argmax(s)  # index with s[j]==1
    c = np.zeros(B.shape[1], dtype=np.uint8)
    c[j] = 1
    # build v = B @ c
    v = (B @ c) % 2
    return v


def min_weight_solution_mod2(
    H,
    ell,
    time_limit=None,
    verbosity=0,
    threads=80,
    set_start=False,
    start_x=None,
    start_s=None,
    start_t=None,
):
    H_arr = np.asarray(H).astype(np.int64) % 2
    ell_arr = np.asarray(ell).astype(np.int64).ravel() % 2

    m, n = H_arr.shape
    if ell_arr.shape[0] != n:
        raise ValueError("ell length must equal number of columns of H")

    model = Model()
    model.verbose = verbosity
    model.threads = threads
    model.emphasis = 1
    if time_limit is not None:
        model.max_seconds = float(time_limit)

    x = [model.add_var(var_type=BINARY) for _ in range(n)]
    s = []

    for i in range(m):
        supp_H = np.nonzero(H_arr[i, :])[0]
        w_H = int(supp_H.size)
        if w_H == 0:
            continue
        ub_H = w_H // 2
        s.append(model.add_var(var_type=INTEGER, lb=0, ub=ub_H))
        model += xsum(x[j] for j in supp_H) - 2 * s[i] == 0
    # ell constraint: sum_{j in supp_ell} x_j - 2*t == 1
    supp_ell = np.nonzero(ell_arr)[0]
    if supp_ell.size == 0:
        # ell is zero vector -> impossible to require ell·x == 1
        print("ell is zero vector -> no solution.")
        return None, None
    w_ell = int(supp_ell.size)
    ub_ell = w_ell // 2
    t = model.add_var(var_type=INTEGER, lb=0, ub=ub_ell)
    model += xsum(x[j] for j in supp_ell) - 2 * t == 1
    if set_start:
        start = (
            [(x[i], start_x[i]) for i in range(n)]
            + [(s[i], start_s[i]) for i in range(m)]
            + [(t, start_t)]
        )
        model.start = start

    model.objective = xsum(x[j] for j in range(n))

    status = model.optimize()

    if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        if set_start:
            print("No feasible solution found with set start. Solver status:", status)

        null_basis = gf2_nullspace_basis(H_arr)
        v = find_v_with_odd_overlap(ell_arr, null_basis)
        s = H_arr @ v // 2
        t = ell_arr @ v // 2
        w, x = min_weight_solution_mod2(
            H_arr,
            ell_arr,
            threads=threads,
            verbosity=0,
            set_start=True,
            start_x=v,
            start_s=s,
            start_t=t,
        )
        return w, x

    x_val = np.zeros(n, dtype=int)
    for j in range(n):
        v = x[j].x
        if v is None:
            v = 0
        x_val[j] = int(round(v))

    if m > 0:
        lhs = H_arr.dot(x_val) % 2
        if np.any(lhs != 0):
            print("Warning: returned solution does not satisfy Hx=0 (mod2). lhs:", lhs)
            return None, None
    if int(ell_arr.dot(x_val) % 2) != 1:
        print("Warning: returned solution does not satisfy ell·x=1 (mod2).")
        return None, None

    opt_weight = int(x_val.sum())
    return opt_weight, x_val


def solve_ip_once(args):
    H, ell, tl, verb, threads = args
    return min_weight_solution_mod2(
        H, ell, time_limit=tl, verbosity=verb, threads=threads
    )


def solve_all(H_list, ell_list, time_limit, verbosity, threads, processes=1):
    tasks = [
        (H_list[i], ell_list[i], time_limit, verbosity, threads)
        for i in range(len(H_list))
    ]
    with mp.Pool(processes=processes) as pool:
        results = pool.map(solve_ip_once, tasks)
    return results


def compute_distance(H_X, H_Z, time_limit=None, verbosity=0, threads=80):
    qcode = css_code(H_X, H_Z)

    lz = qcode.lz.toarray()
    lx = qcode.lx.toarray()
    k = lz.shape[0]
    if k == 0:
        return 0
    n = lz.shape[1]
    H_list = [H_X] * 2 * k
    ell_list = [lx[i, :] for i in range(k)] + [lz[i, :] for i in range(k)]
    results = solve_all(H_list, ell_list, time_limit,
                        verbosity, threads, processes=1)

    d = n
    for i in range(2 * k):
        w, _ = results[i]
        d = min(d, w)

    return d
