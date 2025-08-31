import pickle
from concurrent.futures import ProcessPoolExecutor
from find_distance import compute_distance, toric4d_distance_fast
from rotated_toric_code import build_4d_toric_code


def hermite_4x4_for_D(D):
    """生成固定 D 的所有 L"""
    for d0 in range(1, D + 1):
        if D % d0 != 0:
            continue
        D1 = D // d0
        for d1 in range(1, D1 + 1):
            if D1 % d1 != 0:
                continue
            D2 = D1 // d1
            for d2 in range(1, D2 + 1):
                if D2 % d2 != 0:
                    continue
                d3 = D2 // d2
                for L01 in range(d1):
                    for L02 in range(d2):
                        for L03 in range(d3):
                            for L12 in range(d2):
                                for L13 in range(d3):
                                    for L23 in range(d3):
                                        L = [
                                            [d0, L01, L02, L03],
                                            [0, d1, L12, L13],
                                            [0, 0, d2, L23],
                                            [0, 0, 0, d3],
                                        ]
                                        yield L


def distance_exact(L):
    hx, hz = build_4d_toric_code(L, 2, compute_extra=False)
    d = compute_distance(hx, hz, verbosity=0)
    if d is None:
        print(f"d is None for L={L}")
    return d


def distance_approx(L, max_coord=5):
    return toric4d_distance_fast(L, max_coord=max_coord, use_transpose=True)


def process_batch(batch):
    return [distance_exact(L) for L in batch]


def find_opt_L_for_D_parallel(
    D, distance_func=distance_exact, batch_size=1000, max_workers=None
):
    """对于固定 D 并行计算 distance，并返回最大值和对应 L*"""
    L_list = list(hermite_4x4_for_D(D))
    max_d = None
    opt_Ls = []

    # 按 batch 分块
    for i in range(0, len(L_list), batch_size):
        batch = L_list[i: i + batch_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            distances = list(executor.map(distance_func, batch))

        # 更新最大值
        for L, d in zip(batch, distances):
            if (max_d is None) or (d > max_d):
                max_d = d
                opt_Ls = [L]
            elif d == max_d:
                opt_Ls.append(L)

    return D, opt_Ls, max_d


def save_checkpoint(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


if __name__ == "__main__":
    checkpoint_file = "./TMP/distance_checkpoint.pkl"
    results = load_checkpoint(checkpoint_file)
    start_D = max(results.keys(), default=2) + 1
    num_D = 30

    for D in range(start_D, start_D + num_D):
        # D, opt_Ls, max_d = find_opt_L_for_D_parallel(D, batch_size=20, max_workers=50)
        # results[D].update({"opt_Ls": opt_Ls, "max_d": max_d})
        D, approx_Ls, approx_d = find_opt_L_for_D_parallel(
            D, distance_func=distance_approx, batch_size=20, max_workers=50
        )
        results[D] = {"approx_Ls": approx_Ls, "approx_d": approx_d}
        save_checkpoint(checkpoint_file, results)

        # print(f"D = {D}, max distance = {max_d}, 最优矩阵数 = {len(opt_Ls)}")
        print(f"D = {D}, approx distance = {
              approx_d}, approx矩阵数 = {len(approx_Ls)}")
