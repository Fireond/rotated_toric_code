import itertools
import numpy as np
from bposd.css import css_code
from typing import Dict, List, Sequence, Tuple


def rank2(A):
    rows, n = A.shape
    X = np.identity(n, dtype=int)
    for i in range(rows):
        y = np.dot(A[i, :], X) % 2
        not_y = (y + 1) % 2
        good = X[:, np.nonzero(not_y)]
        good = good[:, 0, :]
        bad = X[:, np.nonzero(y)]
        bad = bad[:, 0, :]
        if bad.shape[1] > 0:
            bad = np.add(bad, np.roll(bad, 1, axis=1))
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    return n - X.shape[1]


def add_unit_and_reduce(L, x, axis, delta=1):
    raw = [int(v) for v in x]
    raw[axis - 1] += delta
    return reduce_mod_L(raw, L)


def reduce_mod_L(x, L):
    n = L.shape[0]
    x = np.asarray(x, dtype=int).copy()
    z = np.zeros(n, dtype=int)
    r = np.zeros(n, dtype=int)
    for i in range(n):
        # s = x_i - sum_{j<i} L[j,i]*z_j
        s = x[i] - int((L[:i, i] * z[:i]).sum())
        q, rem = divmod(int(s), int(L[i, i]))
        z[i] = q
        r[i] = rem
    return tuple(int(v) for v in r)


def build_4d_toric_code(L: List, dim_q: int, compute_extra=True):
    L = np.array(L)
    dim = L.shape[0]

    # sets_dim[i] = { dim choose i }
    sets_dim = []
    for i in range(dim):
        set_dim = [frozenset(a)
                   for a in itertools.combinations(range(1, dim + 1), i)]
        sets_dim.append(set_dim)

    # cells_dim[i] = { ((x_0,...,x_{dim-1}), (direction)) }
    cells_dim = []
    cells_index = []
    for i in range(dim):
        coordinates = list(itertools.product(*(range(a) for a in np.diag(L))))
        cells = [(x, dir) for x in coordinates for dir in sets_dim[i]]
        cells_dim.append(cells)
        index_map = {cell: idx for idx, cell in enumerate(cells)}
        cells_index.append(index_map)

    num_q = len(cells_dim[dim_q])
    num_x = len(cells_dim[dim_q - 1])
    num_z = len(cells_dim[dim_q + 1])
    hx = np.zeros((num_x, num_q))
    hz = np.zeros((num_z, num_q))

    # compute hx
    for idx, (x, dir) in enumerate(cells_dim[dim_q - 1]):
        check = ("Xcheck", idx)
        for d in set(range(1, dim + 1)) - dir:
            q_cells = [
                (x, dir | {d}),
                (add_unit_and_reduce(L, x, d, delta=-1), dir | {d}),
            ]
            q_idxs = [cells_index[dim_q][q] for q in q_cells]
            hx[idx, q_idxs] = 1

    # compute hz
    for idx, (x, dir) in enumerate(cells_dim[dim_q + 1]):
        check = ("Zcheck", idx)
        for d in dir:
            q_cells = [
                (x, dir - {d}),
                (add_unit_and_reduce(L, x, d, delta=+1), dir - {d}),
            ]
            q_idxs = [cells_index[dim_q][q] for q in q_cells]
            hz[idx, q_idxs] = 1

    if not compute_extra:
        return hx, hz

    # compute_extra
    data_qubits = [("data", i) for i in range(num_q)]
    Xchecks = [("Xcheck", i) for i in range(num_x)]
    Zchecks = [("Zcheck", i) for i in range(num_z)]
    con_list: Dict[str, Dict[int, List[Tuple[str, int]]]] = {
        k: {} for k in ["X", "Z"]}

    lin_order: Dict[Tuple[str, int], int] = {}
    idx = 0
    for q in Xchecks:
        lin_order[q] = idx
        idx += 1
    for q in data_qubits:
        lin_order[q] = idx
        idx += 1
    for q in Zchecks:
        lin_order[q] = idx
        idx += 1

    for idx, (x, dir) in enumerate(cells_dim[dim_q - 1]):
        check = ("Xcheck", idx)
        for d in set(range(1, dim + 1)) - dir:
            q_cells = [
                (x, dir | {d}),
                (add_unit_and_reduce(L, x, d, delta=-1), dir | {d}),
            ]
            q_idxs = [cells_index[dim_q][q] for q in q_cells]
            con_list["X"].setdefault(d, []).append(
                (check, ("data", q_idxs[0])))
            con_list["X"].setdefault(-d, []
                                     ).append((check, ("data", q_idxs[1])))

    for idx, (x, dir) in enumerate(cells_dim[dim_q + 1]):
        check = ("Zcheck", idx)
        for d in dir:
            q_cells = [
                (x, dir - {d}),
                (add_unit_and_reduce(L, x, d, delta=+1), dir - {d}),
            ]
            q_idxs = [cells_index[dim_q][q] for q in q_cells]
            con_list["Z"].setdefault(d, []).append(
                (check, ("data", q_idxs[0])))
            con_list["Z"].setdefault(-d, []
                                     ).append((check, ("data", q_idxs[1])))

    nodes = {
        "data_qubits": data_qubits,
        "Xchecks": Xchecks,
        "Zchecks": Zchecks,
    }
    qcode = css_code(hx, hz)
    lx, lz = qcode.lx, qcode.lz

    return hx, hz, lx, lz, nodes, lin_order, con_list


def build_syndrome_cycle(
    con_list: Dict[str, Dict[int, List[Tuple[str, int]]]],
    Xchecks: Sequence[Tuple[str, int]],
    Zchecks: Sequence[Tuple[str, int]],
    data_qubits: Sequence[Tuple[str, int]],
    lin_order: Dict[Tuple[str, int], int],
    type: str = "compact",
) -> List[Tuple]:
    if type == "compact":
        return build_syndrome_cycle_compact(
            con_list, Xchecks, Zchecks, data_qubits, lin_order
        )
    if type == "starfish":
        return build_syndrome_cycle_starfish(
            con_list, Xchecks, Zchecks, data_qubits, lin_order
        )


def build_syndrome_cycle_compact(
    con_list: Dict[str, Dict[int, List[Tuple[str, int]]]],
    Xchecks: Sequence[Tuple[str, int]],
    Zchecks: Sequence[Tuple[str, int]],
    data_qubits: Sequence[Tuple[str, int]],
    lin_order: Dict[Tuple[str, int], int],
) -> List[Tuple]:
    cycle: List[Tuple] = []

    # Prep Xchecks, Prep Zchecks, IDLE data_qubits
    for q in Xchecks:
        cycle.append(("PrepX", q))
    for q in Zchecks:
        cycle.append(("PrepZ", q))
    for q in data_qubits:
        cycle.append(("IDLE", q))

    # CNOT data, IDLE 1/4 Checks
    for d in (i for i in range(-4, 5) if i != 0):
        for control, target in con_list["X"][d]:
            cycle.append(("CNOT", control, target))
        for target, control in con_list["Z"][d]:
            cycle.append(("CNOT", control, target))
        idle_X = set(Xchecks) - {c for c, _ in con_list["X"][d]}
        idle_Z = set(Zchecks) - {c for c, _ in con_list["Z"][d]}
        for c in idle_X | idle_Z:
            cycle.append(("IDLE", c))

    # Meas Xchecks, Zchecks, IDLE data_qubits
    for q in Xchecks:
        cycle.append(("MeasX", q))
    for q in Zchecks:
        cycle.append(("MeasZ", q))
    for q in data_qubits:
        cycle.append(("IDLE", q))

    return cycle


def build_syndrome_cycle_starfish(
    con_list: Dict[str, Dict[int, List[Tuple[str, int]]]],
    Xchecks: Sequence[Tuple[str, int]],
    Zchecks: Sequence[Tuple[str, int]],
    data_qubits: Sequence[Tuple[str, int]],
    lin_order: Dict[Tuple[str, int], int],
) -> List[Tuple]:
    cycle: List[Tuple] = []

    # Prep Xchecks, IDLE Zchecks, data_qubits
    for q in Xchecks:
        cycle.append(("PrepX", q))
    for q in Zchecks + data_qubits:
        cycle.append(("IDLE", q))

    # CNOT half data, IDLE 1/4 XChecks, Zchecks
    for d in (i for i in range(-4, 5) if i != 0):
        for control, target in con_list["X"][d]:
            cycle.append(("CNOT", control, target))
        idle_X = set(Xchecks) - {c for c, _ in con_list["X"][d]}
        idle_data = set(data_qubits) - {d for _, d in con_list["X"][d]}
        for c in idle_X | set(Zchecks) | idle_data:
            cycle.append(("IDLE", c))

    # Meas Xchecks, IDLE Zchecks, data_qubits
    for q in Xchecks:
        cycle.append(("MeasX", q))
    for q in Zchecks + data_qubits:
        cycle.append(("IDLE", q))

    # Prep Zchecks, IDLE Xchecks, data_qubits
    for q in Zchecks:
        cycle.append(("PrepZ", q))
    for q in Xchecks + data_qubits:
        cycle.append(("IDLE", q))

    # CNOT half data, IDLE 1/4 ZChecks, Xchecks
    for d in (i for i in range(-4, 5) if i != 0):
        for target, control in con_list["Z"][d]:
            cycle.append(("CNOT", control, target))
        idle_Z = set(Zchecks) - {c for c, _ in con_list["Z"][d]}
        idle_data = set(data_qubits) - {d for _, d in con_list["Z"][d]}
        for c in idle_Z | set(Xchecks) | idle_data:
            cycle.append(("IDLE", c))

    # Meas Zchecks, IDLE Xchecks, data_qubits
    for q in Zchecks:
        cycle.append(("MeasZ", q))
    for q in Xchecks + data_qubits:
        cycle.append(("IDLE", q))

    return cycle
