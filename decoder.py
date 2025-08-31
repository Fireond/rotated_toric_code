import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, hstack
from ldpc import BpOsdDecoder

# ----------------------------- Utilities ------------------------------------


def rank2(A: np.ndarray) -> int:
    """Binary rank over GF(2) (kept from original, small cleanup).
    A: (m, n) binary numpy array
    Returns: rank(A) over GF(2)
    """
    A = A.copy() % 2
    m, n = A.shape
    r = 0
    col = 0
    for row in range(m):
        # find pivot
        while col < n and A[row:, col].max() == 0:
            col += 1
        if col == n:
            break
        # swap pivot row
        pivot = row + np.argmax(A[row:, col])
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # eliminate below
        for rr in range(row + 1, m):
            if A[rr, col]:
                A[rr] ^= A[row]
        r += 1
        col += 1
    # eliminate above (optional for rank)
    return r


# ------------------------------ Noise ---------------------------------------


@dataclass
class NoiseModel:
    """Abstract noise model.
    Implement `sample_errors_for_gate` to yield a list of error events that
    immediately follow a given gate in the circuit description.
    """

    def sample_errors_for_gate(
        self, gate: Tuple, rng: np.random.Generator
    ) -> List[Tuple]:
        raise NotImplementedError


@dataclass
class DepolarizingNoise(NoiseModel):
    p_meas: float
    p_idle: float
    p_init: float
    p_cnot: float

    def sample_errors_for_gate(
        self, gate: Tuple, rng: np.random.Generator
    ) -> List[Tuple]:
        g = gate[0]
        out: List[Tuple] = []
        if g == "MeasX":
            if rng.random() <= self.p_meas:
                out.append(("Z", gate[1]))
        elif g == "MeasZ":
            if rng.random() <= self.p_meas:
                out.append(("X", gate[1]))
        elif g == "IDLE":
            if rng.random() <= self.p_idle:
                ptype = rng.integers(3)
                out.append((("X", "Y", "Z")[ptype], gate[1]))
        elif g == "PrepX":
            if rng.random() <= self.p_init:
                out.append(("Z", gate[1]))
        elif g == "PrepZ":
            if rng.random() <= self.p_init:
                out.append(("X", gate[1]))
        elif g == "CNOT":
            if rng.random() <= self.p_cnot:
                # One of the 15 non-identity two-qubit Pauli errors
                t = rng.integers(15)
                lut = [
                    ("X", 1),
                    ("Y", 1),
                    ("Z", 1),
                    ("X", 2),
                    ("Y", 2),
                    ("Z", 2),
                    ("XX", None),
                    ("YY", None),
                    ("ZZ", None),
                    ("XY", None),
                    ("YX", None),
                    ("YZ", None),
                    ("ZY", None),
                    ("XZ", None),
                    ("ZX", None),
                ]
                label, which = lut[t]
                if which == 1:
                    out.append((label, gate[1]))
                elif which == 2:
                    out.append((label, gate[2]))
                else:
                    out.append((label, gate[1], gate[2]))
        return out


# ------------------------------ Simulation ----------------------------------


class CircuitSimulator:
    def __init__(self, lin_order: Dict[Tuple[str, int], int]):
        self.lin_order = lin_order
        self.n = len(lin_order)

    def _simulate(
        self, C: Sequence[Tuple], measure_axis: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict, int]:
        """Generic simulator for X- or Z-channel.
        measure_axis: 'X' to collect X-check syndromes (Z-channel),
                      'Z' to collect Z-check syndromes (X-channel).
        """
        syndrome_history: List[int] = []
        syndrome_map: Dict[Tuple[str, int], List[int]] = {}
        state = np.zeros(self.n, dtype=int)
        err_cnt = 0
        syn_cnt = 0
        for gate in C:
            g = gate[0]
            if g == "CNOT":
                control = self.lin_order[gate[1]]
                target = self.lin_order[gate[2]]
                # Z-channel: Z flows from target to control; X opposite.
                if measure_axis == "X":
                    state[control] ^= state[target]
                else:  # 'Z'
                    state[target] ^= state[control]
            elif g == "PrepX" and measure_axis == "X":
                state[self.lin_order[gate[1]]] = 0
            elif g == "PrepZ" and measure_axis == "Z":
                state[self.lin_order[gate[1]]] = 0
            elif g == "MeasX" and measure_axis == "X":
                q = gate[1]
                qidx = self.lin_order[q]
                syndrome_history.append(state[qidx])
                syndrome_map.setdefault(q, []).append(syn_cnt)
                syn_cnt += 1
            elif g == "MeasZ" and measure_axis == "Z":
                q = gate[1]
                qidx = self.lin_order[q]
                syndrome_history.append(state[qidx])
                syndrome_map.setdefault(q, []).append(syn_cnt)
                syn_cnt += 1
            else:
                # error events are represented as extra pseudo-gates in C
                if g in ("Z", "Y") and measure_axis == "X":
                    state[self.lin_order[gate[1]]] ^= 1
                    err_cnt += 1
                elif g in ("X", "Y") and measure_axis == "Z":
                    state[self.lin_order[gate[1]]] ^= 1
                    err_cnt += 1
                elif g in ("ZX", "YX") and measure_axis == "X":
                    state[self.lin_order[gate[1]]] ^= 1
                    err_cnt += 1
                elif g in ("XZ", "XY") and measure_axis == "X":
                    state[self.lin_order[gate[2]]] ^= 1
                    err_cnt += 1
                elif g in ("XZ", "YZ") and measure_axis == "Z":
                    state[self.lin_order[gate[1]]] ^= 1
                    err_cnt += 1
                elif g in ("ZX", "ZY") and measure_axis == "Z":
                    state[self.lin_order[gate[2]]] ^= 1
                    err_cnt += 1
                elif g in ("ZZ", "YY", "YZ", "ZY") and measure_axis == "X":
                    state[self.lin_order[gate[1]]] ^= 1
                    state[self.lin_order[gate[2]]] ^= 1
                    err_cnt += 1
                elif g in ("XX", "YY", "XY", "YX") and measure_axis == "Z":
                    state[self.lin_order[gate[1]]] ^= 1
                    state[self.lin_order[gate[2]]] ^= 1
                    err_cnt += 1
                else:
                    pass
        return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

    def simulate_Z_channel(self, C: Sequence[Tuple]):
        return self._simulate(C, "X")  # measure Xchecks

    def simulate_X_channel(self, C: Sequence[Tuple]):
        return self._simulate(C, "Z")  # measure Zchecks


# -------------------------- Decoder preparation ------------------------------


def _sparsify_history(
    history: np.ndarray,
    syndrome_map: Dict,
    checks: Sequence[Tuple[str, int]],
    num_cycles: int,
    num_add_cycle: int,
) -> np.ndarray:
    h1 = history.copy()
    h2 = history.copy()
    for c in checks:
        pos = syndrome_map[c]
        for row in range(1, num_cycles + num_add_cycle):
            h2[pos[row]] ^= h1[pos[row - 1]]
    return h2


def prepare_decoder_data(
    *,
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    lin_order: Dict[Tuple[str, int], int],
    Xchecks: Sequence[Tuple[str, int]],
    Zchecks: Sequence[Tuple[str, int]],
    data_qubits: Sequence[Tuple[str, int]],
    cycle: List[Tuple],
    num_cycles: int,
    num_add_cycle: int = 1,
    error_rate: float = 1,
    save_title: str,
):
    num_checks = len(Xchecks)
    cycle_repeated = cycle * num_cycles

    # Enumerate single-fault circuits with probabilities (depolarizing split like original)
    def enumerate_faults(axis: str):
        circuits: List[List[Tuple]] = []
        probs: List[float] = []
        head: List[Tuple] = []
        tail: List[Tuple] = list(cycle_repeated)
        for gate in cycle_repeated:
            g = gate[0]
            assert g in ["CNOT", "IDLE", "PrepX", "PrepZ", "MeasX", "MeasZ"]
            # measurement fault of opposite axis
            assert len(gate) == 2 + int(g == "CNOT")
            if axis == "X" and g == "MeasZ":
                circuits.append(head + [("X", gate[1])] + tail)
                probs.append(error_rate)
            if axis == "Z" and g == "MeasX":
                circuits.append(head + [("Z", gate[1])] + tail)
                probs.append(error_rate)
            # move window
            head.append(gate)
            tail.pop(0)
            # init/idle faults
            if g == "PrepZ" and axis == "X":
                circuits.append(head + [("X", gate[1])] + tail)
                probs.append(error_rate)
            if g == "PrepX" and axis == "Z":
                circuits.append(head + [("Z", gate[1])] + tail)
                probs.append(error_rate)
            if g == "IDLE":
                # depolarizing splits 2/3 weight to axis of interest
                circuits.append(head + [(axis, gate[1])] + tail)
                probs.append(error_rate * 2 / 3)
            # CNOT faults: distribute equally among Pauli pairs that flip the tracked axis
            if g == "CNOT":
                if axis == "X":
                    # X on either, or XX, XY, YX, YY type that flips X-channel syndromes
                    for lab, pr in [
                        (("X", gate[1]), 4 / 15),
                        (("X", gate[2]), 4 / 15),
                        (("XX", gate[1], gate[2]), 4 / 15),
                    ]:
                        circuits.append(head + [lab] + tail)
                        probs.append(error_rate * pr)
                else:
                    # Z-channel counterparts
                    for lab, pr in [
                        (("Z", gate[1]), 4 / 15),
                        (("Z", gate[2]), 4 / 15),
                        (("ZZ", gate[1], gate[2]), 4 / 15),
                    ]:
                        circuits.append(head + [lab] + tail)
                        probs.append(error_rate * pr)
        return circuits, probs

    circuitsX, ProbX = enumerate_faults("X")
    circuitsZ, ProbZ = enumerate_faults("Z")

    sim = CircuitSimulator(lin_order)

    # Helper to build H matrices by grouping identical syndrome columns

    def build_channel(side: str):
        Hdict: Dict[Tuple[int, ...], List[int]] = {}
        probs: List[float] = ProbX if side == "X" else ProbZ
        circuits = circuitsX if side == "X" else circuitsZ
        for idx, circ in enumerate(circuits):
            if side == "X":
                hist, state, smap, err_cnt = sim.simulate_X_channel(
                    circ + cycle * num_add_cycle
                )
                state_data = [state[lin_order[q]] for q in data_qubits]
                final_log = (lz @ state_data) % 2
                hist = _sparsify_history(
                    hist, smap, Zchecks, num_cycles, num_add_cycle)
            else:
                hist, state, smap, err_cnt = sim.simulate_Z_channel(
                    circ + cycle * num_add_cycle
                )
                state_data = [state[lin_order[q]] for q in data_qubits]
                final_log = (lx @ state_data) % 2
                hist = _sparsify_history(
                    hist, smap, Xchecks, num_cycles, num_add_cycle)
            assert err_cnt == 1
            assert len(hist) == num_checks * (num_cycles + num_add_cycle)
            aug = np.hstack([hist, final_log])
            supp = tuple(np.nonzero(aug)[0])
            Hdict.setdefault(supp, []).append(idx)

        first_logical_row = len(hist)  # after last time-slice row
        Hcols = []
        Hdec = []
        ch_probs = []
        for supp, idxs in Hdict.items():
            col = np.zeros((first_logical_row + lx.shape[0], 1), dtype=int)
            col_short = np.zeros((first_logical_row, 1), dtype=int)
            if supp:
                col[list(supp), 0] = 1
                col_short[:, 0] = col[:first_logical_row, 0]
            Hcols.append(coo_matrix(col))
            Hdec.append(coo_matrix(col_short))
            ch_probs.append(np.sum([probs[i] for i in idxs]))
        H = hstack(Hcols) if Hcols else coo_matrix(
            (first_logical_row + lx.shape[0], 0))
        Hdec = hstack(Hdec) if Hdec else coo_matrix((first_logical_row, 0))
        return H, Hdec, ch_probs, first_logical_row

    HX, HdecX, probX, first_logical_rowX = build_channel("X")
    HZ, HdecZ, probZ, first_logical_rowZ = build_channel("Z")

    mydata = dict(
        HdecX=HdecX,
        HdecZ=HdecZ,
        probX=probX,
        probZ=probZ,
        cycle=cycle,
        num_cycles=num_cycles,
        num_add_cycle=num_add_cycle,
        lin_order=lin_order,
        Xchecks=list(Xchecks),
        Zchecks=list(Zchecks),
        data_qubits=list(data_qubits),
        HX=HX,
        HZ=HZ,
        lx=lx,
        lz=lz,
        first_logical_rowZ=first_logical_rowZ,
        first_logical_rowX=first_logical_rowX,
        error_rate=error_rate,
    )

    # save
    dirname = os.path.dirname(save_title)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(save_title, "wb") as fp:
        pickle.dump(mydata, fp)


# --------------------------- Monte Carlo runner ------------------------------


def generate_noisy_circuit(
    cycle_repeated: Sequence[Tuple], noise: NoiseModel, rng: np.random.Generator
) -> List[Tuple]:
    circ: List[Tuple] = []
    for gate in cycle_repeated:
        # pre or post errors depending on gate semantics
        # Here we follow your original convention: measurement/init errors are appended around op
        g = gate[0]
        if g in ("PrepX", "PrepZ", "CNOT"):
            circ.append(gate)
            circ.extend(noise.sample_errors_for_gate(gate, rng))
        elif g in ("MeasX", "MeasZ"):
            circ.extend(noise.sample_errors_for_gate(gate, rng))
            circ.append(gate)
        else:  # IDLE
            circ.extend(noise.sample_errors_for_gate(gate, rng))
        # no-op for anything else
    return circ


def monte_carlo_run(
    *,
    data_path: str,
    ntrials: int = 50_000,
    error_rate: float,
    bp_method: str = "minimum_sum",
    max_iter: int = 10_000,
    osd_method: str = "OSD_CS",
    osd_order: int = 7,
    ms_scaling_factor: float = 0.0,
    seed: Optional[int] = None,
    omp_thread_count: int = 4,
    verbosity: bool = False,
):
    with open(data_path, "rb") as fp:
        mydata = pickle.load(fp)
    HdecX = mydata["HdecX"]
    HdecZ = mydata["HdecZ"]
    probX = [x * error_rate for x in mydata["probX"]]
    probZ = [x * error_rate for x in mydata["probZ"]]
    lin_order = mydata["lin_order"]
    data_qubits = mydata["data_qubits"]
    Xchecks = mydata["Xchecks"]
    Zchecks = mydata["Zchecks"]
    cycle = mydata["cycle"]
    num_cycles = mydata["num_cycles"]
    num_add_cycle = mydata["num_add_cycle"]
    HX = mydata["HX"]
    HZ = mydata["HZ"]
    lx = mydata["lx"]
    lz = mydata["lz"]
    first_logical_rowZ = mydata["first_logical_rowZ"]
    first_logical_rowX = mydata["first_logical_rowX"]

    # Build repeated cycle
    cycle_repeated = cycle * num_cycles

    # Setup decoders
    bpdX = BpOsdDecoder(
        HdecX,
        error_channel=probX,
        # error_rate=error_rate,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        osd_method=osd_method,
        osd_order=osd_order,
    )
    bpdZ = BpOsdDecoder(
        HdecZ,
        error_channel=probZ,
        # error_rate=error_rate,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        osd_method=osd_method,
        osd_order=osd_order,
    )

    rng = np.random.default_rng(seed)
    sim = CircuitSimulator(lin_order)

    good = bad = 0
    for t in tqdm(range(ntrials), disable=True):
        # noise with scalar p for all types (depolarizing)
        noise = DepolarizingNoise(
            error_rate, error_rate, error_rate, error_rate)
        circ = generate_noisy_circuit(cycle_repeated, noise, rng)

        # Z-channel first
        hist_Z, state_Z, smap_Z, err_cnt_Z = sim.simulate_Z_channel(
            circ + cycle * num_add_cycle
        )
        state_data = [state_Z[lin_order[q]] for q in data_qubits]
        final_log = (lx @ state_data) % 2
        hist_Z = _sparsify_history(
            hist_Z, smap_Z, Xchecks, num_cycles, num_add_cycle)
        low_w = bpdZ.decode(hist_Z)
        guess_aug = (HZ @ low_w) % 2
        guess_log = guess_aug[first_logical_rowZ: first_logical_rowZ + lx.shape[0]]
        okZ = np.array_equal(guess_log, final_log)

        okX = False
        if okZ:
            hist_X, state_X, smap_X, err_cnt_X = sim.simulate_X_channel(
                circ + cycle * num_add_cycle
            )
            state_data = [state_X[lin_order[q]] for q in data_qubits]
            final_log = (lz @ state_data) % 2
            hist_X = _sparsify_history(
                hist_X, smap_X, Zchecks, num_cycles, num_add_cycle
            )
            low_w = bpdX.decode(hist_X)
            guess_aug = (HX @ low_w) % 2
            guess_log = guess_aug[first_logical_rowX: first_logical_rowX + lz.shape[0]]
            okX = np.array_equal(guess_log, final_log)

        if okZ and okX:
            good += 1
        else:
            bad += 1
        if bad >= 250:
            break

        if verbosity:
            print(f"{error_rate}\t{num_cycles}\t{t + 1}\t{bad}")

    return t + 1, bad
