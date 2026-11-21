"""Offline, zero-temperature initialization for Sakidja's layered dual-rail bath."""

from dataclasses import astuple, dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

FIT_BOUNDS = np.array([
    (0.1, 0.5), (0.5, 0.99), (0.1, 0.5), (0.5, 0.99),
    (0.0001, 0.002), (0.0001, 0.002),
])
PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]).astype(complex),
}
PREPARATIONS = np.array([[2**-0.5, 2**-0.5], [1, 0], [0, 1]])
OBSERVABLES = np.stack([PAULIS[k] for k in ("X", "Y", "Z")])
ERASURE_PENALTY = 0.125


@dataclass(frozen=True)
class BathParameters:
    j_sb: float
    j_l1: float
    j_l12: float
    j_l2: float
    gamma_l1: float
    gamma_l2: float

    def __post_init__(self):
        values = np.asarray(astuple(self), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("bath couplings and emission rates must be finite and nonnegative")


@dataclass(frozen=True)
class BathChannel:
    time: float
    erasure_probability: float
    near_field_probability: float
    far_field_loss_probability: float
    worst_case_rail_return: float
    conditional_paulis: dict[str, float]
    frame_correction: np.ndarray

    @property
    def unflagged_pauli_probability(self) -> float:
        """Joint probability of passing the check and having a nonidentity Pauli."""
        return (1 - self.erasure_probability) * self.pauli_probability_given_pass

    @property
    def pauli_probability_given_pass(self) -> float:
        return sum(self.conditional_paulis[name] for name in "XYZ")

    @property
    def balanced_cost(self) -> float:
        return self.pauli_probability_given_pass + ERASURE_PENALTY * self.erasure_probability


class BathModel:
    def __init__(self, parameters: BathParameters, attachment: str = "one"):
        j = parameters
        if attachment == "one":
            rail_links = [(1, 2, j.j_sb), (1, 3, j.j_sb)]
        elif attachment == "split":
            rail_links = [(0, 2, 0.8 * j.j_sb), (1, 3, 1.2 * j.j_sb)]
        else:
            raise ValueError("attachment must be one or split")
        links = rail_links + [(2, 3, j.j_l1)]
        for a, b, factor in ((2, 4, .8), (2, 5, 1.1), (3, 5, .9), (3, 6, 1.2)):
            links.append((a, b, factor * j.j_l12))
        links += [(4, 5, .5 * j.j_l2), (5, 6, 1.5 * j.j_l2)]
        hamiltonian = np.zeros((7, 7))
        # Drop the constant energy of the one-excitation sector. Each XX+YY
        # term hops with amplitude 2J and each incident ZZ term shifts by -2J.
        for a, b, coupling in links:
            hamiltonian[a, b] = hamiltonian[b, a] = 2 * coupling
            hamiltonian[a, a] -= 2 * coupling
            hamiltonian[b, b] -= 2 * coupling
        losses = np.array([0, 0, j.gamma_l1, j.gamma_l1, j.gamma_l2, j.gamma_l2, j.gamma_l2])
        # Every emission puts the system in the vacuum. With no re-excitation,
        # the one-excitation block is the evolution conditioned on no emission.
        self.generator = -1j * hamiltonian - np.diag(losses / 2)
        self.rates, self.modes = np.linalg.eig(self.generator)
        self.input_modes = None
        if np.linalg.cond(self.modes) < 1e8:
            self.input_modes = np.linalg.inv(self.modes)[:, :2]

    def _transfer(self, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("times must be a finite nonnegative vector")
        if not len(times):
            return np.empty((0, 7, 2), dtype=complex)
        if self.input_modes is None:
            return np.array([expm(self.generator * t)[:, :2] for t in times])
        phase = np.exp(times[:, None] * self.rates)
        return (self.modes * phase[:, None, :]) @ self.input_modes

    def rail_transfer(self, times) -> np.ndarray:
        return self._transfer(times)[:, :2]

    def calibration_traces(self, times) -> np.ndarray:
        """Unnormalized Pauli expectations, time × preparation (plus,L,R) × (X,Y,Z)."""
        rail_states = self.rail_transfer(times) @ PREPARATIONS.T
        return np.einsum("tip,oij,tjp->tpo", rail_states.conj(), OBSERVABLES, rail_states).real

    @staticmethod
    def _channel(time, transfer, frame_correction=None):
        rail = transfer[:2]
        u, singular_values, vh = np.linalg.svd(rail)
        if frame_correction is None:
            frame_correction = (u @ vh).conj().T
        corrected = frame_correction @ rail
        # Squared Pauli amplitudes are the joint pass-and-Pauli probabilities.
        paulis = {name: float(abs(np.vdot(operator, corrected) / 2)**2)
                  for name, operator in PAULIS.items()}
        passed = sum(paulis.values())
        if passed <= np.finfo(float).tiny:
            raise ValueError("conditional Pauli channel is undefined at zero rail retention")
        return BathChannel(float(time), float(np.clip(1 - passed, 0, 1)),
                           float(np.clip(np.linalg.norm(transfer[2:])**2 / 2, 0, 1)),
                           float(np.clip(1 - np.linalg.norm(transfer)**2 / 2, 0, 1)),
                           float(np.clip(singular_values[-1]**2, 0, 1)),
                           {name: weight / passed for name, weight in paulis.items()}, frame_correction)

    def channel(self, time: float, *, frame_correction=None) -> BathChannel:
        if frame_correction is not None:
            frame_correction = np.asarray(frame_correction, dtype=complex)
            if (frame_correction.shape != (2, 2) or
                    not np.allclose(frame_correction.conj().T @ frame_correction, np.eye(2))):
                raise ValueError("frame correction must be a 2×2 unitary")
        return self._channel(time, self._transfer([time])[0], frame_correction)

    def initialize(self, clock_time: float, window_factor: float = 2.5, points: int = 151):
        if (not np.isfinite(clock_time) or clock_time <= 0 or
                not np.isfinite(window_factor) or window_factor < 1 or
                type(points) is not int or points < 2):
            raise ValueError("check time must be positive, window factor >= 1 and points >= 2")
        times = np.linspace(clock_time, clock_time * window_factor, points)
        channels = [self._channel(t, a) for t, a in zip(times, self._transfer(times), strict=True)]
        return channels[0], min(channels, key=lambda channel: channel.balanced_cost)


@dataclass(frozen=True)
class BathFit:
    parameters: BathParameters
    rmse: float
    optimizer_converged: bool
    starts_run: int
    evaluations: int


def fit_bath(times, traces, attachment="one", starts=6, max_evaluations=100, seed=17,
             initial: BathParameters | None = None) -> BathFit:
    """Refine an optional regressor estimate or use seeded bounded multistart."""
    times, traces = np.asarray(times, dtype=float), np.asarray(traces, dtype=float)
    if (times.ndim != 1 or len(times) < 20 or not np.isfinite(times).all() or times[0] < 0 or
            np.any(np.diff(times) <= 0) or traces.shape != (len(times), 3, 3) or
            not np.isfinite(traces).all()):
        raise ValueError("expected at least 20 increasing nonnegative times and finite (times, 3, 3) traces")
    if starts < 1 or max_evaluations < 1:
        raise ValueError("starts and evaluation budget must be positive")
    lo, hi = FIT_BOUNDS.T
    span = hi - lo
    rng = np.random.default_rng(seed)
    first = np.full(len(lo), 0.5) if initial is None else (np.array(astuple(initial)) - lo) / span
    if np.any(first < 0) or np.any(first > 1):
        raise ValueError("starting bath parameters must be within Table 1 bounds")

    def parameters(scaled):
        return BathParameters(*(lo + scaled * span))

    # Fit successively longer windows to reduce cycle slips. The final score
    # always uses the full supplied trace.
    windows = sorted({min(len(times), n) for n in (40, 120, len(times))})
    best = None
    evaluations = 0
    for index in range(starts):
        point = first if index == 0 else rng.uniform(0, 1, len(lo))
        for n in windows:
            def residual(x):
                model = BathModel(parameters(x), attachment)
                return (model.calibration_traces(times[:n]) - traces[:n]).ravel()

            attempt = least_squares(residual, point, bounds=(0, 1), max_nfev=max_evaluations,
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
            point = attempt.x
            evaluations += attempt.nfev
        rmse = float(np.sqrt(np.mean(attempt.fun**2)))
        if best is None or rmse < best.rmse:
            best = BathFit(parameters(point), rmse, bool(attempt.success), index + 1, evaluations)
        if best.rmse < 1e-8:
            break
    return BathFit(best.parameters, best.rmse, best.optimizer_converged, index + 1, evaluations)
