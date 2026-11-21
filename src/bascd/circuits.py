import stim


def memory_circuit(basis: str, distance: int, rounds: int, p: float) -> stim.Circuit:
    """Generate a rotated memory with the exploratory simple_uniform preset."""
    if basis not in ("x", "z"):
        raise ValueError("basis must be x or z")
    if distance < 2 or rounds < 1:
        raise ValueError("distance must be at least 2 and rounds must be positive")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
