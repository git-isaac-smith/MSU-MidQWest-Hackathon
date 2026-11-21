import json

import numpy as np
import pytest

from bascd.accuracy_campaign import REFERENCES, draw, recipe_from_profile, select, summarize, validate_case
from bascd.accuracy_decoders import circuit_hash, compile_candidate, load_frozen_decoder
from bascd.measure import Scenario
from bascd.hypergraph import HypergraphSpec, compile_hypergraph
from bascd.stats import shot_failures


def recipe():
    def channel(x, y, z):
        return dict(conditional_paulis={"I": 1-x-y-z, "X": x, "Y": y, "Z": z})
    return recipe_from_profile(json.dumps(dict(format_version=1, clock=channel(.04,.02,.01),
                                               selected=channel(.01,.02,.03))).encode())


def test_distance_selection_requires_complete_coverage():
    def row(d, **counts):
        return dict(metadata=dict(distance=d), failures={k: dict(failures=v) for k,v in counts.items()})
    choices, totals = select([row(3, native=3, belief=1, exact_small=0), row(3, native=2, belief=1),
                              row(5, native=3, belief=3)])
    assert choices == {"3": "belief", "5": "native"}
    assert "exact_small" not in totals["3"]


def test_summary_counts_a_reference_winner_once_per_scenario():
    rows, pairs = [], []
    choices = {"3":"native", "5":"belief"}
    for d in (3,5):
        n = 100
        failures = {"native":10, "bath_clock":10, "belief":6}
        winner = choices[str(d)]
        for ref in REFERENCES:
            rows.append(dict(decoder=ref, distance=d, shots=n, failures=failures[ref], median_batch_s=.001))
            delta = failures[winner] - failures[ref]
            pairs.append(dict(reference=ref, shots=n, recovered=max(0,-delta), lost=max(0,delta)))
    result = summarize(rows, pairs, choices)
    assert result["frozen"]["shots"] == 200
    assert result["frozen"]["failures"] == 16
    assert result["frozen"]["paired_vs_native"]["ler_difference"] == pytest.approx(-.02)


def test_composed_validation_predictions_match_deployed_paths_and_disjoint_splits(tmp_path):
    pytest.importorskip("stabilizer_qec")
    case = Scenario(5, .01, "x")
    settings = recipe()
    row = validate_case(case, 443, 384, settings)
    circuit, params, data = draw(case, 443, "validation", 384)
    _, _, test = draw(case, 443, "final_test", 384)
    assert data.metadata["seed"] != test.metadata["seed"]
    assert data.metadata["sample_sha256"] == row["metadata"]["sample_sha256"] != test.metadata["sample_sha256"]
    raw, exhausted = compile_hypergraph(data.model, HypergraphSpec(**settings["search"])).predictions_and_low_confidence(data.detectors)
    views = {name: compile_candidate(circuit, params, settings, name).decode_batch(data.detectors)
             for name in ("native", "belief", "ensemble5", "bath_belief")}
    expected = {name: np.where(exhausted[:, None], views[base], raw)
                for name, base in (("search_native", "native"), ("search_belief", "belief"))}
    for name, other in (("dispute", "ensemble5"), ("belief_pair", "bath_belief")):
        expected[name] = np.where(views["belief"] == views[other], views["belief"], expected["search_belief"])
    for name in ("search_native", "search_belief", "dispute", "belief_pair"):
        decoder = compile_candidate(circuit, params, settings, name)
        predictions = decoder.decode_batch(data.detectors)
        np.testing.assert_array_equal(predictions, expected[name])
        assert int(shot_failures(predictions, data.truth).sum()) == row["failures"][name]["failures"]
        np.testing.assert_array_equal(decoder.decode_one(data.detectors[0]), predictions[0])
        np.testing.assert_array_equal(decoder.decode_batch(data.detectors[::12]), predictions[::12])
    frozen = dict(format_version=1, by_distance={"5":"belief_pair"}, recipe=settings,
                  circuits=[dict(sha256=circuit_hash(circuit), scenario=params)])
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen))
    deployed = load_frozen_decoder(path, circuit)
    np.testing.assert_array_equal(deployed.decode_batch(data.detectors[:8]), predictions[:8])
    circuit.append("TICK")
    with pytest.raises(ValueError, match="circuit"):
        load_frozen_decoder(path, circuit)
