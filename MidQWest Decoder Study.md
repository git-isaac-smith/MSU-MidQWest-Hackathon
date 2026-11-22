# Decoding a surface code memory

Missouri State University Quantum Bears • MidQWest 2026 • Northrop Grumman decoder challenge

### Abstract

Our main result is **when to check a dual-rail qubit**. A photon can leave the rails and return from the nearby quantum bath. Dr. Ridwan Sakidja proposed retaining the time and phase of each qubit’s calibration signal so that its check occurs near a predicted return. At distance 7 in simulated Z memory, the original fixed clock failed on **3,317 of 20,000 shots (16.585%)**. Site-specific timing failed on **4 of 20,000 (0.020%)**. A larger independent run recorded **74 of 500,000 (0.0148%)**, with a 95% interval of **0.0118–0.0186%** for these five layouts. Even at the highest tested background noise, the timed check produced about one-third as many failures. Both schedules used loss-location flags and kept every shot. [Logical error and runtime sheet](output/pdf/Decoder%20error%20rates%20-%20bath%20timing.pdf) · [New shots and uncertainty](experiments/bath-rare-confirmation-02/README.md) · [Full stress sweep](experiments/bath-background-stress-01/README.md)

Changing the check schedule produced the largest error reduction we observed. On the separate fixed-circuit decoder benchmark, the distance-3 lookup table made 15.3% fewer errors than ordinary PyMatching and decoded batches 12.4 times faster. At distances 5 and 7, selective matching improved on ordinary matching at similar or better throughput. Always using correlated matching further lowered error and cost 2.9 to 3.4 times the selective decoding time. Across the full 480,000-shot grid, our distance-based hybrid made **13.9% fewer errors** than standard correlated matching and cost **24.2 times as much** in batch inference. These results quantify a decoder trade-off on identical detector records. [Small- and larger-code comparisons](experiments/circuit-frontier-01/results.md) · [Hybrid results](experiments/iteration-04/accuracy/summary.json)

### The same hardware and calibration signal

Northrop Grumman’s RQL-controlled superconducting dual-rail architecture gives the timing proposal a concrete hardware setting. The code stores a qubit in the one-excitation states |10⟩ and |01⟩. Having two rails supplies an orthogonal empty-rail state |00⟩. Measuring the total occupation can therefore flag loss while preserving the coherent choice of rail. Both simulated schedules use the same rail hardware, calibration-signal types, and number of checks. Even the fixed-clock comparison receives its calibrated frame correction and matching priors. What it gives up is the **time information when scheduling**. A rate such as T1 or T2 cannot specify when a particular return will occur. Northrop Grumman’s hardware results establish the control architecture; all error percentages here are simulations. [Northrop Grumman’s APS presentation](https://meetings-archive.aps.org/smt/2026/evw-t52/1/) · [Sakidja’s model](https://github.com/sakidja/RQL_Quantum_bath)

### Reading the return time with Fourier analysis

We tested whether a longer wait alone explains the result. On a fresh 20,000 shots per schedule, a single shared time of 15.00 chosen on 400 separate training fits gave **9.915%** logical error. A single shared time of 14.88 calibrated on the installed sites gave **9.645%**. Reassigning the individualized waits among sites left the distribution of waits identical in each layout but raised logical error from **0.020% to 9.460%**. At each assigned time the frame and decoder weights were recalibrated for the receiving site. Knowing **which time belongs to which qubit** mattered in these five layouts. The comparison covers two specified globally calibrated policies; it leaves grouped or other adaptive timing policies open. [Schedule controls and fresh-shot counts](experiments/bath-clock-controls-02/README.md)

For each of the 49 held-out baths, we converted simulated rail calibration traces into complex Fourier coefficients, retaining magnitude and phase. We reconstructed the rail return over the same 151 candidate times used by the fit and applied the same preselected cost for unflagged error and loss. The full FFT selected **all 49 fitted times exactly** in the ideal first-attachment model. They ranged from 6.36 to 15.00 model-time units. On bath 4000, the strongest occupation frequency was 0.3793 cycles per model-time unit and the selected check occurred at 13.44. Across the entire bank, the single strongest tone, even retaining its phase, matched only **4 of 49**. Several frequencies and their relative phases can determine a useful return.

The calibration record covers the candidate window, so this is a test of reconstructing a known device’s time dependence. With simulated measurement noise, **48.1%** of estimates at 1,000 shots per measured time and observable and **75.1%** at 10,000 landed within one grid step of the noiseless choice. A new device would need its own calibration, a suitable measured signal, and a check that the predicted returns persist. At the surface-code level we also use the record across rounds to characterize temporal structure. Ordinary matching already sees that record. Our comparisons show the effect of different decoding rules; they do not isolate memory as the sole source of the gain. [FFT traces, spectrum, and calibration shot test](experiments/bath-fft-timing-02/README.md) · [Surface-code spectral diagnostics](experiments/circuit-frontier-01/results.md)

We also used the FFT to supply all three controller inputs: check time, phase correction, and decoder weights. On 20,000 fresh shots this end-to-end FFT policy failed **6 times (0.030%)**. With FFT timing and fitted frames and weights, it failed twice; its five circuits were identical to the corresponding fitted-time circuits. The single-tone timing policy failed **396 times (1.980%)**. Two, four, and six failures are too few to rank the full FFT and fitted policies by those counts. The 500,000-shot confirmation measured the fitted policy. Neither finite-shot calibration nor the later dynamical-decoupling pulses were used in these surface-code runs. [End-to-end FFT simulation and circuit checks](experiments/bath-clock-controls-02/fft_result.json)

### Catching the return and holding it

Sakidja’s slides describe flipping the phase by π on **both** rails after the catch. The pulse leaves their relative phase intact at the instant it acts and reverses the rail amplitude relative to the bath. We tested a finite hold with the joint rail and bath state carried between pulses. In an ideal simulation of one additional site-specific return interval, mean final loss after a successful catch was **6.565%** with free evolution, **34.349%** with two pulses, and **0.153%** with 128 pulses. In a second interval, stopping the 128-pulse train allowed peak loss to rise to **38.401%**; continuing gave **0.279%**. All surface-code rates above come from check scheduling alone. The pulse simulation assumes ideal instantaneous control and covers two finite intervals. A hardware claim about operation until the next cooldown would require measured drift, pulse imperfections, and a longer-duration test. [Pulse-hold implementation and data](experiments/bath-pulse-hold-02/README.md)

### What one logical error looks like

A surface code spreads one unit of quantum information, a **logical qubit**, across several physical qubits. Neighboring groups undergo repeated checks. An unexpected change in a check produces a detection event. The collection of events over space and time is the **syndrome**, the evidence available to the decoder. Several different faults can produce the same evidence. The decoder needs to infer their effect on the stored information. [1, 2]

![Distance 3 Z memory circuit and gates on check qubit 18](experiments/iteration-04/screen-run-954131/circuit-overview.png)

*Figure 1. An actual 17-qubit benchmark circuit. Nine data qubits store the information. Eight check qubits help read its syndrome. The highlighted check collects information from four neighbors before it is measured and reset. The diagram shows one round of gates on that check.*

On the sixth trial of this circuit, one detector fired at the highlighted location in time step 2. The hybrid decoder predicted no logical flip (0). The simulator recorded a flip (1). That mismatch counted as one wrong logical prediction. The hybrid decoder missed 114 of 10,000 trials at this setting, an observed rate of **1.14%** with a 95% confidence interval of **0.95% to 1.37%**. [5]

## How we built and tested the decoder

### A fair comparison

We used Stim to generate noisy circuits and PyMatching to run the baseline decoders. Each circuit stored one logical qubit. A Z-memory experiment tests whether the decoder catches the effect of a logical bit flip. An X-memory experiment tests whether it catches the effect of a logical phase flip. We used code distances 3, 5 and 7. Distance is the minimum number of physical qubit errors in an undetectable logical operation on the code. Each memory ran as many check rounds as its distance. [1–3]

We tried four physical error probabilities per noise location, 0.1%, 0.3%, 0.5% and 1%. The provisional model, called `simple_uniform`, applies noise after gates, after resets, before measurements and to idle data qubits before each round. Combining three distances, two bases and four probabilities gives **24 circuit settings**. Within each setting, every decoder received identical syndrome records. Only the evaluator saw Stim’s recorded logical flips.

### From matching to a policy by distance

**Start with matching.** Minimum-weight perfect matching (MWPM) connects detection events to one another or to a boundary using likely fault paths. Correlated matching also uses information about faults that can cause several related events. An initial 2,000-shot smoke test checked the pipeline. On the subsequent 240,000-shot comparison, ordinary matching missed 6,545 shots and correlated matching missed 5,614. [6]

**Try cheaper decisions first.** We built a reproducible search over distance-dependent choices and weights for events across rounds or at boundaries. Eight randomly chosen policies established a control for future AI experiments. A separate adaptive decoder sent only selected shots to correlated matching. Routing by the number of events cut errors by 5.35% relative to ordinary matching and used about half the batch time of full correlated matching in its own held-out run. Full correlated matching still made fewer errors. [6]

**Choose the final rule on validation data.** At distance 3, a lookup table combines the modeled probabilities of faults for each possible syndrome and returns the more probable logical bit. The number of possible syndromes grows too quickly for that table at larger distances. At distances 5 and 7, a belief-matching decoder estimates which faults are likely, and five matching decoders vote on a second answer. If the two answers agree, the policy returns that bit. If they disagree, it searches plausible fault combinations up to a fixed limit, then falls back to the belief result if needed. Two of the five voters use bath-derived weights. [6]

We selected the hybrid decoder’s settings on 80,000 validation shots per distance and locked them before the final test. The deployed decoder takes detector events from a calibrated circuit and runs locally without a language model.

## Accuracy and computing time

**Table 1.** Each shot is one run of a noisy circuit. All three decoders processed the same 240,000 shots, with 10,000 per setting. The time column gives batch decoding time after warm-up divided by the number of shots, measured on the project host. This timing excludes compilation, sampling and scoring. [5]

| Decoder | Wrong logical predictions | Logical error rate | Batch µs per shot |
| --- | ---: | ---: | ---: |
| Standard correlated matching | 5,411 | 2.2546% | 14.66 |
| Bath-weighted correlated matching | 5,362 | 2.2342% | 14.74 |
| Distance-based hybrid decoder | 4,660 | 1.9417% | 380.16 |

Standard correlated matching uses the circuit’s fault probabilities. Bath-weighted correlated matching uses the same algorithm with weights derived from a bath model at a fixed check time. The hybrid decoder corrected 1,597 shots the standard matcher missed and missed 846 shots it got right. That gives **751 fewer failures** on the same trials, a **13.88% relative reduction**. The absolute reduction was **0.313 percentage points**, with an approximate paired 95% confidence interval of **0.273 to 0.353 points**. The hybrid took about **26 times as long per shot in batch**. A single request can take a different amount of time.

![Observed error rates by code distance and physical noise](experiments/iteration-04/screen-run-954131/verified-scenario-rates.png)

*Figure 2. At 0.5% physical error in X memory, the observed failure rate fell from 1.52% at distance 3 to 0.58% at distance 7. At 1% physical error the ordering reversed. “0 seen” means zero failures in 10,000 shots. Each such cell still has a Wilson 95% upper bound of 0.0384%.*

An earlier held-out test of **480,000 different shots** found the same accuracy ordering. The distance-based hybrid decoder missed 9,460 (1.9708%), belief matching missed 9,866 (2.0554%), bath-weighted correlated matching missed 10,781 (2.2460%), and standard correlated matching missed 10,982 (2.2879%). This test used 20,000 shots per setting. Both comparisons support an accuracy gain for the complete policy on the simulated circuits. [6]

## What a bath can tell a decoder

In Sakidja’s model, a qubit stores an excitation on two paths called rails. The excitation can move into its surroundings, the **bath**, and later return. A check asks whether the excitation is on the rails. A relaxation or coherence time summarizes a decay curve, but two curves with similar decay times can have different return times. Sakidja proposed fitting the time-dependent signal so that the check can be scheduled near a predicted return. [4]

### A timing experiment

We fitted six bath parameters to simulated calibration traces from three prepared states. The fit used 257 time points and held out 64 from the same synthetic bath. The model selected time 8.08 in place of the fixed check at 4.00, in model units. The predicted chance of finding the excitation off the rails fell from **60.57% to 23.54%**, at a cost of waiting about twice as long. These numbers describe erasure at a check. Surface-code logical failure is a different measurement. [6]

We then asked what a decoder could do with a flag identifying the site of an erasure. Each row below compares two decoders on the **same 20,000 shots** of a simulated distance-3 Z memory. Changing the time between rows changes the physical channel. [6]

**Table 2.** Logical failures with and without erasure locations in the bath simulation.

| Physical check time | Locations hidden | Locations supplied |
| --- | ---: | ---: |
| Fixed at 4.00 | 9,829 (49.145%) | 9,497 (47.485%) |
| Selected at 8.08 | 3,150 (15.750%) | 885 (4.425%) |

At the selected time, the flags prevented a net **2,265 failures**. The simulation assumes perfect flags, independent sites, one error opportunity per site and ideal checks. The proposal to follow the check with repeated phase pulses on both rails to suppress renewed exchange still needs an experimental test. [4, 6]

### Carrying the idea into the fixed circuit

The benchmark supplies detector bits on a fixed clock. We used the bath fit to change the relative weights the decoder gives to possible bit and phase errors during data-qubit idles. Stim still sampled the original noise at the original times. The decoder received no erasure flags. This tests whether the bath can supply useful initial guesses, or **priors**, when the physical circuit is already set.

In a dedicated 480,000-shot comparison, fixed-clock bath weights yielded **10,896** failures. An otherwise equivalent model with uniform weights yielded **10,953**. The estimated bath advantage was **0.0119 percentage points**, with a paired 95% interval from **−0.0084 to +0.0322**. That interval includes zero. The large gain with perfect erasure flags and the smaller, uncertain gain from changing weights answer different questions. The full decoder also uses a lookup table, voting and search, so its total gain cannot establish the bath’s individual contribution. [6]

## What the results mean for the challenge

Lead with the per-qubit timing test and its shuffled control. This is our largest demonstrated improvement, and it depends on scheduling loss checks on a dual-rail memory with observable loss locations. For fixed detector records the judges can instead compare predictions directly: show lookup, selective matching and the hybrid alongside their runtimes. Across three million paired fixed-circuit shots, a fixed-clock bath prior alone reduced error from **2.2607% to 2.2477%** against equal-axis correlated matching, or 390 fewer failures. This distinguishes the modest gain from changing decoder weights from the much larger gain available when the physical check can move. [Fixed-circuit confirmation](experiments/bath-confirmation-01/README.md)

On the circuits we could test, choosing a decoder by code distance reduced logical errors even against a strong belief-matching reference. The practical choice depends on the time available to act on each syndrome. Most of the computing cost came from the larger codes. In the 480,000-shot held-out test, distance-3 table lookup averaged about 0.03 µs per shot after compilation, compared with about 187 µs at distance 5 and 906 µs at distance 7. Independent shots can be divided among workers. A measured speedup curve on the judging machine is still needed. [6]

The organizer brief calls for decoder code and metric tests in a public repository, plus a roughly ten-minute explanation. It also values novelty, parallelizability and code quality. The supplied materials leave the official circuit noise and scoring formula open. Gate and readout fidelities describe individual hardware operations, while our percentages describe failure of an entire simulated logical memory. The next competition test is to freeze policies for the organizer’s circuits and measure both logical accuracy and latency on the judging hardware.

Sakidja also proposed an offline generative-AI search for decoder settings. The repository has a working random-search control, a structured proposal format and a shared scoring pipeline. Comparing an actual model-guided search against random search still requires equal trial budgets, common training and validation shots, and a fresh final test after both policies are frozen. [6]

### How to check our numbers

The audit regenerated all 240,000 shots from their circuits and seeds, reran the three decoders from detector inputs, and matched **all 720,000 predictions** to the saved bits. It reproduced the 72 scenario-level error rates and confidence intervals as well as all 48 paired comparisons. From the repository root, the following command repeats the audit in the project Python environment. [5]

```text
python scripts/audit_decoder_rates.py experiments/iteration-04/screen-run-954131
```

### References and project records

[1] A. deMarti iOlius et al. Decoding algorithms for surface codes. [arXiv:2307.14989v4](https://arxiv.org/abs/2307.14989v4).

[2] C. Gidney. Stim: a fast stabilizer circuit simulator. Quantum 5, 497 (2021). [Paper](https://doi.org/10.22331/q-2021-07-06-497).

[3] O. Higgott and contributors. PyMatching, the matching decoder used for the baselines. [Software](https://github.com/oscarhiggott/PyMatching).

[4] R. Sakidja. Erasure Check Timing from an Inferred Structured Bath: Dual Rail Qubits with Near Field Memory. [Author’s code and study materials](https://github.com/sakidja/RQL_Quantum_bath).

[5] Quantum Bears. Stim archive with root seed 954131. [Run report](experiments/iteration-04/screen-run-954131/Decoder%20results.md), [audit](experiments/iteration-04/screen-run-954131/error-rate-audit.json) and [per-setting counts](experiments/iteration-04/screen-run-954131/stats.csv).

[6] Quantum Bears. [Baseline](Iteration%2001.md), [search trials](Iteration%2002.md), [bath and routing experiments](Iteration%2003.md) and [policy selection and held-out tests](Iteration%2004.md).
