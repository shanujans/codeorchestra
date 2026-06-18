# Quantum Test Optimizer

CodeOrchestra's `TesterAgent` generates a list of candidate tests, each
covering specific code paths. Picking the *best* subset -- maximum coverage,
minimum redundancy -- is a flavor of minimum set cover, which is NP-hard:
classically, checking all subsets of N tests means searching 2^N
possibilities.

QAOA (Quantum Approximate Optimization Algorithm) encodes this as a QUBO-style
Hamiltonian: each test contributes a linear reward proportional to its
coverage breadth, and each overlapping pair of tests contributes a quadratic
penalty. A parameterized quantum circuit puts all N tests into superposition
and applies alternating cost/mixer layers, so the measurement distribution
concentrates probability on low-cost (high-coverage, low-redundancy) bitstrings.
Crucially, this isn't a single fixed-parameter guess: `scipy.optimize.minimize`
(COBYLA) classically tunes the circuit's gamma/beta parameters across multiple
iterations, each evaluated by simulating the circuit on `AerSimulator`, before
the final tuned circuit is sampled for the test selection.
