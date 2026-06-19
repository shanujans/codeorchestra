import logging
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

# Hard cap on circuit size for local simulation -- a 2^n-dimensional
# statevector/measurement simulation gets expensive fast, and this needs
# to stay fast enough for a live demo.
MAX_QUBITS = 20
# COBYLA iterations for the classical optimization loop. Each iteration
# runs a full circuit simulation, so this is kept small enough to finish
# in a few seconds even at MAX_QUBITS, while still doing genuine
# variational optimization rather than a single fixed-parameter guess.
COBYLA_MAXITER = 30


class QuantumTestOptimizer:
    """
    QAOA-based test path optimizer.

    Problem: given N test functions that cover overlapping code paths,
    find the subset that maximizes total coverage while minimizing
    redundant (overlapping) coverage between selected tests. Classically
    this is a flavor of weighted maximum-coverage / minimum-redundancy
    set selection (NP-hard in general). QAOA encodes it as a QUBO-style
    Hamiltonian -- a linear reward term per test (its coverage breadth)
    and a quadratic penalty term per overlapping pair -- and variationally
    searches the 2^N-dimensional measurement distribution for a low-cost
    bitstring, with a classical optimizer (COBYLA) tuning the circuit's
    gamma/beta parameters across iterations rather than using fixed values.
    """

    def __init__(self, n_layers: int = 2, use_ibm_hardware: bool = False) -> None:
        self.sim = AerSimulator()
        self.n_layers = n_layers
        self.use_ibm_hardware = use_ibm_hardware
        self._ibm_backend = None

        if self.use_ibm_hardware:
            try:
                # Requires `pip install qiskit-ibm-runtime` and a saved
                # account (QiskitRuntimeService.save_account(...)) or
                # IBM Quantum Platform / IBM Cloud credentials available
                # in the environment.
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                self._ibm_backend = service.least_busy(operational=True, simulator=False)
                logger.info(f"IBM Quantum: connected, using real backend '{self._ibm_backend.name}'.")
            except Exception as e:
                logger.warning(
                    f"Could not connect to IBM Quantum hardware ({e}). "
                    f"Falling back to local AerSimulator only for this run."
                )
                self.use_ibm_hardware = False

    def _run_on_ibm_hardware(self, circuit: QuantumCircuit, shots: int = 1024,
                              progress_callback=None) -> Dict[str, int]:
        """Transpiles for and runs the given circuit on the connected real
        IBM backend, returning measurement counts. Only used for the final
        tuned circuit -- NOT inside the COBYLA loop, since real-hardware
        queue times make dozens of iterative calls impractical.

        progress_callback(event_dict): optional callable invoked once the
        job is submitted (before blocking on result), so callers can emit
        a live status event without waiting for the queue to clear.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        pm = generate_preset_pass_manager(backend=self._ibm_backend, optimization_level=1)
        isa_circuit = pm.run(circuit)

        sampler = Sampler(mode=self._ibm_backend)
        sampler.options.default_shots = shots
        job = sampler.run([isa_circuit])

        logger.info(
            f"Submitted job {job.job_id()} to real IBM backend "
            f"'{self._ibm_backend.name}' -- this may take a while if the backend has a queue."
        )

        # Notify caller immediately after submission so they can show a
        # live "waiting for IBM queue" status without blocking.
        if progress_callback:
            try:
                progress_callback({
                    "type": "quantum_submitted",
                    "job_id": job.job_id(),
                    "backend": self._ibm_backend.name,
                })
            except Exception:
                pass

        pub_result = job.result()[0]
        return pub_result.data.meas.get_counts()

    def build_overlap_matrix(self, tests: List[Dict[str, Any]]) -> np.ndarray:
        """Returns an NxN matrix where M[i][j] is the Jaccard overlap
        (intersection / union) between test i's and test j's 'covers'
        sets. Diagonal is left at 0 -- this matrix only represents
        pairwise redundancy; each test's own coverage "reward" is handled
        separately in optimize() before being folded into the QUBO matrix
        passed to build_qaoa_circuit/cost_function.
        """
        n = len(tests)
        matrix = np.zeros((n, n))
        cover_sets = [set(t.get("covers", [])) for t in tests]
        for i in range(n):
            for j in range(i + 1, n):
                union = cover_sets[i] | cover_sets[j]
                jaccard = len(cover_sets[i] & cover_sets[j]) / len(union) if union else 0.0
                matrix[i][j] = jaccard
                matrix[j][i] = jaccard
        return matrix

    def build_qaoa_circuit(self, n: int, gamma: List[float], beta: List[float],
                            cost_matrix: np.ndarray) -> QuantumCircuit:
        """Builds a p-layer QAOA ansatz (p = self.n_layers, so gamma/beta
        must each have length p). cost_matrix is a QUBO-style matrix:
        diagonal entries are linear per-test reward/penalty terms, off-
        diagonal entries are pairwise overlap penalties. Cost unitary
        applies RZ rotations for both; mixer unitary is the standard
        X-rotation mixer.
        """
        qc = QuantumCircuit(n)
        qc.h(range(n))

        for layer in range(self.n_layers):
            g = gamma[layer]
            b = beta[layer]

            # Cost unitary -- linear terms (diagonal)
            for i in range(n):
                weight = cost_matrix[i][i]
                if weight != 0:
                    qc.rz(2 * g * weight, i)

            # Cost unitary -- quadratic terms (pairwise overlap penalty),
            # implemented via the standard CX-RZ-CX ZZ-interaction pattern
            for i in range(n):
                for j in range(i + 1, n):
                    weight = cost_matrix[i][j]
                    if weight != 0:
                        qc.cx(i, j)
                        qc.rz(2 * g * weight, j)
                        qc.cx(i, j)

            # Mixer unitary
            for i in range(n):
                qc.rx(2 * b, i)

        qc.measure_all()
        return qc

    def cost_function(self, params: np.ndarray, n: int, cost_matrix: np.ndarray) -> float:
        """Builds + runs the circuit for the given params, then computes
        the expected QUBO cost over the measured bitstring distribution.
        Returned value is what scipy.optimize.minimize tries to minimize.
        """
        p = self.n_layers
        gamma = list(params[:p])
        beta = list(params[p:])

        qc = self.build_qaoa_circuit(n, gamma, beta, cost_matrix)
        transpiled = transpile(qc, self.sim)
        counts = self.sim.run(transpiled, shots=512).result().get_counts()
        total_shots = sum(counts.values())

        expected_cost = 0.0
        for bitstring, count in counts.items():
            # Qiskit's measurement bitstrings are little-endian relative
            # to qubit index ordering -- reverse so x[i] lines up with
            # qubit i / test i.
            x = [int(b) for b in bitstring[::-1]]
            c = 0.0
            for i in range(n):
                c += cost_matrix[i][i] * x[i]
                for j in range(i + 1, n):
                    c += cost_matrix[i][j] * x[i] * x[j]
            expected_cost += c * (count / total_shots)

        return expected_cost

    def optimize(self, tests: List[Dict[str, Any]],
                 progress_callback=None) -> List[Dict[str, Any]]:
        """
        1. Build the pairwise overlap matrix.
        2. Fold in per-test coverage-breadth reward to get a QUBO cost matrix.
        3. Variationally optimize gamma/beta via scipy.optimize.minimize (COBYLA).
        4. Run the final tuned circuit and take the most probable bitstring.
        5. Return the selected tests with quantum_selected=True and a qaoa_score.
        """
        n = len(tests)
        if n == 0:
            return tests

        if n > MAX_QUBITS:
            logger.warning(f"Too many tests ({n}) for local QAOA simulation. Capping to {MAX_QUBITS}.")
            tests = tests[:MAX_QUBITS]
            n = MAX_QUBITS

        overlap_matrix = self.build_overlap_matrix(tests)

        # Linear reward per test: broader coverage = more negative cost
        # (more desirable), normalized so it's on a comparable scale to
        # the [0, 1] Jaccard overlap penalties.
        weights = np.array([len(t.get("covers", [])) for t in tests], dtype=float)
        if weights.max() > 0:
            weights = weights / weights.max()

        cost_matrix = overlap_matrix.copy()
        for i in range(n):
            cost_matrix[i][i] = -weights[i]

        p = self.n_layers
        init_params = np.random.uniform(0, np.pi, size=2 * p)

        result = minimize(
            self.cost_function,
            init_params,
            args=(n, cost_matrix),
            method="COBYLA",
            options={"maxiter": COBYLA_MAXITER},
        )

        best_gamma = list(result.x[:p])
        best_beta = list(result.x[p:])

        logger.info(
            f"QAOA variational optimization complete after {result.nfev} circuit "
            f"evaluations -- final cost={result.fun:.4f}"
        )

        final_circuit = self.build_qaoa_circuit(n, best_gamma, best_beta, cost_matrix)

        if self.use_ibm_hardware and self._ibm_backend:
            try:
                counts = self._run_on_ibm_hardware(final_circuit, shots=1024,
                                                    progress_callback=progress_callback)
                logger.info(f"Final selection sampled from REAL IBM hardware ({self._ibm_backend.name}).")
            except Exception as e:
                logger.warning(f"IBM hardware run failed ({e}); falling back to local AerSimulator for final sampling.")
                transpiled = transpile(final_circuit, self.sim)
                counts = self.sim.run(transpiled, shots=1024).result().get_counts()
        else:
            transpiled = transpile(final_circuit, self.sim)
            counts = self.sim.run(transpiled, shots=1024).result().get_counts()

        best_bitstring = max(counts, key=counts.get)
        best_prob = counts[best_bitstring] / 1024
        selected_bits = best_bitstring[::-1]  # reverse little-endian -> qubit/test index order

        optimized_tests = []
        for i, bit in enumerate(selected_bits):
            if bit == "1":
                t = dict(tests[i])
                t["quantum_selected"] = True
                t["qaoa_score"] = float(best_prob)
                optimized_tests.append(t)

        if not optimized_tests and tests:
            logger.info("QAOA's most probable bitstring selected nothing; appending baseline test.")
            t = dict(tests[0])
            t["quantum_selected"] = True
            t["qaoa_score"] = 0.0
            optimized_tests.append(t)

        return optimized_tests
