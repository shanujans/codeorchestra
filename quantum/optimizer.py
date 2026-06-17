import logging
from typing import List, Dict, Any
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)

class QuantumTestOptimizer:
    """
    Uses a real Quantum Approximate Optimization Algorithm (QAOA) circuit 
    via local AerSimulator to select the best subset of software tests.
    """
    def __init__(self) -> None:
        self.backend = AerSimulator()

    def optimize(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs a QAOA circuit to map maximum test coverage over minimal execution overlapping.
        """
        n_qubits = len(tests)
        if n_qubits == 0:
            return tests
            
        if n_qubits > 20: 
            logger.warning(f"Too many tests ({n_qubits}) for local QAOA simulation. Capping to 20.")
            tests = tests[:20]
            n_qubits = 20
        
        gamma, beta = 0.5, 0.5
        qc = QuantumCircuit(n_qubits)
        
        # 1. Initial superposition state
        for i in range(n_qubits):
            qc.h(i)
        
        # 2. Problem unitary: Maximize individual coverage, penalize intersection
        for i in range(n_qubits):
            weight = len(tests[i].get("covers", []))
            qc.rz(2 * gamma * weight, i)
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                covers_i = set(tests[i].get("covers", []))
                covers_j = set(tests[j].get("covers", []))
                overlap = len(covers_i.intersection(covers_j))
                if overlap > 0:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * overlap, j)
                    qc.cx(i, j)
        
        # 3. Mixer unitary
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
        
        qc.measure_all()
        
        logger.info(f"Transpiling and running QAOA circuit for {n_qubits} tests...")
        transpiled = transpile(qc, self.backend)
        result = self.backend.run(transpiled, shots=1024).result()
        counts = result.get_counts()
        
        # Select most frequent optimized bitstring
        best_bitstring = max(counts, key=counts.get)
        best_bitstring = best_bitstring[::-1]  # Reverse little-endian structure
        
        optimized_tests = []
        for i, bit in enumerate(best_bitstring):
            if bit == '1':
                t = dict(tests[i])
                t["quantum_selected"] = True
                t["qaoa_score"] = float(counts[best_bitstring[::-1]] / 1024)
                optimized_tests.append(t)
                
        # Failsafe if QAOA filters everything out
        if not optimized_tests and tests:
            logger.info("QAOA returned empty subset, appending baseline test.")
            t = dict(tests[0])
            t["quantum_selected"] = True
            t["qaoa_score"] = 0.0
            optimized_tests.append(t)
            
        return optimized_tests