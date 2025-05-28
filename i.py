import json
import random
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Parameter
from qiskit.circuit.library import BlueprintCircuit

class BellStateCircuit(BlueprintCircuit):
    """Blueprint for Bell state preparation circuit."""
    
    def __init__(self, name=None):
        super().__init__(name=name or f"Bell_State")
        self._num_qubits = 2
        
    def _check_configuration(self, raise_on_failure=True):
        return True
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            
        self.h(0)
        self.cx(0, 1)
        
        self._is_built = True
        
    def get_description(self):
        return "A quantum circuit that creates a Bell state, which is a maximally entangled quantum state between two qubits."

class GHZCircuit(BlueprintCircuit):
    """Blueprint for GHZ state preparation circuit."""
    
    def __init__(self, num_qubits=3, name=None):
        super().__init__(name=name or f"GHZ_{num_qubits}")
        self._num_qubits = num_qubits
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 3
        if not valid and raise_on_failure:
            raise ValueError("GHZ circuit requires at least 3 qubits")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            
        self.h(0)
        for i in range(self._num_qubits - 1):
            self.cx(i, i+1)
        
        self._is_built = True
        
    def get_description(self):
        return f"A quantum circuit that creates a GHZ state, which is a maximally entangled state across {self._num_qubits} qubits."

class QFTCircuit(BlueprintCircuit):
    """Blueprint for Quantum Fourier Transform circuit."""
    
    def __init__(self, num_qubits=4, name=None):
        super().__init__(name=name or f"QFT_{num_qubits}")
        self._num_qubits = num_qubits
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 2
        if not valid and raise_on_failure:
            raise ValueError("QFT circuit requires at least 2 qubits")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            
        for i in range(self._num_qubits):
            self.h(i)
            for j in range(i+1, self._num_qubits):
                self.cp(2*np.pi/2**(j-i+1), j, i)
        
        for i in range(self._num_qubits//2):
            self.swap(i, self._num_qubits-i-1)
        
        self._is_built = True
        
    def get_description(self):
        return f"A {self._num_qubits}-qubit Quantum Fourier Transform (QFT) circuit, which is the quantum analogue of the discrete Fourier transform."

class VQECircuit(BlueprintCircuit):
    """Blueprint for Variational Quantum Eigensolver circuit."""
    
    def __init__(self, num_qubits=4, depth=2, name=None):
        super().__init__(name=name or f"VQE_{num_qubits}_d{depth}")
        self._num_qubits = num_qubits
        self._depth = depth
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 2 and self._depth >= 1
        if not valid and raise_on_failure:
            raise ValueError("VQE circuit requires at least 2 qubits and depth >= 1")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
        
        for d in range(self._depth):
            for i in range(self._num_qubits):
                self.rx(Parameter(f"θ_{d}_{i}_x"), i)
                self.ry(Parameter(f"θ_{d}_{i}_y"), i)
                self.rz(Parameter(f"θ_{d}_{i}_z"), i)
            
            for i in range(self._num_qubits-1):
                self.cx(i, i+1)
        
        self._is_built = True
        
    def get_description(self):
        return f"A {self._num_qubits}-qubit Variational Quantum Eigensolver (VQE) circuit with depth {self._depth}. VQE is a hybrid quantum-classical algorithm used to find ground state energies of molecules and materials."

class QAOACircuit(BlueprintCircuit):
    """Blueprint for Quantum Approximate Optimization Algorithm circuit."""
    
    def __init__(self, num_qubits=4, p=1, name=None):
        super().__init__(name=name or f"QAOA_{num_qubits}_p{p}")
        self._num_qubits = num_qubits
        self._p = p
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 2 and self._p >= 1
        if not valid and raise_on_failure:
            raise ValueError("QAOA circuit requires at least 2 qubits and p >= 1")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
        
        for i in range(self._num_qubits):
            self.h(i)
        
        for layer in range(self._p):
            for i in range(self._num_qubits-1):
                self.cx(i, i+1)
                self.rz(Parameter(f"γ_{layer}_{i}"), i+1)
                self.cx(i, i+1)
            
            for i in range(self._num_qubits):
                self.rx(Parameter(f"β_{layer}_{i}"), i)
        
        self._is_built = True
        
    def get_description(self):
        return f"A {self._num_qubits}-qubit Quantum Approximate Optimization Algorithm (QAOA) circuit with p={self._p}. QAOA is used to solve combinatorial optimization problems like MaxCut, TSP, or 3SAT."

class TeleportationCircuit(BlueprintCircuit):
    """Blueprint for Quantum Teleportation circuit."""
    
    def __init__(self, name=None):
        super().__init__(name=name or "Teleportation")
        self._num_qubits = 3
        
    def _check_configuration(self, raise_on_failure=True):
        return True
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            self.add_register(ClassicalRegister(2, "c"))
        
        self.rx(Parameter("θ"), 0)
        self.rz(Parameter("φ"), 0)
        
        self.h(1)
        self.cx(1, 2)
        
        self.cx(0, 1)
        self.h(0)
        self.measure([0, 1], [0, 1])
        self.cx(1, 2)
        self.cz(0, 2)
        
        self._is_built = True
        
    def get_description(self):
        return "A quantum teleportation circuit that transfers the quantum state of one qubit to another using entanglement and classical communication."

class BernsteinVaziraniCircuit(BlueprintCircuit):
    """Blueprint for Bernstein-Vazirani algorithm circuit."""
    
    def __init__(self, num_qubits=4, name=None):
        super().__init__(name=name or f"BV_{num_qubits}")
        self._num_qubits = num_qubits
        self._secret = [random.randint(0, 1) for _ in range(num_qubits-1)]
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 3
        if not valid and raise_on_failure:
            raise ValueError("Bernstein-Vazirani circuit requires at least 3 qubits")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            self.add_register(ClassicalRegister(self._num_qubits-1, "c"))
        
        oracle_qubit = self._num_qubits - 1
        self.x(oracle_qubit)
        self.h(oracle_qubit)
        
        for i in range(self._num_qubits - 1):
            self.h(i)
        
        for i, bit in enumerate(self._secret):
            if bit == 1:
                self.cx(i, oracle_qubit)
        
        for i in range(self._num_qubits - 1):
            self.h(i)
        
        self.measure(range(self._num_qubits - 1), range(self._num_qubits - 1))
        
        self._is_built = True
        
    def get_description(self):
        secret_str = ''.join(str(b) for b in self._secret)
        return f"A Bernstein-Vazirani algorithm circuit that determines a secret string ({secret_str}) with a single quantum query. Uses {self._num_qubits-1} input qubits and 1 oracle qubit."

class DeutschJozsaCircuit(BlueprintCircuit):
    """Blueprint for Deutsch-Jozsa algorithm circuit."""
    
    def __init__(self, num_qubits=4, balanced=True, name=None):
        super().__init__(name=name or f"DJ_{num_qubits}_{'balanced' if balanced else 'constant'}")
        self._num_qubits = num_qubits
        self._balanced = balanced
        if balanced:
            num_ones = 2**(num_qubits-1)
            self._oracle_pattern = random.sample(range(2**(num_qubits-1)), num_ones)
        
    def _check_configuration(self, raise_on_failure=True):
        valid = self._num_qubits >= 3
        if not valid and raise_on_failure:
            raise ValueError("Deutsch-Jozsa circuit requires at least 3 qubits")
        return valid
        
    def _build(self):
        if self._is_built:
            return
            
        if not self.qregs:
            self.add_register(QuantumRegister(self._num_qubits, "q"))
            self.add_register(ClassicalRegister(self._num_qubits-1, "c"))
        
        oracle_qubit = self._num_qubits - 1
        self.x(oracle_qubit)
        self.h(oracle_qubit)
        
        for i in range(self._num_qubits - 1):
            self.h(i)
        
        if not self._balanced:
            if random.choice([0, 1]) == 1:
                self.x(oracle_qubit)
        else:
            bit_strings = [format(i, f'0{self._num_qubits-1}b') for i in self._oracle_pattern]
            for bit_string in bit_strings:
                controls = [i for i, bit in enumerate(bit_string) if bit == '1']
                if controls:
                    for control in controls[:-1]:
                        self.cx(control, controls[-1])
                    self.cx(controls[-1], oracle_qubit)
                    for control in reversed(controls[:-1]):
                        self.cx(control, controls[-1])
        
        for i in range(self._num_qubits - 1):
            self.h(i)
        
        self.measure(range(self._num_qubits - 1), range(self._num_qubits - 1))
        
        self._is_built = True
        
    def get_description(self):
        function_type = "balanced" if self._balanced else "constant"
        return f"A Deutsch-Jozsa algorithm circuit that determines if a function is {function_type} with a single quantum query. Uses {self._num_qubits-1} input qubits and 1 oracle qubit."

def generate_circuit_code(circuit):
    """Generate code that would create the given circuit."""
    circuit_type = type(circuit).__name__
    circuit_code = ""
    
    if circuit_type == "BellStateCircuit":
        circuit_code = """
from qiskit import QuantumCircuit

# Create Bell state circuit
qc = QuantumCircuit(2, 2)

# Apply Hadamard to the first qubit
qc.h(0)

# Apply CNOT with control=first qubit, target=second qubit
qc.cx(0, 1)

# Optional: measure the qubits
qc.measure([0, 1], [0, 1])
"""
    elif circuit_type == "GHZCircuit":
        num_qubits = circuit._num_qubits
        circuit_code = f"""
from qiskit import QuantumCircuit

# Create GHZ state circuit for {num_qubits} qubits
qc = QuantumCircuit({num_qubits}, {num_qubits})

# Apply Hadamard to the first qubit
qc.h(0)

# Apply CNOT gates to create entanglement
for i in range({num_qubits-1}):
    qc.cx(i, i+1)

# Optional: measure all qubits
qc.measure(range({num_qubits}), range({num_qubits}))
"""
    elif circuit_type == "QFTCircuit":
        num_qubits = circuit._num_qubits
        circuit_code = f"""
from qiskit import QuantumCircuit
import numpy as np

# Create QFT circuit for {num_qubits} qubits
qc = QuantumCircuit({num_qubits})

# Define a function for QFT
def qft(circuit, n):
    for i in range(n):
        circuit.h(i)
        for j in range(i+1, n):
            circuit.cp(2*np.pi/2**(j-i+1), j, i)
    for i in range(n//2):
        circuit.swap(i, n-i-1)
    return circuit

# Apply QFT
qft(qc, {num_qubits})
"""
    elif circuit_type == "VQECircuit":
        num_qubits = circuit._num_qubits
        depth = circuit._depth
        circuit_code = f"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

# Create a hardware-efficient ansatz for VQE with {num_qubits} qubits and depth {depth}
qc = QuantumCircuit({num_qubits})

for d in range({depth}):
    for i in range({num_qubits}):
        theta_x = Parameter(f"θ_{{d}}_{{i}}_x")
        theta_y = Parameter(f"θ_{{d}}_{{i}}_y")
        theta_z = Parameter(f"θ_{{d}}_{{i}}_z")
        qc.rx(theta_x, i)
        qc.ry(theta_y, i)
        qc.rz(theta_z, i)
    for i in range({num_qubits}-1):
        qc.cx(i, i+1)
"""
    elif circuit_type == "QAOACircuit":
        num_qubits = circuit._num_qubits
        p = circuit._p
        circuit_code = f"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

# Create QAOA circuit for {num_qubits} qubits with p={p}
qc = QuantumCircuit({num_qubits})

for i in range({num_qubits}):
    qc.h(i)

for layer in range({p}):
    for i in range({num_qubits}-1):
        gamma = Parameter(f"γ_{{layer}}_{{i}}")
        qc.cx(i, i+1)
        qc.rz(gamma, i+1)
        qc.cx(i, i+1)
    for i in range({num_qubits}):
        beta = Parameter(f"β_{{layer}}_{{i}}")
        qc.rx(beta, i)
"""
    elif circuit_type == "TeleportationCircuit":
        circuit_code = """
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Create quantum teleportation circuit
qc = QuantumCircuit(3, 2)

theta = Parameter("θ")
phi = Parameter("φ")
qc.rx(theta, 0)
qc.rz(phi, 0)

qc.h(1)
qc.cx(1, 2)

qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])
qc.cx(1, 2)
qc.cz(0, 2)
"""
    elif circuit_type == "BernsteinVaziraniCircuit":
        num_qubits = circuit._num_qubits
        secret = circuit._secret
        secret_str = ','.join(str(b) for b in secret)
        circuit_code = f"""
from qiskit import QuantumCircuit

# Create Bernstein-Vazirani circuit for {num_qubits-1}-bit secret: [{secret_str}]
qc = QuantumCircuit({num_qubits}, {num_qubits-1})

oracle_qubit = {num_qubits - 1}
qc.x(oracle_qubit)
qc.h(oracle_qubit)
for i in range({num_qubits - 1}):
    qc.h(i)
secret = [{secret_str}]
for i, bit in enumerate(secret):
    if bit == 1:
        qc.cx(i, oracle_qubit)
for i in range({num_qubits - 1}):
    qc.h(i)
qc.measure(range({num_qubits - 1}), range({num_qubits - 1}))
"""
    elif circuit_type == "DeutschJozsaCircuit":
        num_qubits = circuit._num_qubits
        balanced = circuit._balanced
        circuit_code = f"""
from qiskit import QuantumCircuit

# Create Deutsch-Jozsa circuit for {'balanced' if balanced else 'constant'} function
qc = QuantumCircuit({num_qubits}, {num_qubits-1})

oracle_qubit = {num_qubits - 1}
qc.x(oracle_qubit)
qc.h(oracle_qubit)
for i in range({num_qubits - 1}):
    qc.h(i)
"""
        if not balanced:
            circuit_code += """# For constant function - either do nothing (f(x)=0) or flip oracle qubit (f(x)=1)
# This example implements f(x)=0 (do nothing)
"""
        else:
            circuit_code += """# For balanced function - implementation depends on specific balanced function
# This is a simplified example that flips the oracle qubit for half of the inputs
qc.cx(0, oracle_qubit)
"""
        
        circuit_code += f"""
for i in range({num_qubits - 1}):
    qc.h(i)
qc.measure(range({num_qubits - 1}), range({num_qubits - 1}))
"""
    
    return circuit_code

def generate_dataset_entry(circuit, circuit_index):
    """Create a dataset entry for the given circuit in the desired JSON format."""
    circuit_type = type(circuit).__name__.replace("Circuit", "")
    circuit_name = circuit.name if hasattr(circuit, "name") else f"{circuit_type}_{circuit_index}"
    
    qiskit_code = generate_circuit_code(circuit)
    description = circuit.get_description()
    
    requirements = []
    if hasattr(circuit, "_num_qubits"):
        num_qubits = circuit._num_qubits
        requirements.append(f"Create a {circuit_type} circuit with {num_qubits} qubits")
    else:
        requirements.append(f"Create a {circuit_type} circuit")
    
    if hasattr(circuit, "_depth"):
        depth = circuit._depth
        requirements.append(f"The circuit should have a depth of {depth}")
    
    if hasattr(circuit, "_p"):
        p = circuit._p
        requirements.append(f"Use {p} QAOA layer{'s' if p > 1 else ''}")
    
    current_date = datetime.now(timezone.utc).isoformat()
    authors = [{"username": "quantum_researcher"}]
    
    entry = {
        "id": str(uuid.uuid4()),
        "title": f"{circuit_type} Circuit: {circuit_name}",
        "authors": authors,
        "dateOfPublication": current_date,
        "dateOfLastModification": current_date,
        "categories": ["Quantum Computing", "Quantum Circuits", circuit_type],
        "tags": [circuit_type, "Quantum Circuit", "Qiskit"],
        "seoDescription": f"{description}",
        "requirements": requirements,
        "description": description,
        "qiskit_code": qiskit_code,
        "circuit_type": circuit_type,
        "num_qubits": circuit._num_qubits if hasattr(circuit, "_num_qubits") else 2,
        "attributes": {
            "depth": circuit._depth if hasattr(circuit, "_depth") else None,
            "p": circuit._p if hasattr(circuit, "_p") else None,
            "balanced": circuit._balanced if hasattr(circuit, "_balanced") else None
        },
        "references": [
            {
                "id": "qiskit_textbook",
                "type": "website",
                "title": "Qiskit Textbook",
                "url": "https://qiskit.org/textbook/"
            }
        ]
    }
    
    return entry

def generate_quantum_circuit_dataset(output_file="quantum_circuit_dataset.json", num_samples=100):
    """Generate a dataset of quantum circuits with descriptions and code."""
    dataset = []
    circuit_index = 0
    
    circuit_classes = [
        (BellStateCircuit, {}),
        (GHZCircuit, {"num_qubits": range(3, 11)}),
        (QFTCircuit, {"num_qubits": range(3, 9)}),
        (VQECircuit, {"num_qubits": range(2, 9), "depth": range(1, 6)}),
        (QAOACircuit, {"num_qubits": range(2, 9), "p": range(1, 4)}),
        (TeleportationCircuit, {}),
        (BernsteinVaziraniCircuit, {"num_qubits": range(3, 9)}),
        (DeutschJozsaCircuit, {"num_qubits": range(3, 9), "balanced": [True, False]})
    ]
    
    samples_per_type = max(1, num_samples // len(circuit_classes))
    
    for circuit_class, param_ranges in circuit_classes:
        for _ in range(samples_per_type):
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, range):
                    params[param_name] = random.choice(list(param_range))
                elif isinstance(param_range, list):
                    params[param_name] = random.choice(param_range)
            
            try:
                circuit = circuit_class(**params)
                entry = generate_dataset_entry(circuit, circuit_index)
                dataset.append(entry)
                circuit_index += 1
            except Exception as e:
                print(f"Error creating circuit {circuit_class.__name__} with params {params}: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} circuit samples saved to {output_file}")
    return dataset

if __name__ == "__main__":
    generate_quantum_circuit_dataset(num_samples=100)