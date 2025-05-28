# QuantumGen

# Quantum Circuit Generator
A machine learning-powered system that automatically generates quantum circuits from natural language descriptions using a fine-tuned T5 transformer model and IBM's Qiskit framework.

# Features

AI-Powered Circuit Generation: Generate quantum circuits from natural language prompts
Multiple Circuit Types: Support for Bell states, GHZ states, QFT, VQE, QAOA, quantum algorithms
Visual Circuit Diagrams: Automatic generation of PNG circuit visualizations
Web Interface: Django implementation for easy maintainance
JSON Dataset: Structured dataset of quantum circuits with metadata
Database Integration: includes full database support for circuit storage
Extensible Architecture: Easy to add new circuit types and patterns

# Architecture
The project consists of three main components:

Dataset Generation (i.py): Creates synthetic quantum circuit dataset using Qiskit blueprint circuits
