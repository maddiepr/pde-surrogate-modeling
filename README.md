# Surrogate Modeling on Nonlinear PDEs

*Finite difference solvers and neural network surrogates for nonlinear PDEs, with applications to predictive modeling and simulation speedup.*

## Abstract

This project investigates the use of machine learning as a surrogate for high-fidelity numerical solvers of nonlinear partial differential equations (PDEs). Burgers’ equation is used as a representative test case, with the PDE solved via finite difference methods in Python and C++ to establish accurate reference solutions. These simulations produce training data for a neural network surrogate model implemented in PyTorch. The surrogate’s ability to reproduce solver outputs is evaluated across a range of initial and boundary conditions, with a focus on numerical stability, predictive accuracy, generalization, and computational efficiency. Comparative analyses between the traditional solver and the learned model include error quantification, stability visualization, and runtime profiling.

---

## Project Overview

This project demonstrates how machine learning can accelerate numerical simulations of nonlinear partial differential equations (PDEs). The workflow:

1. **Numerical Solver**: Implement finite difference methods (FDM) in Python and C++ to solve PDEs such as Burger's equation.
2. **Data Generation**: Produce simulation datasets under varied initial and boundary conditions.
3. **Surrogate Modeling**: Train a neural network in PyTorch to approximate PDE solver outputs.
4. **Evaluation** – Compare surrogate predictions against traditional solvers in terms of:
   - Numerical accuracy
   - Computational speed
   - Generalization to unseen scenarios
5. **Visualization**: Generate plots of solutions, stability regions, error heatmaps, and runtime benchmarks.

---

## Repository Structure

*(Planned - subject to change as implementations progresses)*

```text
surrogate-pde/
├── data/              # Generated training and test datasets
├── notebooks/         # Jupyter notebooks for exploration and visualization
├── src/
│ ├── cpp/             # High-performance C++ solver implementations
│ ├── python/          # Python-based solvers and ML models
│ ├── utils/           # Common utilities (I/O, plotting, etc.)
├── tests/             # Unit and integration tests
├── requirements.txt   # Python dependencies
├── CMakeLists.txt     # Build configuration for C++ code
└── README.md
```

---

## Methods

- **Finite Difference Schemes**: Upwind, Lax-Friedrichs, and other descretization methods.
- **Stability Analysis**: CFL condition verification and timestep restrictions.
- **Neural Network Surrogates**: Fully connected and convolutional architectures in PyTorch.
- **Performance Profiling**: CPU vs GPU benchmarks; C++ vs Python runtimes.

---

## Technology Stack

- **Languages**: Python, C++
- **Libraries**: NumPy, PyTorch, Matplotlib
- **Build Tools**: CMake
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest, Google Test

---

## Example Applications

- Real-time PDE prediction for engineering simulations.
- Reduced-order modeling in scientific computing.
- Benchmarking hybrid simulation-ML pipelines.

---

## Future Work

- Extend to more complex PDEs (e.g., Navier-Stokes, reaction-diffusion systems).
- Explore physics-informed neural networks (PINNs) as alternative surrogates.
- Add parameter sweep automation for large-scale experiments.

---

## License 

This project is license under the MIT License.











