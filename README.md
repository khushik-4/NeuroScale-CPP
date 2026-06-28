# NeuroScale C++: First-Principles Neural Network Image Upscaler

A native C++ image upscaling application built from scratch (no external ML libraries like TensorFlow or PyTorch) featuring a real-time training visualizer powered by the **raylib** graphics library.

---

## Key Features

* **Continuous Coordinate-Based Mapping**: Models images as a continuous function $f(x, y) \rightarrow \text{brightness}$ instead of a discrete pixel grid, enabling smooth, resolution-independent upscaling.
* **From-Scratch ML Engine**: Custom C++ implementation of matrix classes, neural layers, activation functions (ReLU, Sigmoid), and forward propagation.
* **Empirical Optimization**: Trains using **Finite Difference Gradient Approximation** rather than backpropagation, serving as a first-principles demonstration of gradient descent.
* **Real-time GUI Visualizer**: An interactive dashboard showing:
  * A live synapse-and-neuron diagram illustrating real-time updates to weights and biases.
  * Side-by-side comparison of the original $28 \times 28$ image against the real-time $224 \times 224$ upscaled reconstruction.
  * Live metrics panel (Epochs, Cost, and Progress).
* **Robust Dataset Handling**: Non-crashing switcher that allows swapping target MNIST digits in real-time with error notifications if resources are missing.

---

## Directory Structure

```text
├── docs/
│   ├── DESIGN CHOICES.md    -> Analysis of network configurations and performance
│   ├── PROGRESS.md          -> Completed milestones and history
│   └── TODO.md              -> Future features and tasks
├── include/
│   ├── matrix.hpp           -> Template class for 2D matrix math & file serialization
│   └── neural_network.hpp   -> Neural network structure, cost functions & finite diff math
├── resources/
│   └── font.ttf             -> Smooth TTF font for GUI text drawing
├── mnist/
│   └── train/               -> Contains sample MNIST PNG target images (100.png - 500.png)
├── main.cpp                 -> Main program driver, raylib event loop & rendering
├── gym.cpp                  -> Code for batch data loading/testing helper
├── network.arch             -> Config defining network layers (default: 2 8 8 1)
├── layers.functions         -> Config specifying activation functions per layer
└── README.md                -> This file
```

---

## Getting Started

### Prerequisites
* **Compiler**: MinGW C++ Compiler (GCC `g++` 11+ with C++17 support)
* **Libraries**: raylib (version 6.0 recommended)

### Build Instructions
Run the following compilation command in your terminal (adjusting your library paths as needed):
```bash
g++ -W main.cpp -I./include/ -L./include/ -IC:/additional-libs/raylib-6.0_win64_mingw-w64/raylib-6.0_win64_mingw-w64/include/ -LC:/additional-libs/raylib-6.0_win64_mingw-w64/raylib-6.0_win64_mingw-w64/lib/ -lraylib -lgdi32 -lwinmm -o ./debug/upscaler
```

### Execution
Run the compiled binary:
```bash
# Default paths
.\debug\upscaler.exe

# Custom parameters: <image_path> <arch_path> <functions_path> <matrix_save_path>
.\debug\upscaler.exe mnist/train/100.png network.arch layers.functions img.mat
```

### GUI Key Controls
* **[SPACE]** -> Pause / Resume training.
* **[R]** -> Re-randomize all weights and biases to restart training from scratch.
* **[1] - [5]** -> Switch instantly between the 5 loaded MNIST images (automatically restarts training for the selected digit).

---

## Datasets
This project uses grayscale handwritten digits from the **MNIST Database** (saved in standard $28\times 28$ PNG format). You can download more samples from Kaggle's [MNIST PNG Dataset](https://www.kaggle.com/datasets/jidhumohan/mnist-png).

---

## Project Documentation & Links

* **Technical Interview Prep Guide** -> [PROJECT_EXPLAINED.md](../PROJECT_EXPLAINED.md) — A detailed, first-principles document to study for interviews.
* **Design Choices** -> [DESIGN CHOICES.md](./docs/DESIGN%20CHOICES.md) — Architectural notes and activation explanations.
* **Progress Tracker** -> [PROGRESS.md](./docs/PROGRESS.md) — Project milestones.
* **Todo Roadmap** -> [TODO.md](./docs/TODO.md) — Planned roadmap.
