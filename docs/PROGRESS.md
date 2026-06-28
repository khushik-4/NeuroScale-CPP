# Project Progress Timeline

A chronological record of milestones and features implemented during the development of the NeuroScale C++ project.

---

## July 2025
* **Designed Custom Matrix Foundation (`matrix.hpp`)**
  * Created a template-based C++ matrix class for managing 2D dynamic memory arrays.
  * Implemented element-wise matrix filling, randomization, and math functions.
  * Implemented matrix dot product (multiplication) and element-wise addition.
  * Added binary serialization to save and load matrix datasets to disk as `.mat` files.

---

## October 2025
* **Developed the Custom Neural Network Engine (`neural_network.hpp`)**
  * Implemented feedforward propagation logic layer-by-layer.
  * Added Mean Squared Error (MSE) cost function calculation.
  * Implemented Finite Difference Gradient Approximation to calculate weights and biases slopes.
  * Implemented gradient descent weight updates (`learn`) and verified the engine by successfully training the network to resolve the XOR logical operation.

---

## January 2026
* **Integrated Dataset Preprocessing & Loading Pipelines**
  * Integrated `stb_image` header libraries for reading grayscale images.
  * Built conversion routines mapping image pixels to normalized coordinates: $(x, y) \rightarrow \text{pixel intensity}$.
  * Created dynamic architecture loader parser files (`network.arch` and `layers.functions`) to initialize network sizes dynamically.
  * Conducted initial image mapping tests using standalone MNIST training samples.

---

## May 2026
* **Developed Real-Time raylib GUI Visualizer**
  * Integrated raylib 6.0 graphics window interface.
  * Visualized neural network nodes and weights in real-time, mapping weight strengths to connection line thickness and color.
  * Built a side-by-side rendering panel comparing point-filtered $28 \times 28$ original inputs against the $224 \times 224$ upscaled reconstructions.
  * Created statistical overlay boxes displaying epochs, loss, and training status.

---

## June 2026
* **Optimized Systems, Fixed Memory Leaks, & Added Interactive Features**
  * **Portable Header Refactoring** -> Replaced GCC-only headers (`<bits/stdc++.h>`) in `gym.cpp` with portable Standard C++ headers, resolving compiler dependency failures.
  * **VS Code Project Configuration** -> Updated compilation parameters to enforce C++17 template argument deductions.
  * **GPU VRAM Leak Fix** -> Replaced frame-by-frame texture creation routines with a single, pre-allocated GPU texture updated via `UpdateTexture()`, stabilizing system memory.
  * **Non-Crashing Target Swapping** -> Implemented key listeners `[1] - [5]` to switch targets in real-time. Added file-system checks to prevent crashes and display user-friendly warning overlays if images are missing.
  * **High-Fidelity Text Rendering** -> Integrated Microsoft's Segoe UI TTF smooth font loader using `LoadFontEx` and `DrawTextEx` to remove pixelation.
  * **Documentation Release** -> Wrote the comprehensive placement interview study guide (`PROJECT_EXPLAINED.md`).