# Real-Time CFD Parametric PINN Dashboard

A cutting-edge, interactive web application that evaluates a **Parametric Physics-Informed Neural Network (PINN)** entirely on the client side. The dashboard uses **WebAssembly (WASM)** to run real-time fluid dynamics inference directly in the browser, allowing users to manipulate fluid velocity and viscosity coefficients via UI sliders and see instantaneous flow-field deformations.

---

## 🚀 Live Demo
👉 **[Insert Your Vercel Live Link Here]**

---

## 💡 System Architecture

Unlike traditional Deep Learning surrogates that require massive datasets to train, this dashboard runs a model trained completely **data-free**. By leveraging a **Parametric PINN**, the neural network was trained by optimizing partial differential equations (Navier-Stokes) using automatic differentiation across a spectrum of boundary conditions.


🛠️ Core FeaturesClient-Side Physics Solver: Zero server dependency or GPU requirements. Fluid evaluation runs smoothly on local client CPUs via optimized WebAssembly execution providers.Dynamic Parameter Injections: Features a 4D parametric neural network that processes continuous spatial grids $(x, y)$ coupled with real-time UI scalar inputs $(V_{in}, \nu)$.Navier-Stokes Integrity: The underlying model evaluates real fluid phenomena—including mass conservation squeezing, stagnation zones, and velocity boundary layer development around a solid step obstacle.Instantaneous Rendering: High-resolution matrix data ($100 \times 40$ resolution grid) is mapped frame-by-frame to scientific Turbo color profiles using custom HTML5 Canvas operations.🏗️ Technical Stack & FrameworksFrontend Framework: React 18+ (Vite Build Pipeline)AI Runtime Environment: ONNX Runtime Web (onnxruntime-web) via WebAssembly (WASM) BackendInference Pipeline: Multi-dimensional Tensor mapping (Float32Array buffers)Styling & Presentation: Component-level CSS embedded with dark-mode scientific aesthetic layout rules
