# Real-Time CFD Parametric PINN Dashboard

A cutting-edge, interactive web application that evaluates a **Parametric Physics-Informed Neural Network (PINN)** entirely on the client side. The dashboard uses **WebAssembly (WASM)** to run real-time fluid dynamics inference directly in the browser, allowing users to manipulate fluid velocity and viscosity coefficients via UI sliders and see instantaneous flow-field deformations.

---

## 🚀 Live Demo
👉 **[Insert Your Vercel Live Link Here]**

---

## 💡 System Architecture

Unlike traditional Deep Learning surrogates that require massive datasets to train, this dashboard runs a model trained completely **data-free**. By leveraging a **Parametric PINN**, the neural network was trained by optimizing partial differential equations (Navier-Stokes) using automatic differentiation across a spectrum of boundary conditions.

```text
[User Sliders: Velocity & Viscosity]
                │
                ▼ (React State Trigger)
[ONNX Runtime Web (WASM Engine)] ⚛️ ──(Sub-millisecond Inference)──> [Raw Output Matrix [4000, 3]]
                                                                                    │
                                                                                    ▼
                                                                       [HTML5 Canvas 2D Viewport]
                                                                  (Dynamic Turbo-Jet Heatmap Rendering)
```

