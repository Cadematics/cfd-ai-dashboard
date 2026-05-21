# Real-Time CFD Parametric PINN Dashboard

A cutting-edge, interactive web application that evaluates a **Parametric Physics-Informed Neural Network (PINN)** entirely on the client side. The dashboard uses **WebAssembly (WASM)** to run real-time fluid dynamics inference directly in the browser, allowing users to manipulate fluid velocity and viscosity coefficients via UI sliders and see instantaneous flow-field deformations.

---

## 🚀 Live Demo
👉 **[Insert Your Vercel Live Link Here]**

---

## 💡 System Architecture

Unlike traditional Deep Learning surrogates that require massive datasets to train, this dashboard runs a model trained completely **data-free**. By leveraging a **Parametric PINN**, the neural network was trained by optimizing partial differential equations (Navier-Stokes) using automatic differentiation across a spectrum of boundary conditions.



$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Continuity)}$$$$u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \quad \text{(X-Momentum)}$$
