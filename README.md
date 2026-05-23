# Real-Time CFD Parametric PINN Dashboard

A cutting-edge, interactive web application that evaluates a **Parametric Physics-Informed Neural Network (PINN)** entirely on the client side. The dashboard uses **WebAssembly (WASM)** to run real-time fluid dynamics inference directly in the browser, allowing users to manipulate fluid velocity and viscosity coefficients via UI sliders and see instantaneous flow-field deformations.

---

## 🚀 Live Demo
https://cfd-ai-dashboard.vercel.app/  

---

## 💡 System Architecture

Unlike traditional Deep Learning surrogates that require massive datasets to train, this dashboard runs a model trained completely **data-free**. By leveraging a **Parametric PINN**, the neural network was trained by optimizing partial differential equations (Navier-Stokes) using automatic differentiation across a spectrum of boundary conditions.


🛠️ Core Features
- Client-Side Physics Solver: Zero server dependency or GPU requirements. Fluid evaluation runs smoothly on local client CPUs via optimized WebAssembly execution providers.
- Dynamic Parameter Injections: Features a 4D parametric neural network that processes continuous spatial grids $(x, y)$ coupled with real-time UI scalar inputs $(V_{in}, \nu)$.
- Navier-Stokes Integrity: The underlying model evaluates real fluid phenomena—including mass conservation squeezing, stagnation zones, and velocity boundary layer development around a solid step obstacle.
- Instantaneous Rendering: High-resolution matrix data ($100 \times 40$ resolution grid) is mapped frame-by-frame to scientific Turbo color profiles using custom HTML5 Canvas operations.

### 🏗️ Technical Stack & Frameworks
- Frontend Framework: React 18+ (Vite Build Pipeline)
- AI Runtime Environment: ONNX Runtime Web (onnxruntime-web) via WebAssembly (WASM) Backend
- Inference Pipeline: Multi-dimensional Tensor mapping (Float32Array buffers)
- Styling & Presentation: Component-level CSS embedded with dark-mode scientific aesthetic layout rules


📂 Repository Structure
cfd-ai-dashboard/
├── public/
│   ├── fluid_model.onnx       # Compiled PyTorch 2.x Neural Graph Topology
│   └── fluid_model.onnx.data  # Decoupled Binary Weight Matrix Parameters
├── src/
│   ├── App.jsx                # Core Logic (WASM Setup, Inference Loop, Canvas Renderer)
│   └── main.jsx               # React Dom Mounting Node
├── vercel.json                # Network Routing Configurations for External Binary Streams
├── package.json               # JavaScript Package Dependencies
└── README.md                  # Project Documentation



💻 Local Setup and Installation
Follow these steps to get the interactive dashboard running on your local machine:
1- Clone the repository:
git clone [https://github.com/YOUR_USERNAME/cfd-ai-dashboard.git](https://github.com/YOUR_USERNAME/cfd-ai-dashboard.git)
cd cfd-ai-dashboard
2- Install JavaScript dependencies:
npm install
3- Incorporate your model assets:
Ensure your generated fluid_model.onnx and fluid_model.onnx.data are located directly within the /public folder.
4- Launch the local development environment:
npm run dev
Open your browser and navigate to the local address printed in the terminal (typically http://localhost:5173).



🧠 Behind the Physics Model (PyTorch Backend)
The model deployed in this application was compiled using a custom-engineered Parametric Physics-Informed Neural Network (PINN) framework. The network architecture consists of dense layers mapped using hyperbolic tangent (Tanh) activation profiles.

Rather than training on traditional grid dataset files, the model minimizes residuals from the governing fluid mechanics equations directly via PyTorch's automatic differentiation graph loop:

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Continuity)}$$

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Continuity)}$$$$u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \quad \text{(X-Momentum)}$$


The model optimizes its loss function by dynamically sampling random velocity domains ($0.5\text{ m/s}$ to $1.5\text{ m/s}$) and viscosity parameters ($0.05$ to $0.2$) across every epoch, allowing it to generalize fluid flow transitions seamlessly.


