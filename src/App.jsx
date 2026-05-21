// src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

export default function App() {
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);
  const [velocity, setVelocity] = useState(1.0);
  const [viscosity, setViscosity] = useState(0.3);
  const [loading, setLoading] = useState(true);

  // Constants matching our training grid domain
  const Nx = 100;
  const Ny = 40;

  // 1. Initialize the ONNX Runtime Engine inside the browser
  useEffect(() => {
    async function initOnnx() {
      try {
        console.log("Fetching ONNX structural model...");
        const responseModel = await fetch('/fluid_model.onnx');
        const modelBuffer = await responseModel.arrayBuffer();
        
        console.log("Fetching companion weights data file...");
        const responseData = await fetch('/fluid_model.onnx.data');
        const dataBuffer = await responseData.arrayBuffer();
        
        console.log("Compiling unified session with external data...");
        const externalDataArray = new Uint8Array(dataBuffer);

        const runtimeSession = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          externalData: [
            {
              path: 'fluid_model.onnx.data',
              data: externalDataArray
            }
          ]
        });
        
        console.log("ONNX Session compiled successfully!");
        setSession(runtimeSession);
        setLoading(false);
      } catch (error) {
        console.error("Failed to compile browser ONNX runtime:", error);
      }
    }
    initOnnx();
  }, []);

  // 2. Perform Real-Time Inference loop whenever sliders are dragged
  useEffect(() => {
    if (!session) return;

    async function runInference() {
      const parametricGrid = [];

      // Loop over our spatial domain matrix mapping data rows
      for (let y = 0; y < Ny; y++) {
        for (let x = 0; x < Nx; x++) {
          const normalizedX = (x / (Nx - 1)) * 2.0;
          const normalizedY = (y / (Ny - 1)) * 1.0;
          
          // Inject 4D parameter array data row matching Python expected layout
          parametricGrid.push(normalizedX, normalizedY, velocity, viscosity);
        }
      }

      // Convert combined properties grid into an ONNX Tensor [4000, 4]
      const inputArray = new Float32Array(parametricGrid);
      const inputTensor = new ort.Tensor('float32', inputArray, [4000, 4]);

      // DYNAMIC FIX: Query the session manifest for the exact traced input name dynamically
      const activeInputKey = session.inputNames[0];
      const feeds = {};
      feeds[activeInputKey] = inputTensor;

      // Run real-time client-side evaluation pass
      const outputs = await session.run(feeds);
      
      const outputKey = Object.keys(outputs)[0];
      const rawPredictions = outputs[outputKey].data;

      // Extract velocity metrics to calculate resultant scalars magnitude
      const velocityMagnitudeArray = new Float32Array(Nx * Ny);
      for (let i = 0; i < Nx * Ny; i++) {
        const u = rawPredictions[i * 3 + 0];
        const v = rawPredictions[i * 3 + 1];
        velocityMagnitudeArray[i] = Math.sqrt(u * u + v * v);
      }

      drawFluidField(velocityMagnitudeArray);
    }

    runInference();
  }, [velocity, viscosity, session]);

  // 3. Map numerical prediction arrays to beautiful RGB canvas pixels
  const drawFluidField = (data) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cellWidth = canvas.width / Nx;
    const cellHeight = canvas.height / Ny;

    for (let y = 0; y < Ny; y++) {
      for (let x = 0; x < Nx; x++) {
        const dataIndex = y * Nx + x;
        const value = data[dataIndex];

        // Dynamic Turbo-style scientific jet color scale mapping
        const r = Math.floor(value * 255);
        const g = Math.floor(Math.sin(value * Math.PI) * 200);
        const b = Math.floor((1 - value) * 150);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x * cellWidth, (Ny - 1 - y) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
      }
    }

    // Explicitly overlay the solid step obstacle bounding box onto the canvas
    const obsXStart = (20 / Nx) * canvas.width;
    const obsXWidth = ((35 - 20) / Nx) * canvas.width;
    const obsHeight = (12 / Ny) * canvas.height;
    
    ctx.fillStyle = '#2c3e50'; 
    ctx.fillRect(obsXStart, canvas.height - obsHeight, obsXWidth, obsHeight);
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', height: '100vh', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0f172a', color: '#fff', fontFamily: 'sans-serif' }}>
        <h2>Loading Physics AI Core Engine...</h2>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif', padding: '2rem' }}>
      <header style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <h1 style={{ margin: '0 0 0.5rem 0', color: '#38bdf8' }}>Real-Time CFD Surrogate Dashboard</h1>
        <p style={{ margin: 0, color: '#94a3b8' }}>Neural Network Inference Running Entirely in the Browser Client via WebGL/ONNX</p>
      </header>

      <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem' }}>
        <div style={{ border: '4px solid #1e293b', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)' }}>
          <canvas ref={canvasRef} width={800} height={320} style={{ display: 'block' }} />
        </div>

        <section style={{ backgroundColor: '#1e293b', padding: '2rem', borderRadius: '12px', width: '100%', maxWidth: '600px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <label style={{ fontWeight: 'bold' }}>Inlet Velocity Bound</label>
              <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>{velocity.toFixed(2)} m/s</span>
            </div>
            <input 
              type="range" min="0.5" max="1.5" step="0.01" value={velocity} 
              onChange={(e) => setVelocity(parseFloat(e.target.value))}
              style={{ width: '100%', cursor: 'pointer' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <label style={{ fontWeight: 'bold' }}>Fluid Viscosity Coefficient</label>
              <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>{viscosity.toFixed(3)} N·s/m²</span>
            </div>
            <input 
              type="range" min="0.05" max="0.2" step="0.005" value={viscosity} 
              onChange={(e) => setViscosity(parseFloat(e.target.value))} // Fixed: Cleared redundant velocity updater hook
              style={{ width: '100%', cursor: 'pointer' }}
            />
          </div>
        </section>
      </main>
    </div>
  );
}