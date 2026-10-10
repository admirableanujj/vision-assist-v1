import React, { useState, useEffect, useRef } from 'react';
import { useVisionWS } from './hooks/useVisionWS';
import { Camera, Mic, Activity, HardDrive, Cpu, LogOut, Flame, Sparkles } from 'lucide-react';
import LoginModal from './components/LoginModal';

export default function App() {
  const [user, setUser] = useState(null);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [detections, setDetections] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // 1. Check for stored JWT token on mount
  useEffect(() => {
    const token = localStorage.getItem('vision_assist_token');
    const savedUser = localStorage.getItem('vision_assist_user');

    if (token && savedUser) {
      setUser(JSON.parse(savedUser));
    } else {
      setIsLoginOpen(true);
    }
  }, []);

  // 2. Camera & WebSocket Initialization (Runs when authenticated)
  useEffect(() => {
    if (!user) return;

    // Access Webcam
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((err) => console.error("Webcam error:", err));

    // Connect WebSocket
    const ws = new WebSocket("ws://localhost:8000/ws/vision");

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetections(data.detections || []);
    };

    // Frame Capture Loop (Sends snapshot every 100ms)
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && videoRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);

        canvas.toBlob((blob) => {
          if (blob) ws.send(blob);
        }, "image/jpeg", 0.7);
      }
    }, 100);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [user]);

  // 3. Render bounding boxes over HTML5 canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(({ label, confidence, box }) => {
      const [x, y, w, h] = box;
      ctx.strokeStyle = "#10B981";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w - x, h - y);

      ctx.fillStyle = "#10B981";
      ctx.font = "14px Inter, sans-serif";
      ctx.fillText(`${label} (${(confidence * 100).toFixed(0)}%)`, x, y > 20 ? y - 5 : y + 15);
    });
  }, [detections]);

  // Handle Login Success
  const handleLoginSuccess = (userData) => {
    setUser(userData);
    localStorage.setItem('vision_assist_user', JSON.stringify(userData));
    setIsLoginOpen(false);
  };

  // Handle Logout
  const handleLogout = () => {
    localStorage.removeItem('vision_assist_token');
    localStorage.removeItem('vision_assist_user');
    setUser(null);
    setIsLoginOpen(true);
  };

  // If not logged in, render Unauthenticated Landing State + Genie Modal
  if (!user) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-center">
        <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-tr from-amber-500/20 to-cyan-500/20 border border-amber-400/40 mb-6 shadow-2xl shadow-amber-500/20">
          <Flame className="w-10 h-10 text-amber-400 animate-pulse" />
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-amber-200 via-amber-400 to-cyan-300 bg-clip-text text-transparent mb-3">
          Vision Assist v2.0 Pro
        </h1>
        <p className="text-slate-400 max-w-md mb-8 text-sm leading-relaxed">
          Spatial intelligence, voice reasoning, and real-time vision streams. Authenticate with the Genie to enter.
        </p>
        <button
          onClick={() => setIsLoginOpen(true)}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-amber-500 to-cyan-500 text-slate-950 font-semibold px-6 py-3 rounded-xl hover:opacity-90 transition-all shadow-lg shadow-amber-500/20"
        >
          <Sparkles className="w-5 h-5" /> Open Genie Portal
        </button>

        <LoginModal
          isOpen={isLoginOpen}
          onClose={() => setIsLoginOpen(false)}
          onLoginSuccess={handleLoginSuccess}
        />
      </div>
    );
  }

  // Main Dashboard View (Rendered after successful authentication)
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4 flex justify-between items-center bg-slate-900/50">
        <div className="flex items-center gap-3">
          <Camera className="text-amber-400 w-6 h-6" />
          <h1 className="font-bold text-xl tracking-tight">Vision Assist <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full ml-2">v2.0 Pro</span></h1>
        </div>
        <div className="flex items-center gap-6 text-xs">
          <div className="flex items-center gap-4 text-slate-400">
            <span className="flex items-center gap-1.5"><Cpu className="w-4 h-4" /> YOLO Engine</span>
            <span className="flex items-center gap-1.5"><HardDrive className="w-4 h-4" /> Ollama Local</span>
            <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
          </div>
          <div className="flex items-center gap-3 border-l border-slate-800 pl-6">
            <span className="text-amber-300 font-medium">@{user.username}</span>
            <button
              onClick={handleLogout}
              className="p-1.5 text-slate-400 hover:text-rose-400 transition-colors rounded-lg hover:bg-slate-800"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
        {/* Left 2 Columns: Video Feed with Bounding Box Overlay */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden relative flex items-center justify-center min-h-[480px]">
          <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
          <canvas ref={canvasRef} width={640} height={480} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />

          {/* Controls Overlay */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center bg-slate-950/80 backdrop-blur-md p-3 rounded-lg border border-slate-800">
            <button className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg font-medium text-sm transition">
              <Mic className="w-4 h-4" /> Push to Talk
            </button>
            <span className="text-xs text-slate-400">WS Status: {wsConnected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>

        {/* Right Column: Reasoning & Context Stream */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
          <div>
            <h2 className="font-semibold text-sm text-slate-300 border-b border-slate-800 pb-2 mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" /> Spatial Intelligence Stream
            </h2>
            <div className="space-y-3">
              {detections.length === 0 ? (
                <p className="text-xs text-slate-500 italic">No objects currently detected in view...</p>
              ) : (
                detections.map((d, i) => (
                  <div key={i} className="bg-slate-950/50 border border-slate-800/80 p-3 rounded-lg text-xs flex justify-between">
                    <span className="font-medium text-slate-200 capitalize">{d.label}</span>
                    <span className="text-emerald-400 font-mono">{(d.confidence * 100).toFixed(0)}% Match</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}