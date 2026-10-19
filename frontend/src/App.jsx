import React, { useState, useEffect } from 'react';
import { useVisionWS } from './hooks/useVisionWS';
import { useAudioWS } from './hooks/useAudioWS';
import { Camera, Mic, Activity, HardDrive, Cpu, LogOut, Flame, Sparkles, Video, VideoOff, Layers } from 'lucide-react';
import LoginModal from './components/LoginModal';

export default function App() {
  const [user, setUser] = useState(null);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [selectedZone, setSelectedZone] = useState('zone-1');

  // Check for stored JWT token on mount
  useEffect(() => {
    const token = localStorage.getItem('vision_assist_token');
    const savedUser = localStorage.getItem('vision_assist_user');

    if (token && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch {
        setIsLoginOpen(true);
      }
    } else {
      setIsLoginOpen(true);
    }
  }, []);

  // Hook handles webcam stream, websocket communication, and canvas bounding box rendering automatically when authenticated & active
  const { videoRef, canvasRef, wsConnected, detections } = useVisionWS(Boolean(user) && isCameraActive, selectedZone);
  const { isListening, transcript, genieResponse, audioWsConnected, startListening, stopListening } = useAudioWS();

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
          <h1 className="font-bold text-xl tracking-tight">
            Vision Assist <span className="text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full ml-2">v2.0 Pro</span>
          </h1>
        </div>

        {/* Header Controls: Zone Selector & System Status */}
        <div className="flex items-center gap-6 text-xs">
          {/* Zone Selector */}
          <div className="flex items-center gap-2 bg-slate-950/60 border border-slate-800 px-3 py-1.5 rounded-lg">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <select
              value={selectedZone}
              onChange={(e) => setSelectedZone(e.target.value)}
              className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="zone-1" className="bg-slate-900">Zone 1: Main Lobby</option>
              <option value="zone-2" className="bg-slate-900">Zone 2: Server Rack</option>
              <option value="zone-3" className="bg-slate-900">Zone 3: Perimeter</option>
            </select>
          </div>

          <div className="hidden md:flex items-center gap-4 text-slate-400">
            <span className="flex items-center gap-1.5"><Cpu className="w-4 h-4" /> YOLO</span>
            <span className="flex items-center gap-1.5"><HardDrive className="w-4 h-4" /> Ollama</span>
            <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} title="WebSocket Status" />
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
        {/* Left 2 Columns: Video Feed with Bounding Box Overlay & Camera On/Off Toggle */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden relative flex items-center justify-center min-h-[480px]">
          {isCameraActive ? (
            <>
              <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center text-slate-500 space-y-2">
              <VideoOff className="w-12 h-12 text-slate-600" />
              <p className="text-sm">Camera Stream Offline ({selectedZone})</p>
            </div>
          )}

          {/* Controls Overlay */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center bg-slate-950/80 backdrop-blur-md p-3 rounded-lg border border-slate-800">
            <div className="flex items-center gap-3">
              {/* Push-to-Talk Button */}
              <button
                onMouseDown={startListening}
                onMouseUp={stopListening}
                onTouchStart={startListening}
                onTouchEnd={stopListening}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition ${
                  isListening ? 'bg-rose-600 animate-pulse text-white shadow-lg shadow-rose-600/30' : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                }`}
              >
                <Mic className="w-4 h-4" /> {isListening ? "Listening..." : "Push to Talk"}
              </button>

              {/* Camera On/Off Toggle Button */}
              <button
                onClick={() => setIsCameraActive(!isCameraActive)}
                className={`p-2 rounded-lg border transition ${
                  isCameraActive 
                    ? 'bg-slate-800 hover:bg-slate-700 text-emerald-400 border-slate-700' 
                    : 'bg-rose-950/50 hover:bg-rose-900/50 text-rose-400 border-rose-800/50'
                }`}
                title={isCameraActive ? "Turn Camera Off" : "Turn Camera On"}
              >
                {isCameraActive ? <Video className="w-4 h-4" /> : <VideoOff className="w-4 h-4" />}
              </button>
            </div>

            <span className="text-xs text-slate-400">WS: {audioWsConnected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>

        {/* Right Column: Reasoning & Context Stream */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-6 overflow-y-auto">
          
          {/* Section 1: YOLO Computer Vision Stream */}
          <div>
            <h2 className="font-semibold text-sm text-slate-300 border-b border-slate-800 pb-2 mb-4 flex items-center justify-between">
              <span className="flex items-center gap-2"><Activity className="w-4 h-4 text-emerald-400" /> Spatial Vision Stream</span>
              <span className="text-[10px] text-cyan-400 uppercase font-mono bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-800/50">{selectedZone}</span>
            </h2>
            <div className="space-y-3">
              {!isCameraActive ? (
                <p className="text-xs text-slate-500 italic">Camera is currently powered off...</p>
              ) : detections.length === 0 ? (
                <p className="text-xs text-slate-500 italic">No objects currently detected in view...</p>
              ) : (
                detections.map((d, i) => (
                  <div key={i} className="bg-slate-950/50 border border-slate-800/80 p-3 rounded-lg text-xs flex justify-between items-center">
                    <span className="font-medium text-slate-200 capitalize">{d.label}</span>
                    <span className="text-emerald-400 font-mono">{(d.confidence * 100).toFixed(0)}% Match</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Section 2: Voice & Genie Reasoning Stream */}
          <div className="border-t border-slate-800 pt-4">
            <h2 className="font-semibold text-sm text-amber-300/90 border-b border-slate-800 pb-2 mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" /> Genie Voice Intelligence
            </h2>
            
            <div className="space-y-3 text-xs">
              {/* User Voice Transcript */}
              <div className="bg-slate-950/70 border border-indigo-900/40 p-3 rounded-lg">
                <span className="block text-[10px] uppercase tracking-wider text-indigo-300/60 font-semibold mb-1">
                  Your Voice Query
                </span>
                <p className="text-slate-300 italic">
                  {transcript ? `"${transcript}"` : "Hold 'Push to Talk' and speak..."}
                </p>
              </div>

              {/* Genie Reasoning Output */}
              <div className="bg-gradient-to-br from-indigo-950/40 to-slate-950/80 border border-amber-500/20 p-3 rounded-lg shadow-inner">
                <span className="block text-[10px] uppercase tracking-wider text-amber-400/80 font-semibold mb-1 flex items-center gap-1">
                  <Flame className="w-3 h-3 text-amber-400" /> Genie Response
                </span>
                <p className="text-amber-100/90 leading-relaxed">
                  {genieResponse || "Awaiting your command..."}
                </p>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}