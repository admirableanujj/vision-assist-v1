// frontend/src/pages/CameraFeedDashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Camera, Video, VideoOff, Layers, LogOut, Cpu, HardDrive, LayoutGrid, Bot, Send, Volume2 } from 'lucide-react';
import { useVisionWS } from '../hooks/useVisionWS';

// Deliberately just these two (not a third YOLOE option) — the goal is a
// clear, direct comparison of base vs. fine-tuned detection on this
// project's 47-class lostfound scheme, not a third open-vocabulary mode.
const ENGINE_OPTIONS = ['yolo', 'finetuned'];
const ENGINE_LABELS = { yolo: 'YOLO', finetuned: 'Fine-tuned YOLO' };
const ENGINE_TOOLTIPS = {
  yolo: 'Base model, fixed 80 COCO classes',
  finetuned: 'Fine-tuned on this project\'s 47-class lostfound scheme (19 kept COCO classes + 28 new, e.g. keys/wallet/screwdriver; no person class)',
};

export default function CameraFeedDashboard({ user, onLogout, onToggleDashboard }) {
  const [zones, setZones] = useState([]);
  const [selectedZone, setSelectedZone] = useState('');
  const [isVideoActive, setIsVideoActive] = useState(true);
  const [engineChoice, setEngineChoice] = useState('yolo');
  const [assistantInput, setAssistantInput] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const [chatMessages, setChatMessages] = useState([
    { sender: 'assistant', text: 'Hello Operator. Stream monitoring active. How can I assist you?', timestamp: new Date().toLocaleTimeString() }
  ]);

  const chatEndRef = useRef(null);

  // Auto-scroll chat history to the bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Fetch zones associated with the logged-in user on mount
  useEffect(() => {
    const fetchUserZones = async () => {
      try {
        const res = await fetch('/api/v1/zones', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('vision_assist_token')}` }
        });
        if (res.ok) {
          const data = await res.json();
          if (data.zones && data.zones.length > 0) {
            setZones(data.zones);
            setSelectedZone(data.zones[0].id);
          }
        }
      } catch (err) {
        console.error("Failed to load user zones:", err);
        const fallback = [{ id: 1, name: 'Main Living Zone' }];
        setZones(fallback);
        setSelectedZone(fallback[0].id);
      }
    };
    fetchUserZones();
  }, []);

  const { videoRef, canvasRef, wsConnected, detections, activeEngine } = useVisionWS(isVideoActive, engineChoice, selectedZone);

  // Text-to-Speech (TTS) Browser Speech Synthesis Helper
  const speakText = (text) => {
    if (!('speechSynthesis' in window)) return;
    
    window.speechSynthesis.cancel(); // Stop any ongoing speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const handleAssistantSend = (e) => {
    e.preventDefault();
    if (!assistantInput.trim()) return;

    const userMsg = assistantInput;
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newMessages = [
      ...chatMessages,
      { sender: 'user', text: userMsg, timestamp: timeStr }
    ];
    
    setChatMessages(newMessages);
    setAssistantInput('');

    // Simulate AI response synthesis
    setTimeout(() => {
      const activeLabels = detections.map(d => d.label).join(', ');
      const responseText = activeLabels 
        ? `In zone ID ${selectedZone}, I currently detect: ${activeLabels}.`
        : `Monitoring stream in zone ID ${selectedZone}. All parameters nominal.`;

      const aiTimeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setChatMessages((prev) => [
        ...prev, 
        { sender: 'assistant', text: responseText, timestamp: aiTimeStr }
      ]);

      // Speak the assistant response aloud automatically
      speakText(responseText);
    }, 600);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      
      {/* Header Bar */}
      <header className="border-b border-slate-800 px-6 py-3.5 flex justify-between items-center bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded-xl">
            <Camera className="text-amber-400 w-5 h-5" />
          </div>
          <h1 className="font-bold text-lg tracking-tight flex items-center gap-2">
            Vision Assist <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2 py-0.5 rounded-full font-mono">Live Stream</span>
          </h1>
        </div>

        <div className="flex items-center gap-6 text-xs">
          <div className="hidden md:flex items-center gap-4 text-slate-400">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <div className="flex items-center gap-0.5 bg-slate-800 rounded-full p-0.5">
                {ENGINE_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setEngineChoice(opt)}
                    title={ENGINE_TOOLTIPS[opt]}
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide transition cursor-pointer ${
                      engineChoice === opt ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {ENGINE_LABELS[opt]}
                  </button>
                ))}
              </div>
            </div>
            <span className="flex items-center gap-1.5"><HardDrive className="w-3.5 h-3.5 text-indigo-400" /> Ollama</span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          </div>

          <div className="flex items-center gap-3 border-l border-slate-800 pl-6">
            <button
              onClick={onToggleDashboard}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg transition-colors border border-slate-700 flex items-center gap-1.5 font-medium cursor-pointer"
            >
              <LayoutGrid className="w-3.5 h-3.5" /> Zone Inventory
            </button>
            <span className="text-amber-300 font-medium">@{user?.username || 'operator'}</span>
            <button onClick={onLogout} className="p-2 bg-slate-800 hover:bg-rose-950/80 text-slate-400 hover:text-rose-400 transition-colors rounded-lg border border-slate-700 cursor-pointer" title="Logout">
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid Content */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
        
        {/* Left: Video Feed */}
        <section className="lg:col-span-7 bg-slate-900/80 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
                <span className="text-xs font-mono text-slate-300 uppercase tracking-wider">
                  {wsConnected ? `Streaming Zone ID: ${selectedZone}` : 'Disconnected'}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-700/80 px-2.5 py-1 rounded-lg text-xs">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <select 
                    value={selectedZone} 
                    onChange={(e) => setSelectedZone(Number(e.target.value))}
                    className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
                  >
                    {zones.length > 0 ? (
                      zones.map((zone) => (
                        <option key={zone.id} value={zone.id} className="bg-slate-900 text-slate-100">
                          {zone.name}
                        </option>
                      ))
                    ) : (
                      <option value="" disabled className="bg-slate-900 text-slate-400">Loading zones...</option>
                    )}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={() => setIsVideoActive(!isVideoActive)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                    isVideoActive ? 'bg-emerald-950/40 border-emerald-800 text-emerald-400' : 'bg-rose-950/40 border-rose-800 text-rose-300'
                  }`}
                >
                  {isVideoActive ? <Video className="w-3.5 h-3.5" /> : <VideoOff className="w-3.5 h-3.5" />}
                  {isVideoActive ? "Camera ON" : "Camera OFF"}
                </button>
              </div>
            </div>

            <div className="w-full aspect-[4/3] bg-slate-950 border border-slate-800 rounded-xl overflow-hidden relative flex items-center justify-center shadow-inner">
              {isVideoActive ? (
                <>
                  <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover" />
                  <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-slate-600 space-y-2 p-6">
                  <VideoOff className="w-12 h-12 text-slate-700" />
                  <p className="text-sm font-medium text-slate-500">Camera feed offline</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 bg-slate-950 border border-slate-800 rounded-xl p-3 flex justify-between items-center text-xs">
            <span className="text-slate-400 font-mono">Detections: <strong className="text-emerald-400 font-bold">{detections.length}</strong></span>
            <div className="flex gap-2">
              {detections.map((d, idx) => (
                <span key={idx} className="bg-slate-900 border border-slate-700 text-amber-300 px-2 py-0.5 rounded text-[11px]">
                  {d.label} ({(d.confidence * 100).toFixed(0)}%)
                </span>
              ))}
            </div>
          </div>
        </section>

        {/* Right: On-Screen Chat History & TTS Companion */}
        <section className="lg:col-span-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-5 flex flex-col justify-between shadow-xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-cyan-400" />
              <h3 className="font-semibold text-xs text-slate-200 uppercase tracking-wide">VisionAssist Companion</h3>
            </div>
            <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-mono bg-slate-950 px-2 py-1 rounded border border-slate-800">
              <Volume2 className={`w-3 h-3 ${isSpeaking ? 'text-emerald-400 animate-bounce' : 'text-slate-500'}`} />
              TTS: {isSpeaking ? 'Speaking...' : 'Ready'}
            </div>
          </div>

          {/* Scrollable Chat History Container */}
          <div className="flex-1 overflow-y-auto space-y-3 mb-3 pr-2 max-h-[360px]">
            {chatMessages.map((msg, index) => (
              <div
                key={index}
                className={`p-3 rounded-xl text-xs leading-relaxed max-w-[90%] flex flex-col ${
                  msg.sender === 'user'
                    ? 'bg-amber-500/20 text-amber-200 border border-amber-500/30 ml-auto'
                    : 'bg-slate-950 text-slate-300 border border-slate-800'
                }`}
              >
                <div className="flex justify-between items-center mb-1 text-[10px] text-slate-500 font-mono">
                  <span>{msg.sender === 'user' ? 'Operator' : 'VisionAssist'}</span>
                  <span>{msg.timestamp}</span>
                </div>
                <p>{msg.text}</p>
                {msg.sender === 'assistant' && (
                  <button 
                    onClick={() => speakText(msg.text)}
                    className="mt-2 self-start flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 cursor-pointer"
                  >
                    <Volume2 className="w-3 h-3" /> Replay Audio
                  </button>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={handleAssistantSend} className="flex gap-2">
            <input
              type="text"
              value={assistantInput}
              onChange={(e) => setAssistantInput(e.target.value)}
              placeholder="Ask assistant or query zone..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition-colors"
            />
            <button type="submit" className="p-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-lg transition-colors cursor-pointer">
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </section>

      </main>
    </div>
  );
}