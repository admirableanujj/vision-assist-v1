// frontend/src/pages/ZoneInventoryDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Camera, Sparkles, Flame, Layers, LogOut, Cpu, HardDrive, Video, Mic, MicOff, RefreshCw } from 'lucide-react';
import { useAudioWS } from '../hooks/useAudioWS';

export default function ZoneInventoryDashboard({ user, onLogout, onToggleCameraFeed }) {
  const [selectedZone, setSelectedZone] = useState('zone-1');
  const [searchQuery, setSearchQuery] = useState('');
  const [genieOutput, setGenieOutput] = useState('Awaiting your command...');
  
  // Real Data State from DB / Vector Store
  const [registeredObjects, setRegisteredObjects] = useState([]);
  const [identifiedObjects, setIdentifiedObjects] = useState([]);
  const [loading, setLoading] = useState(true);

  // Audio STT WebSocket Hook Connection
  const { isRecording, transcript, aiResponse, startRecording, stopRecording, wsConnected } = useAudioWS();

  // 1. Fetch Registered Items from Backend DB / Qdrant
  const fetchRegisteredObjects = async () => {
    try {
      const res = await fetch('/api/v1/items/registered', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('vision_assist_token')}` }
      });
      if (res.ok) {
        const data = await res.json();
        setRegisteredObjects(data.items || []);
      }
    } catch (err) {
      console.error("Failed to load registered objects:", err);
    }
  };

  // 2. Fetch Active Zone Objects Identified by YOLO / VisionTracker
  const fetchZoneIdentifiedObjects = async (zoneId) => {
    try {
      const res = await fetch(`/api/v1/zones/${zoneId}/detected`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('vision_assist_token')}` }
      });
      if (res.ok) {
        const data = await res.json();
        setIdentifiedObjects(data.detected_items || []);
      }
    } catch (err) {
      console.error("Failed to load active zone objects:", err);
    }
  };

  // Initial Load & Zone Switch Listener
  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchRegisteredObjects(),
      fetchZoneIdentifiedObjects(selectedZone)
    ]).finally(() => setLoading(false));

    // Poll live zone detections every 3 seconds
    const interval = setInterval(() => {
      fetchZoneIdentifiedObjects(selectedZone);
    }, 3000);

    return () => clearInterval(interval);
  }, [selectedZone]);

  // Sync AI Transcript Response from Audio Hook
  useEffect(() => {
    if (transcript) setSearchQuery(transcript);
    if (aiResponse) setGenieOutput(aiResponse);
  }, [transcript, aiResponse]);

  // 3. Search Database for Registered Item
  const handleSearchRegisteredItem = async (itemName) => {
    setSearchQuery(itemName);
    setGenieOutput(`Querying vector store for: "${itemName}"...`);
    
    try {
      const res = await fetch('/api/v1/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('vision_assist_token')}`
        },
        body: JSON.stringify({ query: itemName, zone_id: selectedZone })
      });
      const data = await res.json();
      setGenieOutput(data.message || `Found ${itemName} in ${data.zone || selectedZone}.`);
    } catch {
      setGenieOutput(`Unable to locate ${itemName} in active vector indices.`);
    }
  };

  // 4. Register Identified Object into DB Vector Store
  const handleRegisterIdentifiedItem = async (item) => {
    setGenieOutput(`Registering "${item.name}" into database vector store...`);
    try {
      const res = await fetch('/api/v1/items/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('vision_assist_token')}`
        },
        body: JSON.stringify({ 
          name: item.name, 
          image: item.image, 
          zone_id: selectedZone,
          confidence: item.confidence 
        })
      });
      if (res.ok) {
        setGenieOutput(`Successfully registered "${item.name}" to database persistence!`);
        fetchRegisteredObjects(); // Refresh left panel list
      }
    } catch {
      setGenieOutput(`Failed to register ${item.name}. Check backend log stream.`);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      
      {/* Top Navigation Header */}
      <header className="border-b border-slate-800 px-6 py-3.5 flex justify-between items-center bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded-xl">
            <Camera className="text-amber-400 w-5 h-5" />
          </div>
          <h1 className="font-bold text-lg tracking-tight flex items-center gap-2">
            Vision Assist <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full font-mono">v2.0 Pro</span>
          </h1>
        </div>

        <div className="flex items-center gap-6 text-xs">
          <div className="hidden md:flex items-center gap-4 text-slate-400">
            <span className="flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5 text-cyan-400" /> YOLOv8</span>
            <span className="flex items-center gap-1.5"><HardDrive className="w-3.5 h-3.5 text-indigo-400" /> Ollama</span>
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          </div>

          <div className="flex items-center gap-3 border-l border-slate-800 pl-6">
            <button
              onClick={onToggleCameraFeed}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg transition-colors border border-slate-700 flex items-center gap-1.5 font-medium cursor-pointer"
            >
              <Video className="w-3.5 h-3.5" /> Camera Stream
            </button>

            <span className="text-amber-300 font-medium">@{user?.username || 'operator'}</span>
            
            <button
              onClick={onLogout}
              className="p-2 bg-slate-800 hover:bg-rose-950/80 text-slate-400 hover:text-rose-400 transition-colors rounded-lg border border-slate-700 cursor-pointer"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Split Layout */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
        
        {/* Left Panel: Registered DB Objects */}
        <section className="lg:col-span-7 bg-slate-900/80 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-6">
              <h2 className="font-semibold text-sm text-slate-200 tracking-wide uppercase flex items-center gap-2">
                Registered Objects
                {loading && <RefreshCw className="w-3.5 h-3.5 text-cyan-400 animate-spin" />}
              </h2>
              <span className="text-xs text-slate-400 font-mono">PostgreSQL / Qdrant Active</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {registeredObjects.length > 0 ? (
                registeredObjects.map((item) => (
                  <div 
                    key={item.id}
                    onClick={() => handleSearchRegisteredItem(item.name)}
                    className="group bg-slate-950/70 border border-slate-800 hover:border-cyan-500/50 rounded-xl p-3 flex flex-col items-center justify-between cursor-pointer transition-all hover:shadow-lg hover:shadow-cyan-950/40"
                  >
                    <div className="w-full aspect-square bg-slate-900 rounded-lg overflow-hidden mb-3 border border-slate-800">
                      <img src={item.image || 'https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=150'} alt={item.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                    </div>
                    <div className="text-center">
                      <span className="block font-medium text-slate-100 text-sm">{item.name}</span>
                      <span className="text-[11px] text-amber-400/90 underline group-hover:text-amber-300">
                        (Click to query in zones)
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-3 text-center py-8 text-xs text-slate-500">
                  No registered items stored in database vector index yet.
                </div>
              )}
            </div>
          </div>

          {/* Audio Push-To-Talk Control Bar */}
          <div className="mt-8 bg-slate-950 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
            <button 
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex items-center gap-2 font-semibold px-5 py-2.5 rounded-lg text-sm transition shadow-lg cursor-pointer ${
                isRecording 
                  ? 'bg-rose-600 hover:bg-rose-500 text-white animate-pulse shadow-rose-900/30' 
                  : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30'
              }`}
            >
              {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              <span>{isRecording ? 'Listening... (Release)' : 'Push to Talk'}</span>
            </button>
            
            <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`} />
              WS Audio: {wsConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </section>

        {/* Right Panel: Objects Identified in Active Zone */}
        <section className="lg:col-span-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between shadow-xl space-y-6">
          <div>
            <div className="flex justify-between items-center border-b border-slate-800 pb-3 mb-4">
              <h2 className="font-semibold text-sm text-slate-200 tracking-wide uppercase flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" /> Objects Identified
              </h2>
              
              <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-700/80 px-3 py-1 rounded-lg text-xs">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <select 
                  value={selectedZone} 
                  onChange={(e) => setSelectedZone(e.target.value)}
                  className="bg-transparent text-slate-200 focus:outline-none cursor-pointer"
                >
                  <option value="zone-1">Zone 1: Main Lobby</option>
                  <option value="zone-2">Zone 2: Server Room</option>
                  <option value="zone-3">Zone 3: Perimeter</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {identifiedObjects.length > 0 ? (
                identifiedObjects.map((obj) => (
                  <div 
                    key={obj.id}
                    onClick={() => handleRegisterIdentifiedItem(obj)}
                    className="group bg-slate-950/80 border border-slate-800 hover:border-emerald-500/50 rounded-xl p-2 flex flex-col items-center justify-between cursor-pointer transition-all"
                  >
                    <div className="w-full aspect-square bg-slate-900 rounded-lg overflow-hidden mb-2 border border-slate-800">
                      <img src={obj.image || 'https://images.unsplash.com/photo-1583209814632-f72be6e06824?w=150'} alt={obj.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                    </div>
                    <div className="text-center">
                      <span className="block font-medium text-slate-200 text-xs">{obj.name}</span>
                      <span className="text-[10px] text-amber-400/90 underline group-hover:text-amber-300">
                        (Click to register)
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-3 text-center py-6 text-xs text-slate-500">
                  Scanning zone stream for objects...
                </div>
              )}
            </div>
          </div>

          {/* Genie Voice Intelligence Responses */}
          <div className="border-t border-slate-800 pt-4 space-y-3">
            <h3 className="font-semibold text-xs text-amber-300 flex items-center gap-1.5 uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Genie Voice Intelligence
            </h3>

            <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl text-xs">
              <span className="block text-[10px] uppercase text-slate-500 font-semibold mb-1">Your Voice Query</span>
              <p className="text-slate-300 italic">
                {searchQuery ? `"${searchQuery}"` : "Hold 'Push to Talk' and speak..."}
              </p>
            </div>

            <div className="bg-gradient-to-br from-indigo-950/50 to-slate-950 border border-amber-500/30 p-3 rounded-xl text-xs shadow-inner">
              <span className="block text-[10px] uppercase text-amber-400 font-semibold mb-1 flex items-center gap-1">
                <Flame className="w-3 h-3 text-amber-400" /> Genie Response
              </span>
              <p className="text-amber-100 leading-relaxed">
                {genieOutput}
              </p>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}