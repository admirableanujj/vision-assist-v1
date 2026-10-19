import React, { useState, useEffect } from 'react';
import LoginModal from './components/LoginModal';
import ZoneInventoryDashboard from './pages/ZoneInventoryDashboard';
import CameraFeedDashboard from './pages/CameraFeedDashboard';
import { Flame, Sparkles } from 'lucide-react';

export default function App() {
  const [user, setUser] = useState(null);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [activeView, setActiveView] = useState('inventory'); // 'inventory' | 'camera'

  // Check for stored token and user session on mount
  useEffect(() => {
    const token = localStorage.getItem('vision_assist_token');
    const savedUser = localStorage.getItem('vision_assist_user');

    if (token && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch {
        handleLogout();
      }
    } else {
      setIsLoginOpen(true);
    }
  }, []);

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    localStorage.setItem('vision_assist_user', JSON.stringify(userData));
    setIsLoginOpen(false);
    setActiveView('inventory');
  };

  const handleLogout = () => {
    localStorage.removeItem('vision_assist_token');
    localStorage.removeItem('vision_assist_user');
    setUser(null);
    setIsLoginOpen(true);
  };

  // If unauthenticated, present the landing state with the LoginModal open
  if (!user) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-6 text-center font-sans">
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
          className="inline-flex items-center gap-2 bg-gradient-to-r from-amber-500 to-cyan-500 text-slate-950 font-semibold px-6 py-3 rounded-xl hover:opacity-90 transition-all shadow-lg shadow-amber-500/20 cursor-pointer"
        >
          <Sparkles className="w-5 h-5" /> Summon Genie Portal
        </button>

        <LoginModal
          isOpen={isLoginOpen}
          onClose={() => {
            // Only allow closing if a user session is active
            if (user) setIsLoginOpen(false);
          }}
          onLoginSuccess={handleLoginSuccess}
        />
      </div>
    );
  }

  // Render Camera Stream View if toggled
  if (activeView === 'camera') {
    return (
      <CameraFeedDashboard
        user={user}
        onLogout={handleLogout}
        onToggleDashboard={() => setActiveView('inventory')}
      />
    );
  }

  // Render Zone Inventory Dashboard by default (Authenticated)
  return (
    <ZoneInventoryDashboard
      user={user}
      onLogout={handleLogout}
      onToggleCameraFeed={() => setActiveView('camera')}
    />
  );
}