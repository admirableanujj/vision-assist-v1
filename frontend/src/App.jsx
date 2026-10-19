import React, { useState, useEffect } from 'react';
import AuthModal from './pages/AuthModal';
import ZoneInventoryDashboard from './pages/ZoneInventoryDashboard';
import CameraFeedDashboard from './pages/CameraFeedDashboard';

export default function App() {
  const [user, setUser] = useState(null);
  const [activeView, setActiveView] = useState('auth'); // 'auth' | 'inventory' | 'camera'

  useEffect(() => {
    const token = localStorage.getItem('vision_assist_token');
    const savedUser = localStorage.getItem('vision_assist_user');

    if (token && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
        setActiveView('inventory');
      } catch {
        setUser(null);
        setActiveView('auth');
      }
    }
  }, []);

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    localStorage.setItem('vision_assist_token', userData.token || 'mock_jwt_token_v2');
    localStorage.setItem('vision_assist_user', JSON.stringify(userData));
    setActiveView('inventory');
  };

  const handleLogout = () => {
    localStorage.removeItem('vision_assist_token');
    localStorage.removeItem('vision_assist_user');
    setUser(null);
    setActiveView('auth');
  };

  if (!user || activeView === 'auth') {
    return <AuthModal onLoginSuccess={handleLoginSuccess} />;
  }

  if (activeView === 'camera') {
    return (
      <CameraFeedDashboard
        user={user}
        onLogout={handleLogout}
        onToggleDashboard={() => setActiveView('inventory')}
      />
    );
  }

  return (
    <ZoneInventoryDashboard
      user={user}
      onLogout={handleLogout}
      onToggleCameraFeed={() => setActiveView('camera')}
    />
  );
}