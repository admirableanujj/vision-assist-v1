import React, { useState } from 'react';
import { Sparkles, KeyRound, User, Flame, Eye, EyeOff, AlertCircle } from 'lucide-react';

export default function LoginModal({ isOpen, onClose, onLoginSuccess }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage('');

    try {
      // Endpoint mapping to backend/app/api/auth_routes.py
      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'The Genie rejects these credentials.');
      }

      // Store JWT token / session details
      if (data.access_token) {
        localStorage.setItem('vision_assist_token', data.access_token);
      }

      onLoginSuccess(data.user || { username });
      onClose();
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4 animate-fade-in">
      {/* Mystical Genie Card */}
      <div className="relative w-full max-w-md bg-gradient-to-b from-indigo-950 via-slate-900 to-indigo-950 border border-amber-500/30 rounded-2xl shadow-2xl shadow-indigo-500/10 overflow-hidden p-8">
        
        {/* Glow Effects */}
        <div className="absolute -top-24 -left-24 w-48 h-48 bg-amber-500/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-cyan-500/20 rounded-full blur-3xl pointer-events-none" />

        {/* Header with Genie Lamp Emblem */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-tr from-amber-500/20 to-cyan-500/20 border border-amber-400/40 mb-3 shadow-lg shadow-amber-500/20 ring-4 ring-amber-500/10">
            <Flame className="w-8 h-8 text-amber-400 animate-pulse" />
          </div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-amber-200 via-amber-400 to-cyan-300 bg-clip-text text-transparent tracking-wide">
            Summon Vision Assist
          </h2>
          <p className="text-xs text-indigo-200/70 mt-1 font-sans">
            Speak your passphrase to awaken the spatial intelligence.
          </p>
        </div>

        {/* Error Alert Box */}
        {errorMessage && (
          <div className="mb-4 p-3 bg-rose-950/60 border border-rose-500/40 rounded-lg flex items-center gap-2 text-rose-300 text-xs animate-shake">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Username Input */}
          <div>
            <label className="block text-xs font-medium text-amber-200/80 mb-1">
              Identity (Username)
            </label>
            <div className="relative">
              <User className="absolute left-3 top-2.5 h-4 w-4 text-indigo-300/60" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. admin"
                className="w-full bg-slate-950/60 border border-indigo-800/50 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-100 placeholder-indigo-300/30 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/50 transition-all"
              />
            </div>
          </div>

          {/* Password Input */}
          <div>
            <label className="block text-xs font-medium text-amber-200/80 mb-1">
              Secret Passphrase
            </label>
            <div className="relative">
              <KeyRound className="absolute left-3 top-2.5 h-4 w-4 text-indigo-300/60" />
              <input
                type={showPassword ? 'text' : 'password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full bg-slate-950/60 border border-indigo-800/50 rounded-lg pl-9 pr-10 py-2 text-sm text-slate-100 placeholder-indigo-300/30 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/50 transition-all"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2.5 text-indigo-300/60 hover:text-amber-300 transition-colors"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-2 relative group overflow-hidden rounded-lg p-px font-semibold text-sm transition-all"
          >
            <span className="absolute inset-0 bg-gradient-to-r from-amber-500 via-amber-400 to-cyan-400 group-hover:opacity-90 transition-opacity" />
            <div className="relative px-4 py-2.5 rounded-[7px] bg-slate-950 text-amber-200 group-hover:bg-transparent group-hover:text-slate-950 transition-all flex items-center justify-center gap-2 font-medium">
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-amber-400 group-hover:text-slate-950 transition-colors" />
                  <span>Rub the Lamp (Login)</span>
                </>
              )}
            </div>
          </button>
        </form>

        {/* Footer / Hint */}
        <div className="mt-6 text-center text-[10px] text-indigo-300/50 border-t border-indigo-900/40 pt-4">
          Default Admin: <code className="text-amber-300/80">test_user</code> / <code className="text-amber-300/80">test_user</code>
        </div>
      </div>
    </div>
  );
}