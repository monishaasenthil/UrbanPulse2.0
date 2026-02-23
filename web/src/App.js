import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, Map, Navigation, 
  Settings, Activity, Brain, RefreshCw, Bell
} from 'lucide-react';
import Dashboard from './components/Dashboard';
import RiskMap from './components/RiskMap';
import IncidentMonitor from './components/IncidentMonitor';
import SignalControl from './components/SignalControl';
import EmergencyRouting from './components/EmergencyRouting';
import Explainability from './components/Explainability';
import Alerts from './components/Alerts';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function Sidebar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/map', icon: Map, label: 'Risk Map' },
    { path: '/incidents', icon: Activity, label: 'Incidents' },
    { path: '/signals', icon: Settings, label: 'Signal Control' },
    { path: '/routing', icon: Navigation, label: 'Emergency Routing' },
    { path: '/alerts', icon: Bell, label: 'Alerts' },
    { path: '/explain', icon: Brain, label: 'Explainability' },
  ];

  return (
    <div className="w-64 bg-slate-900 h-screen fixed left-0 top-0 border-r border-slate-700">
      <div className="p-6">
        <h1 className="text-2xl font-bold text-blue-400">Urban Pulse</h1>
        <p className="text-slate-400 text-sm">v2.0 - Intelligent Traffic</p>
      </div>
      
      <nav className="mt-6">
        {navItems.map(({ path, icon: Icon, label }) => (
          <Link
            key={path}
            to={path}
            className={`flex items-center px-6 py-3 text-sm transition-colors ${
              location.pathname === path
                ? 'bg-blue-600 text-white border-r-4 border-blue-400'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Icon className="w-5 h-5 mr-3" />
            {label}
          </Link>
        ))}
      </nav>
      
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-700">
        <div className="text-xs text-slate-500">
          <p>4 Novelties Integrated</p>
          <p className="mt-1">• CARS • Risk Propagation</p>
          <p>• Feedback Loop • Explainability</p>
        </div>
      </div>
    </div>
  );
}

function Header({ onRefresh, isLoading }) {
  const [alerts, setAlerts] = useState(0);

  useEffect(() => {
    fetch(`${API_BASE}/alerts`)
      .then(res => res.json())
      .then(data => setAlerts(data.total || 0))
      .catch(() => {});
  }, []);

  return (
    <header className="h-16 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-6 fixed top-0 left-64 right-0 z-10">
      <div className="flex items-center space-x-4">
        <h2 className="text-lg font-semibold">Traffic Risk Management System</h2>
      </div>
      
      <div className="flex items-center space-x-4">
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>{isLoading ? 'Refreshing...' : 'Refresh Data'}</span>
        </button>
        
        <div className="relative">
          <Bell className="w-6 h-6 text-slate-400 hover:text-white cursor-pointer" />
          {alerts > 0 && (
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center">
              {alerts}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

function App() {
  const [isLoading, setIsLoading] = useState(false);

  const handleRefresh = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/data/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days_back: 7 })
      });
      const data = await response.json();
      if (data.status === 'success') {
        // Refresh successful
      }
    } catch (error) {
      console.error('Refresh failed:', error);
    }
    setIsLoading(false);
  };

  return (
    <Router>
      <div className="min-h-screen bg-slate-900">
        <Sidebar />
        <Header onRefresh={handleRefresh} isLoading={isLoading} />
        
        <main className="ml-64 pt-16 p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/map" element={<RiskMap />} />
            <Route path="/incidents" element={<IncidentMonitor />} />
            <Route path="/signals" element={<SignalControl />} />
            <Route path="/routing" element={<EmergencyRouting />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/explain" element={<Explainability />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
