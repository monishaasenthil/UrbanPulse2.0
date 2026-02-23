import React, { useState, useEffect } from 'react';
import { Bell, AlertTriangle, CheckCircle, Clock, X } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetchAlerts();
  }, [filter]);

  const fetchAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE}/alerts`);
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
      setAlerts(generateDemoAlerts());
    }
  };

  const generateDemoAlerts = () => {
    const types = ['high_risk', 'elevated_risk', 'heavy_rain', 'incident_cluster', 'hospital_proximity'];
    const levels = ['CRITICAL', 'SEVERE', 'WARNING', 'INFO'];
    
    return Array.from({ length: 15 }, (_, i) => ({
      alert_id: `alert_${i}`,
      rule: types[i % types.length],
      level: levels[i % levels.length],
      level_value: 4 - (i % 4),
      message: `Alert message for ${types[i % types.length].replace('_', ' ')}`,
      action: 'Take appropriate action based on alert type',
      zone_id: `zone_${i}`,
      risk_score: Math.random(),
      timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      status: i < 5 ? 'active' : 'resolved',
      acknowledged: i > 3
    }));
  };

  const handleAcknowledge = async (alertId) => {
    try {
      await fetch(`${API_BASE}/alerts/${alertId}/acknowledge`, { method: 'POST' });
      setAlerts(alerts.map(a => 
        a.alert_id === alertId ? { ...a, acknowledged: true } : a
      ));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const getLevelColor = (level) => {
    const colors = {
      CRITICAL: 'bg-red-500/20 text-red-400 border-red-500/30',
      SEVERE: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      WARNING: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      INFO: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    };
    return colors[level] || colors.INFO;
  };

  const getLevelIcon = (level) => {
    if (level === 'CRITICAL' || level === 'SEVERE') {
      return <AlertTriangle className="w-5 h-5" />;
    }
    return <Bell className="w-5 h-5" />;
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true;
    if (filter === 'active') return alert.status === 'active';
    if (filter === 'acknowledged') return alert.acknowledged;
    if (filter === 'unacknowledged') return !alert.acknowledged && alert.status === 'active';
    return alert.level === filter;
  });

  const alertStats = {
    total: alerts.length,
    active: alerts.filter(a => a.status === 'active').length,
    critical: alerts.filter(a => a.level === 'CRITICAL').length,
    unacknowledged: alerts.filter(a => !a.acknowledged && a.status === 'active').length
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Alert Center</h1>
        <div className="flex items-center space-x-3">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          >
            <option value="all">All Alerts</option>
            <option value="active">Active Only</option>
            <option value="unacknowledged">Unacknowledged</option>
            <option value="CRITICAL">Critical</option>
            <option value="SEVERE">Severe</option>
            <option value="WARNING">Warning</option>
          </select>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Total Alerts</p>
              <p className="text-2xl font-bold">{alertStats.total}</p>
            </div>
            <Bell className="w-8 h-8 text-blue-400" />
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Active</p>
              <p className="text-2xl font-bold">{alertStats.active}</p>
            </div>
            <Clock className="w-8 h-8 text-yellow-400" />
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Critical</p>
              <p className="text-2xl font-bold text-red-400">{alertStats.critical}</p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-400" />
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Unacknowledged</p>
              <p className="text-2xl font-bold text-orange-400">{alertStats.unacknowledged}</p>
            </div>
            <X className="w-8 h-8 text-orange-400" />
          </div>
        </div>
      </div>

      {/* Alert List */}
      <div className="space-y-3">
        {filteredAlerts.map((alert) => (
          <div
            key={alert.alert_id}
            className={`bg-slate-800 rounded-xl p-5 border ${
              alert.status === 'active' && !alert.acknowledged
                ? 'border-l-4 border-l-red-500 border-slate-700'
                : 'border-slate-700'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4">
                <div className={`p-2 rounded-lg ${getLevelColor(alert.level)}`}>
                  {getLevelIcon(alert.level)}
                </div>
                <div>
                  <div className="flex items-center space-x-3 mb-1">
                    <span className={`px-2 py-0.5 rounded text-xs border ${getLevelColor(alert.level)}`}>
                      {alert.level}
                    </span>
                    <span className="text-sm text-slate-400">{alert.rule?.replace(/_/g, ' ')}</span>
                    {alert.acknowledged && (
                      <span className="flex items-center text-xs text-green-400">
                        <CheckCircle className="w-3 h-3 mr-1" /> Acknowledged
                      </span>
                    )}
                  </div>
                  <p className="font-medium">{alert.message}</p>
                  <p className="text-sm text-slate-400 mt-1">{alert.action}</p>
                  <div className="flex items-center space-x-4 mt-2 text-xs text-slate-500">
                    <span>Zone: {alert.zone_id}</span>
                    <span>Risk: {((alert.risk_score || 0) * 100).toFixed(0)}%</span>
                    <span>{new Date(alert.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                {!alert.acknowledged && alert.status === 'active' && (
                  <button
                    onClick={() => handleAcknowledge(alert.alert_id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                  >
                    Acknowledge
                  </button>
                )}
                <button className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-sm">
                  Details
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredAlerts.length === 0 && (
        <div className="text-center py-12 text-slate-400">
          <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No alerts match your filter criteria</p>
        </div>
      )}
    </div>
  );
}

export default Alerts;
