import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Clock, MapPin, TrendingUp } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function IncidentMonitor() {
  const [zones, setZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('risk');
  const [timeRange, setTimeRange] = useState('24h');

  useEffect(() => {
    fetchIncidentData();
  }, [timeRange, sortBy, fetchIncidentData]);

  const fetchIncidentData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/risk/zones`);
      const data = await response.json();
      setZones(data.zones || []);
    } catch (error) {
      console.error('Failed to fetch incident data:', error);
      // Demo data
      setZones(generateDemoIncidents());
    }
    setLoading(false);
  }, [API_BASE]);

  const generateDemoIncidents = () => {
    return Array.from({ length: 30 }, (_, i) => ({
      h3_index: `zone_${i}`,
      lat: 40.7128 + (Math.random() - 0.5) * 0.1,
      lon: -74.0060 + (Math.random() - 0.5) * 0.1,
      risk_score: Math.random(),
      incident_count: Math.floor(Math.random() * 50),
      risk_category: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
      last_incident: new Date(Date.now() - Math.random() * 86400000).toISOString()
    }));
  };

  const sortedZones = [...zones].sort((a, b) => {
    if (sortBy === 'risk') return b.risk_score - a.risk_score;
    if (sortBy === 'incidents') return b.incident_count - a.incident_count;
    return 0;
  });

  const hourlyData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    incidents: Math.floor(Math.random() * 20) + 5
  }));

  const getRiskBadge = (category) => {
    const colors = {
      critical: 'bg-red-500/20 text-red-400 border-red-500/30',
      high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
      medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      low: 'bg-green-500/20 text-green-400 border-green-500/30'
    };
    return colors[category] || colors.low;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Incident Monitor</h1>
        <div className="flex items-center space-x-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          >
            <option value="risk">Sort by Risk</option>
            <option value="incidents">Sort by Incidents</option>
          </select>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-8 h-8 text-red-400" />
            <div>
              <p className="text-2xl font-bold">{zones.reduce((a, b) => a + b.incident_count, 0)}</p>
              <p className="text-sm text-slate-400">Total Incidents</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <MapPin className="w-8 h-8 text-orange-400" />
            <div>
              <p className="text-2xl font-bold">{zones.filter(z => z.risk_category === 'critical').length}</p>
              <p className="text-sm text-slate-400">Critical Zones</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-8 h-8 text-yellow-400" />
            <div>
              <p className="text-2xl font-bold">{(zones.reduce((a, b) => a + b.risk_score, 0) / zones.length * 100 || 0).toFixed(0)}%</p>
              <p className="text-sm text-slate-400">Avg Risk Score</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <Clock className="w-8 h-8 text-blue-400" />
            <div>
              <p className="text-2xl font-bold">{zones.length}</p>
              <p className="text-sm text-slate-400">Monitored Zones</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hourly Chart */}
        <div className="lg:col-span-2 bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4">Incidents by Hour</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="incidents" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="text-lg font-semibold mb-4">Risk Distribution</h3>
          <div className="space-y-3">
            {['critical', 'high', 'medium', 'low'].map((category) => {
              const count = zones.filter(z => z.risk_category === category).length;
              const percentage = (count / zones.length * 100) || 0;
              return (
                <div key={category}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize">{category}</span>
                    <span>{count} zones ({percentage.toFixed(0)}%)</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        category === 'critical' ? 'bg-red-500' :
                        category === 'high' ? 'bg-orange-500' :
                        category === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Incident Table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-4 border-b border-slate-700">
          <h3 className="text-lg font-semibold">Zone Incident Details</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Zone ID</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Risk Score</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Category</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Incidents</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Location</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sortedZones.slice(0, 15).map((zone) => (
                <tr key={zone.h3_index} className="border-t border-slate-700 hover:bg-slate-700/30">
                  <td className="p-4 text-sm font-mono">{zone.h3_index.substring(0, 12)}...</td>
                  <td className="p-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            zone.risk_score > 0.8 ? 'bg-red-500' :
                            zone.risk_score > 0.6 ? 'bg-orange-500' :
                            zone.risk_score > 0.3 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${zone.risk_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm">{(zone.risk_score * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded-full text-xs border ${getRiskBadge(zone.risk_category)}`}>
                      {zone.risk_category}
                    </span>
                  </td>
                  <td className="p-4 text-sm">{zone.incident_count}</td>
                  <td className="p-4 text-sm text-slate-400">
                    {zone.lat?.toFixed(4)}, {zone.lon?.toFixed(4)}
                  </td>
                  <td className="p-4">
                    <button className="text-blue-400 hover:text-blue-300 text-sm">View Details</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default IncidentMonitor;
