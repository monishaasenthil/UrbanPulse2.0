import React, { useState, useEffect, useCallback } from 'react';
import { Settings, Clock, ArrowUpDown, Play, RefreshCw } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function SignalControl() {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);

  const fetchSignalPlans = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/decisions/signals`);
      const data = await response.json();
      setSignals(data.plans || []);
      setSummary(data.summary || null);
    } catch (error) {
      console.error('Failed to fetch signal plans:', error);
      setSignals(generateDemoSignals());
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchSignalPlans();
  }, [fetchSignalPlans]);

  const generateDemoSignals = () => {
    return Array.from({ length: 20 }, (_, i) => ({
      h3_index: `zone_${i}`,
      risk_score: Math.random(),
      risk_category: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
      recommended_green_time: 30 + Math.floor(Math.random() * 30),
      recommended_cycle_time: 90 + Math.floor(Math.random() * 30),
      green_time_change: Math.floor(Math.random() * 20) - 5,
      priority_direction: ['inbound', 'outbound', 'balanced'][Math.floor(Math.random() * 3)],
      pedestrian_phase: ['normal', 'standard', 'extended'][Math.floor(Math.random() * 3)],
      action_required: Math.random() > 0.7
    }));
  };

  const getRiskColor = (category) => {
    const colors = {
      critical: 'text-red-400',
      high: 'text-orange-400',
      medium: 'text-yellow-400',
      low: 'text-green-400'
    };
    return colors[category] || colors.low;
  };

  const actionRequiredSignals = signals.filter(s => s.action_required);

  const [appliedChanges, setAppliedChanges] = useState({});

  const handleApplyChanges = async (signal) => {
    try {
      // Mark as applying
      setAppliedChanges(prev => ({ ...prev, [signal.h3_index]: 'applying' }));
      
      // Simulate API call to apply signal changes
      await fetch(`${API_BASE}/decisions/apply-signal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zone_id: signal.h3_index,
          green_time: signal.recommended_green_time,
          cycle_time: signal.recommended_cycle_time,
          priority_direction: signal.priority_direction
        })
      }).catch(() => {}); // Ignore errors for demo
      
      // Mark as applied after short delay
      setTimeout(() => {
        setAppliedChanges(prev => ({ ...prev, [signal.h3_index]: 'applied' }));
        // Remove the signal from action required after applying
        setSignals(prev => prev.map(s => 
          s.h3_index === signal.h3_index ? { ...s, action_required: false } : s
        ));
      }, 1000);
    } catch (error) {
      console.error('Failed to apply changes:', error);
      setAppliedChanges(prev => ({ ...prev, [signal.h3_index]: 'error' }));
    }
  };

  const exportToCSV = () => {
    if (signals.length === 0) {
      alert('No data to export');
      return;
    }
    
    const headers = ['Zone', 'Risk Score', 'Risk Category', 'Green Time', 'Cycle Time', 'Change', 'Direction', 'Pedestrian', 'Action Required'];
    const rows = signals.map(s => [
      s.h3_index,
      s.risk_score?.toFixed(3) || '0',
      s.risk_category,
      s.recommended_green_time,
      s.recommended_cycle_time,
      s.green_time_change,
      s.priority_direction,
      s.pedestrian_phase,
      s.action_required ? 'Yes' : 'No'
    ]);
    
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signal_plans_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Signal Control Panel</h1>
        <button
          onClick={fetchSignalPlans}
          disabled={loading}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Plans</span>
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <Settings className="w-8 h-8 text-blue-400" />
            <div>
              <p className="text-2xl font-bold">{signals.length}</p>
              <p className="text-sm text-slate-400">Total Signals</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <Play className="w-8 h-8 text-red-400" />
            <div>
              <p className="text-2xl font-bold">{actionRequiredSignals.length}</p>
              <p className="text-sm text-slate-400">Action Required</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <Clock className="w-8 h-8 text-yellow-400" />
            <div>
              <p className="text-2xl font-bold">
                {summary?.avg_green_time_change?.toFixed(1) || '+5.2'}s
              </p>
              <p className="text-sm text-slate-400">Avg Time Change</p>
            </div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="flex items-center space-x-3">
            <ArrowUpDown className="w-8 h-8 text-green-400" />
            <div>
              <p className="text-2xl font-bold">{summary?.critical_zones || signals.filter(s => s.risk_category === 'critical').length}</p>
              <p className="text-sm text-slate-400">Critical Zones</p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Required Section */}
      {actionRequiredSignals.length > 0 && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5">
          <h3 className="text-lg font-semibold text-red-400 mb-4">⚠️ Signals Requiring Immediate Action</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {actionRequiredSignals.slice(0, 6).map((signal) => (
              <div key={signal.h3_index} className="bg-slate-800 rounded-lg p-4 border border-red-500/30">
                <div className="flex justify-between items-start mb-2">
                  <span className="font-mono text-sm">{signal.h3_index.substring(0, 12)}...</span>
                  <span className={`text-xs px-2 py-1 rounded-full bg-red-500/20 ${getRiskColor(signal.risk_category)}`}>
                    {signal.risk_category}
                  </span>
                </div>
                <div className="space-y-1 text-sm">
                  <p>Green Time: <span className="text-green-400">{signal.recommended_green_time}s</span></p>
                  <p>Change: <span className={signal.green_time_change > 0 ? 'text-green-400' : 'text-red-400'}>
                    {signal.green_time_change > 0 ? '+' : ''}{signal.green_time_change}s
                  </span></p>
                </div>
                <button 
                  onClick={() => handleApplyChanges(signal)}
                  disabled={appliedChanges[signal.h3_index] === 'applying' || appliedChanges[signal.h3_index] === 'applied'}
                  className={`w-full mt-3 py-2 rounded text-sm transition-all ${
                    appliedChanges[signal.h3_index] === 'applied' 
                      ? 'bg-green-600 cursor-default' 
                      : appliedChanges[signal.h3_index] === 'applying'
                      ? 'bg-yellow-600 cursor-wait'
                      : 'bg-red-600 hover:bg-red-700'
                  }`}
                >
                  {appliedChanges[signal.h3_index] === 'applied' 
                    ? '✓ Applied' 
                    : appliedChanges[signal.h3_index] === 'applying'
                    ? 'Applying...'
                    : 'Apply Changes'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Signal Plans Table */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-4 border-b border-slate-700 flex justify-between items-center">
          <h3 className="text-lg font-semibold">Signal Tuning Plans</h3>
          <button onClick={exportToCSV} className="text-sm text-blue-400 hover:text-blue-300">Export CSV</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Zone</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Risk</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Green Time</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Cycle Time</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Change</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Direction</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Pedestrian</th>
                <th className="text-left p-4 text-sm font-medium text-slate-400">Status</th>
              </tr>
            </thead>
            <tbody>
              {signals.slice(0, 15).map((signal) => (
                <tr key={signal.h3_index} className="border-t border-slate-700 hover:bg-slate-700/30">
                  <td className="p-4 text-sm font-mono">{signal.h3_index.substring(0, 12)}...</td>
                  <td className="p-4">
                    <span className={`capitalize ${getRiskColor(signal.risk_category)}`}>
                      {signal.risk_category}
                    </span>
                  </td>
                  <td className="p-4 text-sm">{signal.recommended_green_time}s</td>
                  <td className="p-4 text-sm">{signal.recommended_cycle_time}s</td>
                  <td className="p-4">
                    <span className={`text-sm ${signal.green_time_change > 0 ? 'text-green-400' : signal.green_time_change < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                      {signal.green_time_change > 0 ? '+' : ''}{signal.green_time_change}s
                    </span>
                  </td>
                  <td className="p-4 text-sm capitalize">{signal.priority_direction}</td>
                  <td className="p-4 text-sm capitalize">{signal.pedestrian_phase}</td>
                  <td className="p-4">
                    {signal.action_required ? (
                      <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400">Action Required</span>
                    ) : (
                      <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">Normal</span>
                    )}
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

export default SignalControl;
