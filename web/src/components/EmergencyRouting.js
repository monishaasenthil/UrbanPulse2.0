import React, { useState } from 'react';
import { Navigation, MapPin, AlertTriangle, Clock, Route, Search } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function EmergencyRouting() {
  const [incidentLocation, setIncidentLocation] = useState('');
  const [destinationType, setDestinationType] = useState('hospital');
  const [routeResult, setRouteResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recentRoutes, setRecentRoutes] = useState([]);

  const handleFindRoute = async () => {
    if (!incidentLocation) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/decisions/routing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: incidentLocation,
          destination_type: destinationType
        })
      });
      const data = await response.json();
      setRouteResult(data);
      setRecentRoutes(prev => [data, ...prev].slice(0, 5));
    } catch (error) {
      console.error('Failed to find route:', error);
      // Demo result
      setRouteResult({
        status: 'success',
        incident_location: incidentLocation,
        destination: 'hospital_zone_1',
        destination_name: 'NYC General Hospital',
        destination_type: destinationType,
        recommended_route: ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5'],
        estimated_zones: 5,
        risk_assessment: {
          avg_risk: 0.35,
          max_risk: 0.65,
          high_risk_zones: 1
        },
        alternatives: [
          { route_type: 'shortest', path_length: 4, avg_risk: 0.55 },
          { route_type: 'alternative_1', path_length: 6, avg_risk: 0.25 }
        ]
      });
    }
    setLoading(false);
  };

  const getRiskColor = (risk) => {
    if (risk > 0.7) return 'text-red-400';
    if (risk > 0.5) return 'text-orange-400';
    if (risk > 0.3) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Emergency Vehicle Routing</h1>
      </div>

      {/* Route Finder */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Navigation className="w-5 h-5 mr-2 text-blue-400" />
          Find Optimal Route
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Incident Location (H3 Zone)</label>
            <input
              type="text"
              value={incidentLocation}
              onChange={(e) => setIncidentLocation(e.target.value)}
              placeholder="Enter zone ID or coordinates"
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Destination Type</label>
            <select
              value={destinationType}
              onChange={(e) => setDestinationType(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
            >
              <option value="hospital">Hospital</option>
              <option value="fire_station">Fire Station</option>
              <option value="police_station">Police Station</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleFindRoute}
              disabled={loading || !incidentLocation}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 py-2 rounded-lg flex items-center justify-center space-x-2"
            >
              <Search className="w-4 h-4" />
              <span>{loading ? 'Finding Route...' : 'Find Route'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Route Result */}
      {routeResult && routeResult.status === 'success' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Route */}
          <div className="lg:col-span-2 bg-slate-800 rounded-xl p-6 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-green-400">✓ Recommended Route</h3>
              <span className="text-sm text-slate-400">Primary Safe Route</span>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-sm text-slate-400">From</p>
                <p className="font-mono">{routeResult.incident_location}</p>
              </div>
              <div className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-sm text-slate-400">To</p>
                <p className="font-semibold">{routeResult.destination_name}</p>
              </div>
            </div>

            {/* Route Visualization */}
            <div className="mb-6">
              <p className="text-sm text-slate-400 mb-3">Route Path ({routeResult.estimated_zones} zones)</p>
              <div className="flex items-center flex-wrap gap-2">
                {routeResult.recommended_route?.map((zone, idx) => (
                  <React.Fragment key={zone}>
                    <div className="bg-blue-600/20 border border-blue-500/30 px-3 py-1 rounded text-sm">
                      {zone}
                    </div>
                    {idx < routeResult.recommended_route.length - 1 && (
                      <span className="text-slate-500">→</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* Risk Assessment */}
            <div className="bg-slate-700/30 rounded-lg p-4">
              <h4 className="font-semibold mb-3">Risk Assessment</h4>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-slate-400">Average Risk</p>
                  <p className={`text-xl font-bold ${getRiskColor(routeResult.risk_assessment?.avg_risk)}`}>
                    {((routeResult.risk_assessment?.avg_risk || 0) * 100).toFixed(0)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-400">Max Risk</p>
                  <p className={`text-xl font-bold ${getRiskColor(routeResult.risk_assessment?.max_risk)}`}>
                    {((routeResult.risk_assessment?.max_risk || 0) * 100).toFixed(0)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-400">High Risk Zones</p>
                  <p className="text-xl font-bold text-orange-400">
                    {routeResult.risk_assessment?.high_risk_zones || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Alternatives */}
          <div className="space-y-4">
            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <h3 className="font-semibold mb-4">Alternative Routes</h3>
              <div className="space-y-3">
                {routeResult.alternatives?.map((alt, idx) => (
                  <div key={idx} className="bg-slate-700/50 rounded-lg p-3">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium capitalize">{alt.route_type?.replace('_', ' ')}</span>
                      <span className={`text-xs ${getRiskColor(alt.avg_risk)}`}>
                        {((alt.avg_risk || 0) * 100).toFixed(0)}% risk
                      </span>
                    </div>
                    <div className="flex justify-between text-sm text-slate-400">
                      <span>{alt.path_length} zones</span>
                      <button className="text-blue-400 hover:text-blue-300">Select</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
              <h3 className="font-semibold mb-4">Quick Actions</h3>
              <div className="space-y-2">
                <button className="w-full bg-green-600/20 hover:bg-green-600/30 border border-green-500/30 py-2 rounded-lg text-sm text-green-400">
                  Dispatch Emergency Vehicle
                </button>
                <button className="w-full bg-yellow-600/20 hover:bg-yellow-600/30 border border-yellow-500/30 py-2 rounded-lg text-sm text-yellow-400">
                  Clear Route Signals
                </button>
                <button className="w-full bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 py-2 rounded-lg text-sm text-blue-400">
                  Export Route
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Routes */}
      {recentRoutes.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h3 className="font-semibold mb-4">Recent Route Requests</h3>
          <div className="space-y-2">
            {recentRoutes.map((route, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                <div className="flex items-center space-x-4">
                  <Route className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="text-sm">{route.incident_location} → {route.destination_name}</p>
                    <p className="text-xs text-slate-400">{route.estimated_zones} zones</p>
                  </div>
                </div>
                <span className={`text-sm ${getRiskColor(route.risk_assessment?.avg_risk)}`}>
                  {((route.risk_assessment?.avg_risk || 0) * 100).toFixed(0)}% avg risk
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default EmergencyRouting;
