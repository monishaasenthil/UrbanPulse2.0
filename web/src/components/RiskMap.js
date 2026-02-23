import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { RefreshCw, Filter, Layers } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function MapController({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.setView(center, 12);
    }
  }, [center, map]);
  return null;
}

function RiskMap() {
  const [zones, setZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedZone, setSelectedZone] = useState(null);
  const [filter, setFilter] = useState('all');
  const [mapCenter] = useState([40.7128, -74.0060]);

  useEffect(() => {
    fetchZones();
  }, []);

  const fetchZones = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/risk/zones`);
      const data = await response.json();
      setZones(data.zones || []);
    } catch (error) {
      console.error('Failed to fetch zones:', error);
      // Generate demo data
      setZones(generateDemoZones());
    }
    setLoading(false);
  };

  const generateDemoZones = () => {
    const zones = [];
    for (let i = 0; i < 50; i++) {
      zones.push({
        h3_index: `demo_zone_${i}`,
        lat: 40.7128 + (Math.random() - 0.5) * 0.1,
        lon: -74.0060 + (Math.random() - 0.5) * 0.1,
        risk_score: Math.random(),
        propagated_risk: Math.random(),
        incident_count: Math.floor(Math.random() * 50),
        risk_category: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)]
      });
    }
    return zones;
  };

  const getRiskColor = (risk) => {
    if (risk > 0.8) return '#ef4444';
    if (risk > 0.6) return '#f97316';
    if (risk > 0.3) return '#eab308';
    return '#22c55e';
  };

  const filteredZones = zones.filter(zone => {
    if (filter === 'all') return true;
    return zone.risk_category === filter;
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Live Risk Map</h1>
        <div className="flex items-center space-x-3">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          >
            <option value="all">All Zones</option>
            <option value="critical">Critical Only</option>
            <option value="high">High Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="low">Low Risk</option>
          </select>
          <button
            onClick={fetchZones}
            disabled={loading}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Map */}
        <div className="lg:col-span-3 bg-slate-800 rounded-xl overflow-hidden border border-slate-700" style={{ height: '600px' }}>
          <MapContainer
            center={mapCenter}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
            <MapController center={mapCenter} />
            
            {filteredZones.map((zone) => (
              <CircleMarker
                key={zone.h3_index}
                center={[zone.lat, zone.lon]}
                radius={8 + zone.risk_score * 10}
                fillColor={getRiskColor(zone.propagated_risk || zone.risk_score)}
                color={getRiskColor(zone.propagated_risk || zone.risk_score)}
                weight={2}
                opacity={0.8}
                fillOpacity={0.5}
                eventHandlers={{
                  click: () => setSelectedZone(zone)
                }}
              >
                <Popup>
                  <div className="text-slate-900 p-2">
                    <p className="font-bold">Zone: {zone.h3_index.substring(0, 12)}...</p>
                    <p>Risk Score: {(zone.risk_score * 100).toFixed(1)}%</p>
                    <p>Incidents: {zone.incident_count}</p>
                    <p>Category: <span className="font-semibold capitalize">{zone.risk_category}</span></p>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Legend */}
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
            <h3 className="font-semibold mb-3 flex items-center">
              <Layers className="w-4 h-4 mr-2" />
              Risk Legend
            </h3>
            <div className="space-y-2">
              {[
                { label: 'Critical (>80%)', color: '#ef4444' },
                { label: 'High (60-80%)', color: '#f97316' },
                { label: 'Medium (30-60%)', color: '#eab308' },
                { label: 'Low (<30%)', color: '#22c55e' },
              ].map((item) => (
                <div key={item.label} className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: item.color }}></div>
                  <span className="text-sm text-slate-300">{item.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
            <h3 className="font-semibold mb-3">Zone Statistics</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Total Zones</span>
                <span>{zones.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Displayed</span>
                <span>{filteredZones.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Avg Risk</span>
                <span>{(zones.reduce((a, b) => a + b.risk_score, 0) / zones.length * 100 || 0).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Selected Zone */}
          {selectedZone && (
            <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
              <h3 className="font-semibold mb-3">Selected Zone</h3>
              <div className="space-y-2 text-sm">
                <p className="text-slate-400 break-all">{selectedZone.h3_index}</p>
                <div className="flex justify-between">
                  <span className="text-slate-400">Risk Score</span>
                  <span className="font-bold" style={{ color: getRiskColor(selectedZone.risk_score) }}>
                    {(selectedZone.risk_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Incidents</span>
                  <span>{selectedZone.incident_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Category</span>
                  <span className="capitalize">{selectedZone.risk_category}</span>
                </div>
                <button className="w-full mt-3 bg-blue-600 hover:bg-blue-700 py-2 rounded-lg text-sm">
                  View Details
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RiskMap;
