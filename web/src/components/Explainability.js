import React, { useState } from 'react';
import { Brain, Search, BarChart3, Info, ChevronDown, ChevronUp } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function Explainability() {
  const [decisionType, setDecisionType] = useState('signal_change');
  const [riskScore, setRiskScore] = useState(0.75);
  const [actionTaken, setActionTaken] = useState('Extend green light duration by 15 seconds');
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedSection, setExpandedSection] = useState('breakdown');

  const handleExplain = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/explain/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision_type: decisionType,
          risk_score: riskScore,
          action_taken: actionTaken,
          features: {
            hour: new Date().getHours(),
            is_raining: false,
            incident_count: 25,
            severity: 15,
            historical_risk: 0.6,
            temperature: 20,
            wind_speed: 15
          }
        })
      });
      const data = await response.json();
      setExplanation(data);
    } catch (error) {
      console.error('Failed to get explanation:', error);
      // Demo explanation
      setExplanation(generateDemoExplanation());
    }
    setLoading(false);
  };

  const generateDemoExplanation = () => ({
    decision_type: decisionType,
    risk_score: riskScore,
    action_taken: actionTaken,
    confidence: 0.85,
    reasoning: [
      `HIGH risk level (${(riskScore * 100).toFixed(0)}%) warranted preventive action`,
      'Primary risk driver: incident_count (contribution: 0.245)',
      'Signal timing adjustment recommended to reduce congestion'
    ],
    human_summary: `The system detected a high risk level (${(riskScore * 100).toFixed(0)}%) primarily due to incident count and historical risk. In response, the following action was taken: ${actionTaken}. This decision was made with 85% confidence.`,
    prediction_explanation: {
      feature_contributions: {
        incident_count: { value: 25, shap_value: 0.245, impact: 'positive', magnitude: 0.245 },
        historical_risk: { value: 0.6, shap_value: 0.18, impact: 'positive', magnitude: 0.18 },
        severity: { value: 15, shap_value: 0.12, impact: 'positive', magnitude: 0.12 },
        hour: { value: 14, shap_value: 0.08, impact: 'positive', magnitude: 0.08 },
        temperature: { value: 20, shap_value: -0.02, impact: 'negative', magnitude: 0.02 },
        wind_speed: { value: 15, shap_value: 0.03, impact: 'positive', magnitude: 0.03 },
        is_raining: { value: 0, shap_value: -0.05, impact: 'negative', magnitude: 0.05 }
      }
    }
  });

  const getContributionBar = (contribution) => {
    const maxWidth = 150;
    const width = Math.min(Math.abs(contribution.shap_value) * maxWidth * 2, maxWidth);
    const isPositive = contribution.shap_value > 0;
    
    return (
      <div className="flex items-center space-x-2">
        <div className="w-32 text-right">
          <span className={isPositive ? 'text-red-400' : 'text-green-400'}>
            {isPositive ? '+' : ''}{contribution.shap_value.toFixed(3)}
          </span>
        </div>
        <div className="flex-1 h-4 bg-slate-700 rounded relative overflow-hidden">
          <div
            className={`absolute h-full ${isPositive ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ 
              width: `${width}px`,
              left: isPositive ? '50%' : `calc(50% - ${width}px)`
            }}
          ></div>
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-500"></div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Explainability Dashboard</h1>
        <span className="text-sm text-slate-400">NOVELTY 4: Human-in-the-Loop AI</span>
      </div>

      {/* Input Section */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          Generate Decision Explanation
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Decision Type</label>
            <select
              value={decisionType}
              onChange={(e) => setDecisionType(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
            >
              <option value="signal_change">Signal Change</option>
              <option value="emergency_routing">Emergency Routing</option>
              <option value="alert">Alert Generation</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Risk Score</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={riskScore}
              onChange={(e) => setRiskScore(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-center text-sm mt-1">{(riskScore * 100).toFixed(0)}%</p>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-2">Action Taken</label>
            <input
              type="text"
              value={actionTaken}
              onChange={(e) => setActionTaken(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleExplain}
              disabled={loading}
              className="w-full bg-purple-600 hover:bg-purple-700 py-2 rounded-lg flex items-center justify-center space-x-2"
            >
              <Search className="w-4 h-4" />
              <span>{loading ? 'Analyzing...' : 'Explain Decision'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Explanation Result */}
      {explanation && (
        <div className="space-y-4">
          {/* Human Summary */}
          <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-6 border border-purple-500/30">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <Info className="w-5 h-5 mr-2 text-purple-400" />
              Human-Readable Summary
            </h3>
            <p className="text-slate-200 leading-relaxed">{explanation.human_summary}</p>
            <div className="mt-4 flex items-center space-x-4">
              <span className="text-sm text-slate-400">Confidence:</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-purple-500"
                    style={{ width: `${(explanation.confidence || 0.85) * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-bold text-purple-400">
                  {((explanation.confidence || 0.85) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Reasoning */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <button
              onClick={() => setExpandedSection(expandedSection === 'reasoning' ? '' : 'reasoning')}
              className="w-full p-4 flex items-center justify-between hover:bg-slate-700/50"
            >
              <h3 className="font-semibold">Decision Reasoning</h3>
              {expandedSection === 'reasoning' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            {expandedSection === 'reasoning' && (
              <div className="p-4 pt-0 space-y-2">
                {explanation.reasoning?.map((reason, idx) => (
                  <div key={idx} className="flex items-start space-x-3 p-3 bg-slate-700/30 rounded-lg">
                    <span className="text-purple-400 font-bold">{idx + 1}.</span>
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Feature Contributions */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <button
              onClick={() => setExpandedSection(expandedSection === 'breakdown' ? '' : 'breakdown')}
              className="w-full p-4 flex items-center justify-between hover:bg-slate-700/50"
            >
              <h3 className="font-semibold flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Feature Contributions (SHAP Values)
              </h3>
              {expandedSection === 'breakdown' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>
            {expandedSection === 'breakdown' && (
              <div className="p-4 pt-0">
                <div className="mb-4 flex items-center justify-center space-x-6 text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-red-500 rounded"></div>
                    <span>Increases Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-green-500 rounded"></div>
                    <span>Decreases Risk</span>
                  </div>
                </div>
                <div className="space-y-3">
                  {Object.entries(explanation.prediction_explanation?.feature_contributions || {})
                    .sort((a, b) => b[1].magnitude - a[1].magnitude)
                    .map(([feature, data]) => (
                      <div key={feature} className="flex items-center space-x-4">
                        <div className="w-32 text-sm text-slate-400 truncate">{feature}</div>
                        <div className="w-16 text-sm text-right">{typeof data.value === 'number' ? data.value.toFixed(2) : data.value}</div>
                        <div className="flex-1">
                          {getContributionBar(data)}
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>

          {/* Technical Details */}
          <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
            <h3 className="font-semibold mb-4">Technical Details</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-slate-700/50 p-3 rounded-lg">
                <p className="text-slate-400">Decision Type</p>
                <p className="font-medium capitalize">{explanation.decision_type?.replace('_', ' ')}</p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded-lg">
                <p className="text-slate-400">Risk Score</p>
                <p className="font-medium">{((explanation.risk_score || 0) * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded-lg">
                <p className="text-slate-400">Features Analyzed</p>
                <p className="font-medium">{Object.keys(explanation.prediction_explanation?.feature_contributions || {}).length}</p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded-lg">
                <p className="text-slate-400">Model Confidence</p>
                <p className="font-medium">{((explanation.confidence || 0.85) * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-xl p-5">
        <h3 className="font-semibold text-blue-400 mb-2">About Explainability (Novelty 4)</h3>
        <p className="text-sm text-slate-300">
          This module provides human-readable explanations for every decision made by Urban Pulse. 
          Using SHAP (SHapley Additive exPlanations) values, we break down how each feature contributes 
          to the final risk score and decision. This transparency enables operators to understand, 
          verify, and trust the AI system's recommendations.
        </p>
      </div>
    </div>
  );
}

export default Explainability;
