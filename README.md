# Urban Pulse 2.0

## Intelligent Traffic Risk Management System

A comprehensive, end-to-end intelligent pipeline that transforms real-time API data into adaptive, explainable, and continuously improving traffic control decisions.

---

## 🌟 Key Features

### 4 Novel Contributions

1. **CARS (Context Aware Adaptive Risk Scoring)**
   - Dynamic risk scoring with context-dependent weights
   - Adapts to peak hours, weather, and location factors

2. **Priority Aware Risk Propagation**
   - Graph-based spatial modeling using H3 hexagons
   - Risk spreads between neighboring zones realistically

3. **Action Impact Feedback Loop**
   - Closed-loop learning: Predict → Act → Measure → Learn
   - Continuously improves decision effectiveness

4. **Human-in-the-Loop Explainability**
   - SHAP-based explanations for every decision
   - Transparent, trustworthy AI recommendations

---

## 📊 System Architecture

```
APIs (NYC Collisions + Open-Meteo)
      ↓
Data Ingestion (Bronze Layer)
      ↓
Cleaning & H3 Mapping (Silver Layer)
      ↓
Analytics-Ready Data (Gold Layer)
      ↓
Feature Engineering
      ↓
Advanced ML Models (Tuned)
      ↓
NOVELTY 1 – Adaptive Risk Scoring (CARS)
      ↓
NOVELTY 2 – Risk Propagation
      ↓
Decision Engine
      ↓
Actions Generated
      ↓
NOVELTY 3 – Feedback Learning
      ↓
NOVELTY 4 – Explainability
      ↓
Urban Pulse Web Platform
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ (for web frontend)

### Backend Setup

```bash
# Clone/navigate to project
cd Urbanpulse2

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd web
npm install
```

---

## 🚀 Running the System

### Option 1: Run Full Pipeline

```bash
python main.py
```

This runs the complete pipeline:
- Fetches data from APIs
- Processes through Bronze → Silver → Gold layers
- Trains ML models
- Applies all 4 novelties
- Generates decisions and outputs

### Option 2: Run API Server

```bash
python -m api.app
```

API will be available at `http://localhost:5000`

### Option 3: Run Web Frontend

```bash
cd web
npm start
```

Frontend will be available at `http://localhost:3000`

---

## 📁 Project Structure

```
Urbanpulse2/
├── config/                 # Configuration settings
│   └── settings.py
├── data_acquisition/       # API clients
│   ├── nyc_collisions_api.py
│   ├── weather_api.py
│   └── data_fetcher.py
├── data_engineering/       # Data pipeline
│   ├── bronze_layer.py
│   ├── silver_layer.py
│   ├── gold_layer.py
│   ├── h3_processor.py
│   └── pipeline.py
├── features/               # Feature engineering
│   ├── temporal_features.py
│   ├── spatial_features.py
│   ├── environmental_features.py
│   └── feature_engineer.py
├── models/                 # ML models
│   ├── base_models.py
│   ├── lstm_model.py
│   ├── context_experts.py
│   ├── online_learner.py
│   └── model_trainer.py
├── novelties/              # 4 Novel contributions
│   ├── cars.py             # Novelty 1
│   ├── risk_propagation.py # Novelty 2
│   ├── feedback_loop.py    # Novelty 3
│   └── explainability.py   # Novelty 4
├── decision_engine/        # Decision making
│   ├── signal_controller.py
│   ├── emergency_router.py
│   ├── alert_generator.py
│   └── decision_engine.py
├── api/                    # Flask REST API
│   └── app.py
├── web/                    # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── RiskMap.js
│   │   │   ├── IncidentMonitor.js
│   │   │   ├── SignalControl.js
│   │   │   ├── EmergencyRouting.js
│   │   │   ├── Alerts.js
│   │   │   └── Explainability.js
│   │   └── App.js
│   └── package.json
├── data/                   # Data storage
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── outputs/                # Generated outputs
├── main.py                 # Main orchestrator
├── requirements.txt
└── README.md
```

---

## 🔌 APIs Used

### 1. NYC Motor Vehicle Collisions API
- **Provider:** NYC Open Data
- **Endpoint:** `https://data.cityofnewyork.us/resource/h9gi-nx95.json`
- **Data:** Real-time accident/collision records

### 2. Open-Meteo Weather API
- **Provider:** Open-Meteo (Free, No Auth)
- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Data:** Hourly weather forecasts

---

## 📈 ML Models

1. **Linear Regression** - Baseline model
2. **Gradient Boosting** - Primary model with hyperparameter tuning
3. **LSTM** - Temporal sequence modeling with attention
4. **Context Expert Ensemble** - Specialized models for different contexts

---

## 🖥️ Web Platform Modules

- **Dashboard** - Overview with key metrics and trends
- **Live Risk Map** - Interactive H3 hexagon visualization
- **Incident Monitor** - Real-time incident tracking
- **Signal Control** - Traffic signal tuning recommendations
- **Emergency Routing** - Optimal route planning
- **Alerts** - Relief center and emergency alerts
- **Explainability** - AI decision explanations

---

## 📤 Output Files

The system generates:
- `signal_tuning_plan.csv` - Traffic signal adjustments
- `priority_routing_directive.csv` - Emergency vehicle routes
- `relief_center_alerts.csv` - Weather and risk alerts
- `explanations.json` - Decision explanations
- `feedback_data.json` - Feedback loop data

---

## 🎯 Report Text

> "The Urban Pulse system relies on two primary real-time data sources. Traffic incident data is collected from the NYC Motor Vehicle Collisions API provided by NYC Open Data, which supplies geotagged records of accidents, injuries, and fatalities. Environmental context is obtained from the Open-Meteo Weather API, which provides hourly weather attributes such as precipitation, wind speed, and temperature. These two APIs together enable a continuously updating multimodal dataset that supports real-time micro-zone traffic risk forecasting and adaptive decision making."

---

## 📝 License

MIT License

---

## 👥 Contributors

Urban Pulse 2.0 Development Team

---

## 🔮 Future Enhancements

- Real-time streaming with Apache Kafka
- Deep reinforcement learning for signal optimization
- Mobile application for field operators
- Integration with traffic camera feeds
- Predictive maintenance for traffic infrastructure
