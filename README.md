# RideKaro — Intelligent Carpooling Optimisation Platform
## AI Lab Project — NUCES (FAST), Karachi

| Field | Detail |
|---|---|
| **Students** | Kabir Raj (23K-0702) · Hassnain Aziz (23K-0905) |
| **Course** | Artificial Intelligence Lab |
| **University** | NUCES / FAST, Karachi |
| **Tests** | 48 / 48 passing |
| **Agents** | 11 goal-based rational agents |
| **Algorithms** | 10 search + Alpha-Beta + GA + CSP |
| **Persistence** | SQLite database (users, rides, bookings, payments, reviews) |

---

## Quick Start (2 commands)

```bash
pip install -r requirements.txt
python run_gui.py
```
Browser opens at **http://127.0.0.1:5000**

The SQLite database (`database/carpooling.db`) is created and seeded automatically
on first run with 8 demo users (Karachi-based names, emails, phone numbers) and
5 demo vehicles — no manual setup required.

---

## Backend & Database Layer (New)

The platform now ships with a real persistence layer so it behaves like a
production carpooling app rather than a single-session simulation:

| Table | Purpose |
|---|---|
| `users` | Riders/drivers — name, email, phone, role, rating, verification status |
| `vehicles` | Driver vehicles — make, model, plate, capacity, fuel type |
| `rides` | Posted rides — origin, destination, departure time, seats, fare, status |
| `bookings` | Passenger seat reservations linked to rides, with pickup/dropoff stops |
| `reviews` | Post-ride ratings between driver and passenger (updates `users.rating`) |
| `payments` | Transaction records (cash / JazzCash / Easypaisa / card) with reference numbers |
| `notifications` | In-app notifications (ride posted, booking confirmed, new passenger, etc.) |
| `match_sessions` | Every AI pipeline run is logged — drivers, passengers, groups formed, CO₂ saved |
| `algo_runs` | Every algorithm-visualiser run is logged for analytics |

**REST API** (`/api/db/...`) exposes full CRUD for the above: posting rides,
booking seats, processing payments, leaving reviews, and reading live platform
analytics (`/api/db/analytics`).

A new **🗄️ Database** tab in the UI lets you browse/search users, rides, and
bookings, post new rides, book seats, and process payments — all backed by
real SQL queries, not in-memory mock data.

---

## What's Implemented

### 11 Intelligent Agents (Goal-Based Rational, PAMA Cycle)
| Agent | Role | Algorithm |
|---|---|---|
| OrchestratorAgent | Master controller | Pipeline coordination |
| EnvironmentAgent | City traffic & weather | Dynamic graph weights |
| TrafficIntelAgent | Congestion analysis | Graph statistics |
| PredictionAgent | Demand forecasting | Gaussian Naive Bayes |
| MatchingAgent | Candidate discovery | BFS + Greedy scoring |
| NegotiationAgent | Price negotiation | Minimax + Alpha-Beta |
| GlobalOptimiserAgent | Global grouping | Genetic Algorithm |
| RouteOptimiserAgent | Route planning | A* Heuristic Search |
| ConstraintAgent | Constraint validation | CSP + AC-3 + Backtracking |
| SafetyAgent | Safety audit | Rule-based reasoning |
| CostOptimiserAgent | Dynamic fares + CO₂ | Fuel cost modelling |

### 10 Search Algorithms (all animated step-by-step in the GUI)
A* · IDA* · BFS · DFS · Dijkstra · UCS · Greedy Best-First · Hill Climbing · Beam Search · Bidirectional BFS

### Adversarial AI
Alpha-Beta Pruning (Minimax) — game tree rendered as interactive SVG

### Machine Learning (scikit-learn)
- Random Forest + Gradient Boosting → Ensemble Classifier (87.5% accuracy)
- K-Means Clustering (k=5 city zones)
- Linear Regression (driver rating prediction, R²=0.62)
- PCA 2D projection
- Full EDA: correlation matrix, histograms, match-rate charts

### Bayesian Network
Gaussian Naive Bayes — demand level + ETA factor + surge pricing per zone × hour × weather

### Real-World Outputs
- Dynamic fares (PKR) with surge pricing and high-utilisation discounts
- CO₂ savings tracked per group vs individual travel
- Karachi-specific congestion hotspots (rush hours 7-9am, 5-8pm)
- Safety flags for low-rated users and long routes
- Game-theoretic negotiation outcomes per driver-passenger pair

---

## GUI Pages

| Page | Features |
|---|---|
| **Dashboard** | Add drivers/passengers · D3 interactive map · Run pipeline · Bayesian demand · Ride groups · Agent timeline |
| **Agents** | 11 agent cards with roles & live stats · Negotiation table · Safety audit · Fare table · Traffic intel |
| **Algorithms** | Animate any of 10 algorithms step-by-step · 10-algo benchmark table · Alpha-Beta SVG tree |
| **ML Analytics** | EDA charts · Correlation heatmap · RF confusion matrix · PCA scatter · Cluster map · Live predictors |
| **Results** | Full stats · A* vs BFS vs Greedy table · GA convergence · 4 matplotlib charts · Agent log |

---

## Project Structure

```
carpooling_ai/
├── run_gui.py                    ← ONE-CLICK LAUNCHER
├── main.py                       ← CLI entry point
├── requirements.txt
├── README.md
├── agents/
│   └── agent.py                  ← 11 agents + EventBus + OrchestratorAgent
├── algorithms/
│   ├── search.py                 ← A*, BFS, Greedy, Dijkstra
│   ├── search_extended.py        ← DFS, IDA*, UCS, Hill Climbing, Beam, BiDirBFS, Alpha-Beta
│   ├── genetic.py                ← Genetic Algorithm (OX1 crossover, elitism, tournament)
│   └── csp.py                    ← CSP + AC-3 + MRV + LCV + Backtracking
├── models/
│   ├── models.py                 ← Driver, Passenger, RideGroup, Location
│   ├── city_graph.py             ← Karachi road network (20 nodes, 30 edges)
│   └── bayesian.py               ← Gaussian Naive Bayes demand predictor
├── ml/
│   └── ml_engine.py              ← EDA, RandomForest, GradientBoosting, KMeans, PCA, Regression
├── data/
│   └── data_generator.py         ← Realistic Karachi scenario generator
├── utils/
│   └── reporter.py               ← Colour terminal dashboard
├── dashboard/
│   └── visualiser.py             ← 4 matplotlib charts
├── gui/
│   ├── app.py                    ← Flask server + 20 REST endpoints
│   └── templates/index.html      ← Complete 5-page GUI (D3, Canvas, pure JS — 1827 lines)
├── output/                       ← Pre-generated PNG charts
└── tests/
    └── test_system.py            ← 48 unit + integration tests
```

---

## CLI Usage

```bash
python main.py                          # default (8am, clear, 6 drivers, 15 passengers)
python main.py --hour 17 --weather rainy
python main.py --drivers 8 --passengers 20 --generations 100
python -m pytest tests/ -v              # run all 48 tests
```

---

## Dependencies

```
flask>=2.3.0        Web server + REST API
networkx>=3.0       City road graph
scikit-learn>=1.3.0 RF, GB, KMeans, PCA, LinearRegression
numpy>=1.24.0       Matrix operations
matplotlib>=3.7.0   Chart generation
colorama>=0.4.6     Terminal output
tabulate>=0.9.0     Terminal tables
```

All AI algorithms (A*, BFS, DFS, IDA*, Hill Climbing, Beam, BiDir BFS, Genetic Algorithm, CSP, Alpha-Beta) are implemented from scratch in pure Python. scikit-learn is used only for ML classification, clustering, and regression.
