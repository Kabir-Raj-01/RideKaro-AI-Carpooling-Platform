"""
rag/knowledge_base.py
═══════════════════════════════════════════════════════════════════
Real-world Karachi Carpooling Knowledge Base for RAG system.
Contains structured knowledge documents across multiple domains:
  - Karachi traffic & routes
  - Safety guidelines (Pakistan context)
  - Pricing & fare calculation
  - Platform how-to / FAQ
  - AI/Algorithm explanations
  - Ride etiquette & rules
═══════════════════════════════════════════════════════════════════
"""

KNOWLEDGE_DOCUMENTS = [

    # ── KARACHI TRAFFIC & ROUTES ─────────────────────────────────────────

    {
        "id": "traffic_001",
        "category": "traffic",
        "title": "Karachi Morning Rush Hours",
        "content": """Karachi experiences peak morning traffic between 7:30 AM and 9:30 AM.
Major congestion points include: Shahrae Faisal (especially near PIDC chowk),
University Road near Gulshan-e-Iqbal, Abul Hasan Ispahani Road near NUCES/FAST University,
M.A. Jinnah Road near Saddar, and the Lyari Expressway merge at Keamari.
Drivers should depart 20-25 minutes earlier during morning rush.
The RideKaro system applies a 1.6x traffic factor during morning rush hours (7-10 AM).
Recommended pickup windows: 6:45-7:15 AM or after 9:45 AM to avoid peak congestion.""",
        "keywords": ["morning", "rush", "traffic", "7am", "8am", "9am", "congestion", "karachi", "commute"]
    },

    {
        "id": "traffic_002",
        "category": "traffic",
        "title": "Karachi Evening Rush Hours",
        "content": """Evening rush in Karachi runs from 5:00 PM to 8:30 PM with peak at 6-7 PM.
Worst affected routes: DHA Phase 5 to Clifton bridge, Shahra-e-Faisal southbound,
Korangi Road near industrial zone, Johar Chowrangi towards Gulshan.
The RideKaro system applies a 1.8x traffic multiplier during evening rush (5-8 PM).
Evening surge pricing applies: fares increase by 25-60% during peak.
Carpooling during evening rush saves an average of PKR 180-350 per person vs solo ride.
Recommended departure: 4:30-5:00 PM or after 8:30 PM for minimal traffic.""",
        "keywords": ["evening", "rush", "5pm", "6pm", "7pm", "traffic", "surge", "peak"]
    },

    {
        "id": "traffic_003",
        "category": "traffic",
        "title": "Best Routes in Karachi for Carpooling",
        "content": """Top carpooling corridors in Karachi by demand:
1. NUCES/FAST → Gulshan-e-Iqbal → Saddar: High student/professional demand, 12-18 km
2. DHA → Clifton → Downtown: Executive commuters, 8-15 km, avg fare PKR 120-200/person
3. Johar → North Karachi → Federal B Area: Northern corridor, 15-22 km
4. Korangi → Saddar → Gulshan: Industrial to commercial, 18-25 km
5. Airport → DHA → Clifton: Airport-linked commutes, 20-30 km
6. Orangi Town → Lyari → Saddar: Western corridor, 12-18 km
Carpooling reduces per-person cost by 55-70% on all these routes compared to solo rides.""",
        "keywords": ["route", "corridor", "nuces", "gulshan", "dha", "clifton", "johar", "saddar", "distance", "km"]
    },

    {
        "id": "traffic_004",
        "category": "traffic",
        "title": "Rainy Season Traffic in Karachi",
        "content": """Karachi monsoon season (July-September) severely impacts traffic.
During rain, travel times increase by 35-50% on all major routes.
Common flooding points: Lyari River underpasses, Nagan Chowrangi underpass,
Hassan Square, MA Jinnah Road near Teen Talwar.
RideKaro applies a 1.35x weather multiplier for rainy conditions.
Surge pricing activates at 1.25x during rain to compensate drivers.
Recommendations: Book rides 30 min earlier during rain. Drivers should check
Karachi Water & Sewerage Board alerts. Avoid low-lying areas like Keamari, Lyari during heavy rain.""",
        "keywords": ["rain", "monsoon", "flood", "weather", "rainy", "july", "august", "september", "traffic"]
    },

    # ── SAFETY GUIDELINES ────────────────────────────────────────────────

    {
        "id": "safety_001",
        "category": "safety",
        "title": "Safety Guidelines for Female Passengers in Karachi",
        "content": """RideKaro safety guidelines for female passengers:
1. Always verify driver rating is 4.0 or above before accepting a ride.
2. Share your live trip details with a trusted contact using the in-app share feature.
3. Preferred driver gender filtering is available in ride preferences.
4. Use the SOS emergency button in-app for immediate police/emergency contact.
5. Verify vehicle number plate matches what's shown in the app before boarding.
6. Night rides (10 PM - 5 AM) automatically trigger a safety check-in notification.
7. Rated 4.5+ female drivers are marked with a verified safety badge.
8. Emergency contacts: Rescue 1122 (Punjab) / Edhi Foundation 115 / Police 15.
The RideKaro SafetyAgent automatically flags groups with low-rated users or night-time routes.""",
        "keywords": ["female", "women", "safety", "night", "sos", "emergency", "verified", "secure", "gender"]
    },

    {
        "id": "safety_002",
        "category": "safety",
        "title": "Vehicle Safety and Driver Requirements",
        "content": """Driver requirements on RideKaro platform:
- Minimum driver rating: 3.5 stars (auto-flagged by SafetyAgent below this)
- Vehicle must have valid registration (pass valid inspection)
- Maximum vehicle age: 15 years recommended
- Seat belts must be available for all passengers
- No ride-sharing with more passengers than vehicle capacity allows
- CSP (Constraint Satisfaction Problem) solver enforces vehicle capacity automatically
- Drivers with less than 10 rides and rating below 4.0 shown as "New Driver" badge
- Background verification recommended for all drivers
Route safety: Routes longer than 40 km trigger an automatic advisory in the app.""",
        "keywords": ["vehicle", "driver", "requirement", "rating", "capacity", "seat", "registration", "safety"]
    },

    {
        "id": "safety_003",
        "category": "safety",
        "title": "Night Ride Safety Protocol",
        "content": """RideKaro Night Safety Protocol (10 PM to 5 AM):
1. Safety advisory displayed to all passengers booking night rides.
2. Automatic SOS contact sharing prompt before ride start.
3. Only drivers with rating 4.0+ can offer night rides.
4. Trip-sharing is strongly recommended — app generates a shareable link.
5. Designated safe pickup points: illuminated public spaces, restaurants, hospitals.
6. Driver receives safety reminders: maintain speed limits, well-lit routes preferred.
7. Passenger check-in notification sent 10 minutes into the ride.
8. Emergency contacts for Karachi: Police 15, Edhi 115, Chhipa 1020, Rescue 1122.
Night surge: 1.2x fare multiplier applies to compensate drivers for reduced availability.""",
        "keywords": ["night", "10pm", "late", "dark", "sos", "emergency", "protocol", "after hours", "safety"]
    },

    # ── PRICING & FARES ──────────────────────────────────────────────────

    {
        "id": "pricing_001",
        "category": "pricing",
        "title": "How Fares Are Calculated on RideKaro",
        "content": """RideKaro fare calculation uses a dynamic pricing model:
Base Fare = (Distance ÷ Fuel Efficiency) × Petrol Price (PKR 305/litre as of 2024)
Fare per Person = Base Fare × Surge Multiplier × (1 - Utilisation Discount) ÷ (Passengers + 1)

Surge multipliers:
- Low demand zones: 1.0x (no surge)
- Medium demand zones: 1.25x
- High demand zones: 1.6x
- Rainy weather: additional 1.35x
- Night rides (10 PM - 5 AM): 1.2x

Utilisation discount: If vehicle is 75%+ full, passengers get 10% discount.
Example: 15 km ride, 4-seat car, 3 passengers = ~PKR 85-120 per person vs PKR 350+ solo.
CO₂ savings: approximately 2.31 kg CO₂ saved per litre of petrol per shared passenger.""",
        "keywords": ["fare", "price", "cost", "pkr", "rupees", "surge", "discount", "calculate", "how much", "petrol"]
    },

    {
        "id": "pricing_002",
        "category": "pricing",
        "title": "Payment Methods on RideKaro",
        "content": """RideKaro supports multiple payment methods:
1. Cash (most common in Pakistan) - paid directly to driver at end of ride
2. JazzCash - mobile wallet, scan QR or send to driver's JazzCash number
3. Easypaisa - Telenor mobile wallet, widely accepted
4. Debit/Credit Card - processed through the in-app payment gateway
5. Bank Transfer - for pre-arranged corporate/university rides

Payment tips:
- Keep exact cash ready for smooth transactions
- JazzCash and Easypaisa transactions are instantly confirmed via SMS
- Payment reference numbers are stored in the Payments database for 12 months
- Refunds for cancelled rides are processed within 3-5 business days
- Platform service fee: currently 0% (in beta phase — free for all users)""",
        "keywords": ["payment", "jazzcash", "easypaisa", "cash", "card", "pay", "refund", "method", "how to pay"]
    },

    {
        "id": "pricing_003",
        "category": "pricing",
        "title": "Karachi Fuel Prices and Carpooling Savings",
        "content": """Pakistan petrol prices (2024): PKR 305/litre for regular petrol (RON-92).
Average car fuel efficiency in Karachi: 10-15 km/litre in traffic, 15-18 km/litre highway.
Average daily commute in Karachi: 15-25 km each way.

Solo commute cost example (20 km, 12 km/litre efficiency):
Daily fuel cost = (20 ÷ 12) × 305 = PKR 508/day = PKR 12,700/month

With RideKaro (3 passengers sharing):
Per-person cost = PKR 508 ÷ 4 = PKR 127/day = PKR 3,175/month
Monthly savings: PKR 9,525 per passenger (75% saving!)

Additional savings: Reduced vehicle wear (sharing extends car life by 3-4 years),
reduced parking costs (PKR 50-200/day at commercial areas),
and reduced CO₂ emissions (2.31 kg per litre saved).""",
        "keywords": ["fuel", "petrol", "savings", "cost", "monthly", "pkr", "rupees", "save", "expensive"]
    },

    # ── HOW-TO / FAQ ─────────────────────────────────────────────────────

    {
        "id": "howto_001",
        "category": "howto",
        "title": "How to Post a Ride as a Driver",
        "content": """How to post a ride on RideKaro (Driver Guide):
Step 1: Go to the Database tab → Click "Post Ride" button (green + button)
Step 2: Fill in your details:
  - Select your name from the Driver dropdown (must be registered as Driver/Both role)
  - Select your vehicle from the Vehicle dropdown
  - Set Origin: your starting point (e.g., NUCES, DHA_1)
  - Set Destination: where you're going (e.g., Saddar, Clifton)
  - Set departure date and time
  - Set number of available seats (1-5)
  - Set fare per seat in PKR (suggested: use the fare calculator)
  - Add optional notes (AC available, no smoking, female-only, etc.)
Step 3: Click "Post Ride" — your ride is now live and searchable by passengers.
Step 4: You'll receive a notification when a passenger books your ride.""",
        "keywords": ["post", "driver", "how to", "ride", "offer", "create", "add", "new ride", "guide"]
    },

    {
        "id": "howto_002",
        "category": "howto",
        "title": "How to Book a Ride as a Passenger",
        "content": """How to book a ride on RideKaro (Passenger Guide):
Step 1: Go to Database tab → Rides → search for your origin/destination
Step 2: Browse available rides — check departure time, seats, fare, driver rating
Step 3: Click "Details" on a ride to see full route, vehicle info, driver details
Step 4: Click "Book Ride" button or use the + Book Ride form
Step 5: Fill in:
  - Select your name (Passenger/Both role)
  - Select the ride you want to book
  - Enter your pickup stop (specific location name)
  - Enter your dropoff stop
  - Number of seats needed
  - Fare you agree to pay (must match or exceed driver's listed fare)
Step 6: Click "Confirm Booking" — you'll get an in-app notification immediately
Step 7: Pay via your preferred method (cash/JazzCash/Easypaisa/card)
Tip: Always verify the driver's vehicle plate number before boarding.""",
        "keywords": ["book", "passenger", "how to", "find ride", "search", "join", "ride", "guide", "booking"]
    },

    {
        "id": "howto_003",
        "category": "howto",
        "title": "How to Run the AI Pipeline",
        "content": """How to use the RideKaro AI Optimisation Pipeline:
Step 1: Go to Dashboard tab (home page)
Step 2: Add drivers and passengers either:
  - Manually: Fill in the Add Driver / Add Passenger forms on the left sidebar
  - Automatically: Use "Generate Demo Scenario" to create realistic test data
Step 3: Configure pipeline settings:
  - Departure Hour: set the time of day (affects traffic and surge pricing)
  - Weather: Clear / Rainy / Foggy (affects route costs and ETA)
  - GA Population: genetic algorithm population size (recommended: 40-80)
  - GA Generations: how many iterations to run (recommended: 50-100)
Step 4: Click "▶ Run AI Pipeline" — watch the live progress bar
Step 5: Results appear automatically showing:
  - Optimised ride groups with A* routes
  - Bayesian demand predictions per zone
  - Agent timeline of all 11 agent decisions
  - GA convergence chart
Go to Results tab for detailed algorithm comparisons and charts.""",
        "keywords": ["pipeline", "run", "ai", "how to", "start", "optimise", "generate", "demo", "configure"]
    },

    {
        "id": "howto_004",
        "category": "howto",
        "title": "How to Use the Algorithm Visualiser",
        "content": """Algorithm Visualiser guide on RideKaro:
Step 1: Click "⚡ Algorithms" in the top navigation
Step 2: Select an algorithm from the dropdown:
  - A* (Informed Search) - optimal, uses heuristics, fastest in practice
  - IDA* (Iterative Deepening A*) - memory-efficient version of A*
  - BFS (Breadth-First Search) - guarantees shortest hops but ignores weights
  - DFS (Depth-First Search) - not optimal, explores deep paths first
  - Dijkstra - optimal weighted search, no heuristic
  - Greedy Best-First - fast but not always optimal
  - Hill Climbing - local search, can get stuck
  - Beam Search (width=3) - bounded best-first
  - Bidirectional BFS - searches from both ends simultaneously
  - UCS (Uniform Cost Search) - like Dijkstra with queue
Step 3: Select Source and Destination from Karachi locations
Step 4: Click "▶ Run" to see instant result, or "⏩ Animate" for step-by-step animation
Step 5: Use "🏁 Benchmark All" to compare all 10 algorithms on the same route
The map highlights explored nodes in blue, optimal path in teal, src/dst in special colors.""",
        "keywords": ["algorithm", "astar", "bfs", "dfs", "visualise", "animate", "search", "path", "route", "compare"]
    },

    # ── PLATFORM FEATURES ────────────────────────────────────────────────

    {
        "id": "platform_001",
        "category": "platform",
        "title": "RideKaro AI Agents Explained",
        "content": """RideKaro uses 11 intelligent AI agents working together:
1. EnvironmentAgent: Monitors Karachi road network, applies real-time traffic factors
2. TrafficIntelAgent: Analyses congestion scores per zone, identifies hotspots  
3. DemandPredictionAgent: Bayesian network forecasts pickup demand per zone by hour
4. MatchingAgent: BFS (Breadth-First Search) reachability to find compatible pairs
5. NegotiationAgent: Game theory (Minimax + Alpha-Beta pruning) for fare negotiation
6. GAOptimiserAgent: Genetic Algorithm for globally optimal ride group formation
7. RouteOptimizerAgent: A* pathfinding for optimal multi-waypoint routes
8. CSPSolverAgent: Constraint Satisfaction (AC-3) for capacity, time, detour rules
9. SafetyAgent: Flags low-rated users, long routes, night rides, policy violations
10. CostOptimiserAgent: Dynamic surge pricing, fuel cost splitting, CO₂ calculation
11. OrchestratorAgent: Master controller coordinating all 10 sub-agents in sequence
All agents share an EventBus and emit timeline events visible in the Dashboard.""",
        "keywords": ["agent", "ai", "how it works", "algorithm", "pipeline", "orchestrator", "genetic", "bayesian"]
    },

    {
        "id": "platform_002",
        "category": "platform",
        "title": "Machine Learning Features in RideKaro",
        "content": """RideKaro ML Analytics tab features:
1. Exploratory Data Analysis (EDA): Statistical summary, distance distribution, 
   match rate by hour, feature correlation matrix heatmap
2. Random Forest + Gradient Boosting Ensemble Classifier:
   - Predicts whether a driver-passenger pair will successfully match
   - Trained on 800 synthetic ride records with 9 features
   - Typical accuracy: 85-90% on test set
   - Shows feature importances (detour, time_diff, driver_rating most important)
   - Confusion matrix for TP/TN/FP/FN analysis
3. PCA (Principal Component Analysis): 2D visualization of ride data clusters
4. K-Means Zone Clustering: Groups city locations into demand zones
5. Rating Predictor: Gradient regression predicts expected ride rating
Use the interactive prediction forms to test match prediction and rating prediction
with custom inputs.""",
        "keywords": ["machine learning", "ml", "random forest", "classifier", "predict", "accuracy", "pca", "clustering"]
    },

    {
        "id": "platform_003",
        "category": "platform",
        "title": "Database and Real-World Entities in RideKaro",
        "content": """RideKaro uses SQLite database with 9 real-world tables:
1. users: Full user profiles (name, email, phone, role, rating, verification)
2. vehicles: Car details (make, model, year, plate, capacity, fuel type, AC)
3. rides: Posted rides with origin, destination, departure time, seats, fare, status
4. bookings: Seat reservations with pickup/dropoff stops, fare paid, status
5. reviews: Post-ride ratings (1-5 stars) with comments, auto-updates user rating
6. payments: Transaction records (cash/JazzCash/Easypaisa/card) with reference numbers
7. notifications: In-app alerts for ride updates, bookings, payments
8. match_sessions: Every AI pipeline run logged with stats (CO₂ saved, groups formed)
9. algo_runs: Every algorithm visualiser run stored for analytics

Database auto-creates and seeds on first run with 8 demo Karachi users.
REST API available at /api/db/* for all CRUD operations.
Full-text search available in Database tab for users, rides, and bookings.""",
        "keywords": ["database", "sqlite", "table", "users", "rides", "bookings", "payment", "api", "data"]
    },

    # ── RIDE ETIQUETTE ───────────────────────────────────────────────────

    {
        "id": "etiquette_001",
        "category": "etiquette",
        "title": "Ride Etiquette and Community Guidelines",
        "content": """RideKaro Community Guidelines — please follow for a 5-star experience:

FOR DRIVERS:
- Be at pickup location on time (within 3 minutes of agreed time)
- Keep vehicle clean and have seat belts available
- Don't play music above moderate volume — ask passengers
- No smoking in the vehicle
- Keep AC on during summer (unless passengers prefer otherwise)
- Don't take significant detours without passenger consent

FOR PASSENGERS:
- Be ready at pickup spot 2-3 minutes before agreed time
- Inform driver immediately if your pickup changes
- Don't eat strong-smelling food during the ride
- Be respectful — no loud phone calls, keep conversations moderate
- Pay agreed fare promptly at ride end
- Leave honest ratings — it helps the community

BOTH:
- Cancel early (minimum 15 minutes before) to avoid bad rating
- Report any safety concerns immediately via in-app SOS
- Maintain a rating above 3.5 to continue using the platform""",
        "keywords": ["etiquette", "rules", "guidelines", "behaviour", "cancel", "rating", "community", "respect"]
    },

    # ── CO₂ & ENVIRONMENT ────────────────────────────────────────────────

    {
        "id": "environment_001",
        "category": "environment",
        "title": "Environmental Impact of Carpooling in Karachi",
        "content": """Carpooling environmental impact — Karachi context:
Pakistan emits approximately 0.9 tonnes of CO₂ per capita annually from transport.
Karachi has 6+ million registered vehicles adding to severe air pollution.
RideKaro CO₂ calculation: 2.31 kg CO₂ saved per litre of petrol not burned.

Example impact:
- 3 passengers sharing 1 car for 20 km → saves ~4.6 litres petrol → saves ~10.6 kg CO₂
- 100 daily rides with 3 passengers → saves 460 litres/day → 1,063 kg CO₂/day
- Annual impact (100 rides/day): saves 387,000 kg = 387 tonnes CO₂/year

Additional benefits:
- Reduces traffic volume (each carpooled group removes 2 cars from road)
- Reduces fuel demand (important during Pakistan fuel shortage crises)
- Reduces particulate pollution in Karachi (among world's most polluted cities)
- CO₂ savings displayed per ride group in Dashboard results""",
        "keywords": ["co2", "carbon", "environment", "pollution", "green", "emission", "save", "climate", "fuel"]
    },

    # ── TROUBLESHOOTING ──────────────────────────────────────────────────

    {
        "id": "trouble_001",
        "category": "troubleshooting",
        "title": "Common Issues and Solutions on RideKaro",
        "content": """Common RideKaro issues and fixes:

"Pipeline won't run":
- Make sure you have at least 1 driver AND 1 passenger added
- Click Generate Demo Scenario if you want quick test data
- Check that source and destination are different locations

"No ride groups formed after pipeline":
- Try increasing GA generations (from 30 to 80)
- Add more drivers relative to passengers (1 driver per 2-3 passengers is ideal)
- Check time windows — driver and passenger windows must overlap
- Reduce max detour minutes on passengers

"Algorithm shows ∞ cost":
- The selected source and destination may not be connected in the graph
- Try different city locations — not all Karachi zones have direct connections

"ML tab shows loading forever":
- ML training runs in background — wait 10-15 seconds after first load
- Refresh the ML page tab

"Booking fails with seat error":
- The ride may be full — check seats_available in the ride list
- Try a different ride with more available seats

"Notification not showing":
- Click 🔔 bell icon in top-right header to open notification panel
- Notifications load for demo user U001 by default""",
        "keywords": ["error", "problem", "issue", "not working", "fix", "troubleshoot", "help", "stuck", "loading"]
    },
]


def get_all_documents():
    """Return all knowledge documents."""
    return KNOWLEDGE_DOCUMENTS


def get_by_category(category: str):
    """Filter documents by category."""
    return [d for d in KNOWLEDGE_DOCUMENTS if d["category"] == category]


def get_categories():
    """Get all unique categories."""
    return list(set(d["category"] for d in KNOWLEDGE_DOCUMENTS))


def get_document_texts():
    """Get list of (id, full_text) tuples for embedding."""
    result = []
    for doc in KNOWLEDGE_DOCUMENTS:
        full_text = f"{doc['title']}\n\n{doc['content']}"
        result.append((doc["id"], full_text, doc))
    return result
