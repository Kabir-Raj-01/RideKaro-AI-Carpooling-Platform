"""
database/db.py – SQLite persistence layer for Carpooling AI
Real-world entities: Users, Rides, Bookings, Reviews, Payments, Notifications
"""
from __future__ import annotations
import sqlite3, os, json, uuid
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "carpooling.db")

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    phone       TEXT,
    role        TEXT NOT NULL CHECK(role IN ('driver','passenger','both')),
    rating      REAL DEFAULT 5.0,
    total_rides INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    avatar_color TEXT DEFAULT '#4f8ef7',
    verified    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vehicles (
    id           TEXT PRIMARY KEY,
    user_id      TEXT REFERENCES users(id) ON DELETE CASCADE,
    make         TEXT NOT NULL,
    model        TEXT NOT NULL,
    year         INTEGER,
    color        TEXT,
    plate        TEXT UNIQUE NOT NULL,
    capacity     INTEGER DEFAULT 4,
    fuel_type    TEXT DEFAULT 'petrol',
    ac           INTEGER DEFAULT 1,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rides (
    id              TEXT PRIMARY KEY,
    driver_id       TEXT REFERENCES users(id) ON DELETE SET NULL,
    vehicle_id      TEXT REFERENCES vehicles(id) ON DELETE SET NULL,
    origin          TEXT NOT NULL,
    destination     TEXT NOT NULL,
    waypoints       TEXT DEFAULT '[]',
    departure_time  TEXT NOT NULL,
    seats_total     INTEGER NOT NULL,
    seats_available INTEGER NOT NULL,
    fare_per_seat   REAL NOT NULL,
    distance_km     REAL DEFAULT 0,
    duration_min    REAL DEFAULT 0,
    status          TEXT DEFAULT 'scheduled' CHECK(status IN ('scheduled','active','completed','cancelled')),
    weather         TEXT DEFAULT 'clear',
    notes           TEXT DEFAULT '',
    route_path      TEXT DEFAULT '[]',
    co2_saved_kg    REAL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id           TEXT PRIMARY KEY,
    ride_id      TEXT REFERENCES rides(id) ON DELETE CASCADE,
    passenger_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    seats        INTEGER DEFAULT 1,
    pickup_stop  TEXT,
    dropoff_stop TEXT,
    fare_paid    REAL DEFAULT 0,
    status       TEXT DEFAULT 'pending' CHECK(status IN ('pending','confirmed','cancelled','completed')),
    payment_status TEXT DEFAULT 'unpaid' CHECK(payment_status IN ('unpaid','paid','refunded')),
    booked_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id           TEXT PRIMARY KEY,
    ride_id      TEXT REFERENCES rides(id) ON DELETE CASCADE,
    reviewer_id  TEXT REFERENCES users(id) ON DELETE CASCADE,
    reviewee_id  TEXT REFERENCES users(id) ON DELETE CASCADE,
    rating       REAL NOT NULL CHECK(rating BETWEEN 1 AND 5),
    comment      TEXT DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    id           TEXT PRIMARY KEY,
    booking_id   TEXT REFERENCES bookings(id) ON DELETE CASCADE,
    amount       REAL NOT NULL,
    method       TEXT DEFAULT 'cash' CHECK(method IN ('cash','jazzcash','easypaisa','card')),
    status       TEXT DEFAULT 'pending' CHECK(status IN ('pending','completed','failed','refunded')),
    reference    TEXT UNIQUE,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id           TEXT PRIMARY KEY,
    user_id      TEXT REFERENCES users(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    title        TEXT NOT NULL,
    message      TEXT NOT NULL,
    read         INTEGER DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS algo_runs (
    id           TEXT PRIMARY KEY,
    scenario     TEXT,
    algorithm    TEXT NOT NULL,
    src          TEXT,
    dst          TEXT,
    path         TEXT,
    cost         REAL,
    explored     INTEGER,
    duration_ms  REAL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_sessions (
    id              TEXT PRIMARY KEY,
    n_drivers       INTEGER,
    n_passengers    INTEGER,
    groups_formed   INTEGER,
    matched         INTEGER,
    unmatched       INTEGER,
    co2_saved_kg    REAL,
    total_dist_km   REAL,
    avg_fare_pkr    REAL,
    ga_fitness      REAL,
    pipeline_sec    REAL,
    weather         TEXT,
    hour            INTEGER,
    created_at      TEXT NOT NULL
);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    _seed_demo_data()

def _seed_demo_data():
    """Insert realistic Karachi demo users if DB is empty."""
    with get_conn() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            return
        now = datetime.now().isoformat()
        users = [
            ("U001","Ahmed Khan","ahmed.khan@nuces.edu.pk","+92-300-1234567","driver",4.8,"#00d4b4",1),
            ("U002","Sara Ali","sara.ali@gmail.com","+92-321-9876543","passenger",4.6,"#9d6efd",1),
            ("U003","Bilal Ahmed","bilal.ahmed@outlook.com","+92-333-5554444","both",4.9,"#4f8ef7",1),
            ("U004","Fatima Malik","fatima.malik@yahoo.com","+92-311-2223333","passenger",4.3,"#f0a500",0),
            ("U005","Omar Sheikh","omar.sheikh@gmail.com","+92-345-6667778","driver",4.7,"#20b850",1),
            ("U006","Zainab Hussain","zainab.h@nuces.edu.pk","+92-300-9998887","passenger",4.5,"#f04060",1),
            ("U007","Hassan Raza","hassan.r@gmail.com","+92-321-1112223","driver",4.2,"#4f8ef7",0),
            ("U008","Ayesha Siddiqui","ayesha.s@gmail.com","+92-333-4445556","both",4.9,"#00d4b4",1),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO users(id,name,email,phone,role,rating,avatar_color,verified,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            [(*u, now) for u in users]
        )
        vehicles = [
            ("V001","U001","Toyota","Corolla",2020,"White","KHI-001",4,"petrol",1,now),
            ("V002","U003","Honda","City",2019,"Silver","KHI-002",4,"petrol",1,now),
            ("V003","U005","Suzuki","Swift",2021,"Blue","KHI-003",4,"petrol",0,now),
            ("V004","U007","Toyota","Yaris",2022,"Black","KHI-004",4,"hybrid",1,now),
            ("V005","U008","Honda","BR-V",2023,"White","KHI-005",6,"petrol",1,now),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO vehicles VALUES(?,?,?,?,?,?,?,?,?,?,?)", vehicles
        )

# ── CRUD Helpers ──────────────────────────────────────────────

def create_user(name, email, phone, role, avatar_color="#4f8ef7"):
    uid = "U" + uuid.uuid4().hex[:6].upper()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users(id,name,email,phone,role,avatar_color,created_at) VALUES(?,?,?,?,?,?,?)",
            (uid, name, email, phone, role, avatar_color, datetime.now().isoformat())
        )
    return uid

def get_users(role=None, limit=50):
    with get_conn() as conn:
        if role:
            rows = conn.execute("SELECT * FROM users WHERE role=? OR role='both' ORDER BY created_at DESC LIMIT ?", (role, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

def get_user(uid):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(row) if row else None

def create_ride(driver_id, vehicle_id, origin, destination, departure_time,
                seats_total, fare_per_seat, distance_km=0, duration_min=0,
                weather="clear", notes="", waypoints=None):
    rid = "R" + uuid.uuid4().hex[:7].upper()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO rides(id,driver_id,vehicle_id,origin,destination,departure_time,
               seats_total,seats_available,fare_per_seat,distance_km,duration_min,
               weather,notes,waypoints,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, driver_id, vehicle_id, origin, destination, departure_time,
             seats_total, seats_total, fare_per_seat, distance_km, duration_min,
             weather, notes, json.dumps(waypoints or []), datetime.now().isoformat())
        )
    return rid

def get_rides(status=None, origin=None, destination=None, limit=50):
    with get_conn() as conn:
        q = """SELECT r.*, u.name as driver_name, u.rating as driver_rating,
                      u.avatar_color, v.make, v.model, v.plate, v.ac, v.color as vehicle_color
               FROM rides r
               LEFT JOIN users u ON r.driver_id = u.id
               LEFT JOIN vehicles v ON r.vehicle_id = v.id
               WHERE 1=1"""
        params = []
        if status:
            q += " AND r.status=?"; params.append(status)
        if origin:
            q += " AND r.origin LIKE ?"; params.append(f"%{origin}%")
        if destination:
            q += " AND r.destination LIKE ?"; params.append(f"%{destination}%")
        q += " ORDER BY r.departure_time ASC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(q, params).fetchall()]

def get_ride(rid):
    with get_conn() as conn:
        row = conn.execute("""SELECT r.*, u.name as driver_name, u.rating as driver_rating,
                              u.avatar_color, u.phone as driver_phone, v.make, v.model, v.plate, v.ac
                              FROM rides r
                              LEFT JOIN users u ON r.driver_id=u.id
                              LEFT JOIN vehicles v ON r.vehicle_id=v.id
                              WHERE r.id=?""", (rid,)).fetchone()
        return dict(row) if row else None

def create_booking(ride_id, passenger_id, seats, pickup_stop, dropoff_stop, fare_paid):
    bid = "B" + uuid.uuid4().hex[:7].upper()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO bookings(id,ride_id,passenger_id,seats,pickup_stop,dropoff_stop,fare_paid,booked_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (bid, ride_id, passenger_id, seats, pickup_stop, dropoff_stop, fare_paid, datetime.now().isoformat())
        )
        conn.execute("UPDATE rides SET seats_available=seats_available-? WHERE id=?", (seats, ride_id))
    return bid

def get_bookings_for_ride(ride_id):
    with get_conn() as conn:
        rows = conn.execute("""SELECT b.*, u.name as passenger_name, u.rating as passenger_rating,
                               u.avatar_color, u.phone
                               FROM bookings b JOIN users u ON b.passenger_id=u.id
                               WHERE b.ride_id=? ORDER BY b.booked_at ASC""", (ride_id,)).fetchall()
        return [dict(r) for r in rows]

def get_user_rides(user_id):
    with get_conn() as conn:
        as_driver = conn.execute("""SELECT r.*, 'driver' as my_role FROM rides r WHERE r.driver_id=?
                                    ORDER BY r.departure_time DESC LIMIT 20""", (user_id,)).fetchall()
        as_passenger = conn.execute("""SELECT r.*, 'passenger' as my_role FROM rides r
                                       JOIN bookings b ON r.id=b.ride_id
                                       WHERE b.passenger_id=? ORDER BY r.departure_time DESC LIMIT 20""", (user_id,)).fetchall()
        return {"as_driver": [dict(r) for r in as_driver], "as_passenger": [dict(r) for r in as_passenger]}

def add_review(ride_id, reviewer_id, reviewee_id, rating, comment=""):
    rid = "REV" + uuid.uuid4().hex[:6].upper()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO reviews(id,ride_id,reviewer_id,reviewee_id,rating,comment,created_at) VALUES(?,?,?,?,?,?,?)",
            (rid, ride_id, reviewer_id, reviewee_id, rating, comment, datetime.now().isoformat())
        )
        avg = conn.execute("SELECT AVG(rating) FROM reviews WHERE reviewee_id=?", (reviewee_id,)).fetchone()[0]
        conn.execute("UPDATE users SET rating=? WHERE id=?", (round(avg, 2), reviewee_id))
    return rid

def create_payment(booking_id, amount, method="cash"):
    pid = "PAY" + uuid.uuid4().hex[:6].upper()
    ref = "TXN" + uuid.uuid4().hex[:8].upper()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO payments(id,booking_id,amount,method,status,reference,created_at) VALUES(?,?,?,?,?,?,?)",
            (pid, booking_id, amount, method, "completed", ref, datetime.now().isoformat())
        )
        conn.execute("UPDATE bookings SET payment_status='paid', status='confirmed' WHERE id=?", (booking_id,))
    return {"payment_id": pid, "reference": ref}

def add_notification(user_id, ntype, title, message):
    nid = "N" + uuid.uuid4().hex[:7].upper()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notifications(id,user_id,type,title,message,created_at) VALUES(?,?,?,?,?,?)",
            (nid, user_id, ntype, title, message, datetime.now().isoformat())
        )

def get_notifications(user_id, unread_only=False):
    with get_conn() as conn:
        q = "SELECT * FROM notifications WHERE user_id=?"
        if unread_only:
            q += " AND read=0"
        q += " ORDER BY created_at DESC LIMIT 20"
        return [dict(r) for r in conn.execute(q, (user_id,)).fetchall()]

def mark_notifications_read(user_id):
    with get_conn() as conn:
        conn.execute("UPDATE notifications SET read=1 WHERE user_id=?", (user_id,))

def save_algo_run(algorithm, src, dst, path, cost, explored, duration_ms, scenario=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO algo_runs(id,scenario,algorithm,src,dst,path,cost,explored,duration_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("AR" + uuid.uuid4().hex[:6].upper(), scenario, algorithm, src, dst,
             json.dumps(path), cost, explored, duration_ms, datetime.now().isoformat())
        )

def save_match_session(data: dict):
    sid = "MS" + uuid.uuid4().hex[:6].upper()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO match_sessions(id,n_drivers,n_passengers,groups_formed,matched,unmatched,
               co2_saved_kg,total_dist_km,avg_fare_pkr,ga_fitness,pipeline_sec,weather,hour,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, data.get("n_drivers",0), data.get("n_passengers",0),
             data.get("groups_formed",0), data.get("matched",0), data.get("unmatched",0),
             data.get("co2_saved_kg",0), data.get("total_dist_km",0),
             data.get("avg_fare_pkr",0), data.get("ga_fitness",0),
             data.get("pipeline_sec",0), data.get("weather","clear"),
             data.get("hour",8), datetime.now().isoformat())
        )
    return sid

def get_analytics():
    with get_conn() as conn:
        total_rides  = conn.execute("SELECT COUNT(*) FROM rides").fetchone()[0]
        total_users  = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_bookings = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
        completed    = conn.execute("SELECT COUNT(*) FROM rides WHERE status='completed'").fetchone()[0]
        co2_total    = conn.execute("SELECT COALESCE(SUM(co2_saved_kg),0) FROM rides").fetchone()[0]
        avg_rating   = conn.execute("SELECT ROUND(AVG(rating),2) FROM users").fetchone()[0]
        revenue      = conn.execute("SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='completed'").fetchone()[0]
        sessions     = conn.execute("SELECT COUNT(*) FROM match_sessions").fetchone()[0]
        algo_runs    = conn.execute("SELECT COUNT(*) FROM algo_runs").fetchone()[0]
        recent_rides = conn.execute("""SELECT r.origin, r.destination, r.status, r.departure_time,
                                        u.name as driver_name FROM rides r LEFT JOIN users u ON r.driver_id=u.id
                                        ORDER BY r.created_at DESC LIMIT 8""").fetchall()
        ride_by_status = dict(conn.execute("SELECT status, COUNT(*) FROM rides GROUP BY status").fetchall())
        top_routes     = conn.execute("""SELECT origin||' → '||destination as route, COUNT(*) as cnt
                                         FROM rides GROUP BY route ORDER BY cnt DESC LIMIT 5""").fetchall()
        return {
            "total_rides": total_rides, "total_users": total_users,
            "total_bookings": total_bookings, "completed_rides": completed,
            "co2_saved_kg": round(co2_total, 2), "avg_rating": avg_rating or 5.0,
            "total_revenue_pkr": round(revenue, 0), "match_sessions": sessions,
            "algo_runs": algo_runs,
            "recent_rides": [dict(r) for r in recent_rides],
            "ride_by_status": ride_by_status,
            "top_routes": [dict(r) for r in top_routes],
        }
