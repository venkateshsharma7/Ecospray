#!/usr/bin/env python3
# app.py - EcoSpray main Flask application (full file)
# This file includes:
#  - robust MongoDB connection (certifi, retries)
#  - pest-damage analysis (chewing/sucking/discoloration)
#  - authoritative DB-backed latest & /dashboard-data endpoint (cached stats/history)
#  - upload route that persists and returns authoritative saved record
#  - helpful debug route for SSL/Python info
#
# Make sure to add `certifi` to requirements.txt and set MONGO_URI in the env on your host.

from flask import Flask, request, jsonify, render_template, send_from_directory, abort, Response
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timezone
import logging
import cv2
import numpy as np
from PIL import Image
import math
import time
from threading import Lock
from urllib.parse import quote_plus
import ssl
import sys

# Optional pymongo
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    mongo_available = True
except Exception:
    MongoClient = None
    PyMongoError = Exception
    mongo_available = False

# certifi for CA bundle
try:
    import certifi
    have_certifi = True
except Exception:
    certifi = None
    have_certifi = False

# ----- Config -----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Keep MONGO_URI in env for safety; fallback if needed (encoded password assumed)
MONGO_URI = os.environ.get('MONGO_URI',
                           'mongodb+srv://venkateshsharma:Vvs%402005@cluster0.ie4uxy6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

# Connection retry settings
MONGO_CONN_RETRIES = int(os.environ.get('MONGO_CONN_RETRIES', '4'))
MONGO_CONN_INITIAL_BACKOFF = float(os.environ.get('MONGO_CONN_INITIAL_BACKOFF', '1.0'))  # seconds

# Dashboard cache TTL (seconds)
DASHBOARD_TTL = int(os.environ.get('DASHBOARD_TTL', '10'))

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Allow CORS for IoT/demo (restrict in production)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcoSpray")

# ----- Storage fallback -----
in_memory_data = []
client = None
collection = None
use_mongo = False

def log_runtime_ssl_info():
    try:
        info = {
            "python_version": sys.version,
            "openssl_version": getattr(ssl, 'OPENSSL_VERSION', 'unknown')
        }
        logger.info("Runtime info: %s", info)
    except Exception:
        logger.exception("Failed to log runtime ssl info")

def init_mongo_connection():
    """
    Robustly attempt to initialize MongoDB connection using certifi CA bundle if available.
    This will try multiple times with exponential backoff. Sets global client, collection, use_mongo.
    """
    global client, collection, use_mongo
    if not mongo_available or not MONGO_URI:
        logger.info("MongoDB not configured or pymongo not installed; using in-memory storage.")
        use_mongo = False
        return

    # try several attempts
    backoff = MONGO_CONN_INITIAL_BACKOFF
    for attempt in range(1, MONGO_CONN_RETRIES + 1):
        try:
            logger.info("Attempting MongoDB connection (attempt %d/%d)...", attempt, MONGO_CONN_RETRIES)
            # prefer explicit TLS and CA bundle if certifi is present
            if have_certifi:
                client_candidate = MongoClient(MONGO_URI,
                                               serverSelectionTimeoutMS=8000,
                                               tls=True,
                                               tlsCAFile=certifi.where())
            else:
                # fallback without explicit CA (less robust)
                client_candidate = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000, tls=True)

            # try ping
            client_candidate.admin.command('ping')
            db = client_candidate.get_database('ecospray')
            coll = db.get_collection('disease_data')
            # optional write test (insert and delete) to ensure write perms
            try:
                test_doc = {"__connect_test": True, "ts": datetime.utcnow().isoformat()}
                res = coll.insert_one(test_doc)
                coll.delete_one({"_id": res.inserted_id})
            except Exception:
                # if no write permission, don't fail here — we may still read
                logger.warning("MongoDB connected but write test failed or not permitted (insert/delete).")

            # success
            client = client_candidate
            collection = coll
            use_mongo = True
            logger.info("MongoDB connection established (attempt %d).", attempt)
            return
        except Exception as e:
            logger.warning("MongoDB connection attempt %d failed: %s", attempt, e)
            if attempt < MONGO_CONN_RETRIES:
                logger.info("Retrying in %.1f seconds...", backoff)
                time.sleep(backoff)
                backoff *= 2  # exponential
            else:
                logger.exception("MongoDB connection failed after %d attempts. Falling back to in-memory.", MONGO_CONN_RETRIES)
                use_mongo = False
                client = None
                collection = None

# initialize on startup
log_runtime_ssl_info()
init_mongo_connection()

# ----- Helpers -----
def allowed_file(filename):
    if not filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS

def safe_filename_with_ts(filename):
    filename = secure_filename(filename)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{ts}_{filename}"

def load_image(filepath):
    """Robust loader: cv2.imdecode -> cv2.imread -> Pillow fallback. Returns BGR numpy or None."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        try:
            img = cv2.imread(filepath)
        except Exception:
            img = None
    if img is None:
        try:
            pil_img = Image.open(filepath).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    return img

# ----- Image analysis & prediction (pest damage heuristic) -----
def analyze_pest_damage(img):
    """
    Analyze an image for pest damage signs (chewing holes, sucking damage yellow spots, discoloration).
    Returns:
    - severity (0-100 int)
    - chewing_damage_pct (float)
    - sucking_damage_pct (float)
    - discoloration_pct (float)
    - galls_pct (float)  # placeholder 0.0
    - sticky_residue_pct (float)  # placeholder 0.0
    - infected_area: sum of damage pixels
    - total_area: total pixels in image
    """
    if img is None or img.size == 0:
        return {
            "severity": 0,
            "chewing_damage_pct": 0.0,
            "sucking_damage_pct": 0.0,
            "discoloration_pct": 0.0,
            "galls_pct": 0.0,
            "sticky_residue_pct": 0.0,
            "infected_area": 0,
            "total_area": 0,
        }
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Chewing damage approximation: dark/black holes (low brightness)
    lower_hole = np.array([0, 0, 0])
    upper_hole = np.array([180, 255, 50])
    mask_hole = cv2.inRange(hsv, lower_hole, upper_hole)
    chewing_pixels = int(np.count_nonzero(mask_hole))

    # Sucking damage approximation: yellow spots (yellow color range)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    sucking_pixels = int(np.count_nonzero(mask_yellow))

    # Discoloration approximation: brown patches
    lower_brown = np.array([5, 50, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    discoloration_pixels = int(np.count_nonzero(mask_brown))

    total_pixels = img.shape[0] * img.shape[1]

    severity = int(((chewing_pixels + sucking_pixels + discoloration_pixels) / total_pixels) * 100) if total_pixels > 0 else 0

    chewing_pct = (chewing_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    sucking_pct = (sucking_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    discoloration_pct = (discoloration_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # Galls and sticky residue detection placeholders (set to 0.0)
    galls_pct = 0.0
    sticky_residue_pct = 0.0

    return {
        "severity": severity,
        "chewing_damage_pct": round(chewing_pct, 2),
        "sucking_damage_pct": round(sucking_pct, 2),
        "discoloration_pct": round(discoloration_pct, 2),
        "galls_pct": galls_pct,
        "sticky_residue_pct": sticky_residue_pct,
        "infected_area": chewing_pixels + sucking_pixels + discoloration_pixels,
        "total_area": total_pixels,
    }

def softmax(scores):
    exps = [math.exp(s) for s in scores]
    s = sum(exps)
    if s == 0:
        return [0.0 for _ in scores]
    return [float(e / s) for e in exps]

def predict_pest_damage_with_probs(analysis):
    """
    Predict pest damage types and probabilities based on the analysis.
    Categories:
    - Chewing Damage
    - Sucking Damage
    - Leaf Discoloration
    - Healthy
    """
    sev = analysis['severity'] / 100.0
    chewing = analysis['chewing_damage_pct'] / 100.0
    sucking = analysis['sucking_damage_pct'] / 100.0
    discoloration = analysis['discoloration_pct'] / 100.0

    # Heuristic scores for pest damage types
    score_chewing = 4.0 * chewing + 3.0 * sev
    score_sucking = 4.0 * sucking + 2.5 * sev
    score_discoloration = 2.5 * discoloration + 1.2 * sev
    score_healthy = 3.0 * (1.0 - sev)

    scores = [score_chewing, score_sucking, score_discoloration, score_healthy]
    labels = ["Chewing Damage", "Sucking Damage", "Leaf Discoloration", "Healthy"]
    probs = softmax(scores)
    prob_dict = {labels[i]: round(probs[i] * 100, 2) for i in range(len(labels))}
    predicted = max(prob_dict.items(), key=lambda x: x[1])[0]
    return predicted, prob_dict

# ----- Pesticide recommendation (server-side) -----
def get_pesticide_plan_pest(severity):
    s = int(severity or 0)
    if s <= 10:
        return {
            "tier": "Healthy / Monitor",
            "summary": "No pesticide required. Monitor and apply biological control.",
            "water_l_per_ha": [100, 300],
            "dose_ml_per_l": [0, 0]
        }
    elif s <= 30:
        return {
            "tier": "Low (Preventive)",
            "summary": "Preventive measures like neem oil or insecticidal soap recommended.",
            "water_l_per_ha": [200, 400],
            "dose_ml_per_l": [0.5, 1.0]
        }
    elif s <= 60:
        return {
            "tier": "Moderate (Treat)",
            "summary": "Apply targeted insecticides; follow label instructions.",
            "water_l_per_ha": [300, 600],
            "dose_ml_per_l": [1.0, 2.0]
        }
    else:
        return {
            "tier": "Severe (Intensive)",
            "summary": "High pest damage - use systemic insecticides and seek expert advice.",
            "water_l_per_ha": [400, 800],
            "dose_ml_per_l": [2.0, 5.0]
        }

def compute_pesticide_recommendation(severity, area_m2=100.0, mode="knapsack"):
    plan = get_pesticide_plan_pest(severity)
    area = max(1.0, float(area_m2 or 100.0))
    ha = area / 10000.0  # convert m^2 to hectares

    water_min = plan['water_l_per_ha'][0] * ha
    water_max = plan['water_l_per_ha'][1] * ha
    dose_min = plan['dose_ml_per_l'][0]
    dose_max = plan['dose_ml_per_l'][1]

    water_mid = (water_min + water_max) / 2.0
    dose_mid = (dose_min + dose_max) / 2.0

    total_min_ml = water_min * dose_min
    total_max_ml = water_max * dose_max
    total_mid_ml = water_mid * dose_mid

    # tank volumes: knapsack = 15 L, sprayer = 200 L
    tank_vol = 15 if mode == 'knapsack' else 200
    fills = max(1, int(math.ceil(water_mid / tank_vol)))

    return {
        "area_m2": round(area, 3),
        "mode": mode,
        "tier": plan['tier'],
        "summary": plan['summary'],
        "water_l_min": round(water_min, 3),
        "water_l_max": round(water_max, 3),
        "dose_ml_per_l_min": round(dose_min, 3),
        "dose_ml_per_l_max": round(dose_max, 3),
        "total_pesticide_ml_min": round(total_min_ml, 3),
        "total_pesticide_ml_max": round(total_max_ml, 3),
        "total_pesticide_ml_mid": round(total_mid_ml, 3),
        "recommended_water_l_mid": round(water_mid, 3),
        "recommended_dose_ml_per_l_mid": round(dose_mid, 3),
        "tank_volume_l": tank_vol,
        "estimated_tank_fills": fills
    }

# ----- Global last result -----
last_result = {
    "severity": 0,
    "disease": "None",
    "probabilities": {},
    "chewing_damage_pct": 0.0,
    "sucking_damage_pct": 0.0,
    "discoloration_pct": 0.0,
    "timestamp": None,
    "pesticide_recommendation": None
}

# ----- Dashboard cache (single endpoint) -----
_dashboard_cache = {"ts": 0, "data": None}
_dashboard_lock = Lock()

@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    """
    Returns a single JSON with:
      - latest: last persisted record (DB-backed when available)
      - stats: total_uploads, avg_severity (cached)
      - history: recent records (n=8) (cached)
    """
    now = time.time()
    with _dashboard_lock:
        # Use cached stats/history if fresh
        if _dashboard_cache["data"] and now - _dashboard_cache["ts"] < DASHBOARD_TTL:
            payload = _dashboard_cache["data"].copy()
        else:
            # recompute stats and history and cache it
            try:
                if use_mongo and collection is not None:
                    total_uploads = int(collection.count_documents({}))
                    avg = 0.0
                    if total_uploads > 0:
                        agg = list(collection.aggregate([{"$group": {"_id": None, "avgSeverity": {"$avg": "$severity"}}}]))
                        avg = round(float(agg[0]['avgSeverity']) if agg else 0.0, 2)
                else:
                    total_uploads = len(in_memory_data)
                    avg = round(sum(d['severity'] for d in in_memory_data) / total_uploads, 2) if total_uploads > 0 else 0.0
            except Exception:
                total_uploads = 0
                avg = 0.0

            try:
                if use_mongo and collection is not None:
                    history_docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(8))
                else:
                    history_docs = list(reversed(in_memory_data))[-8:]
            except Exception:
                history_docs = []

            payload = {"latest": None, "stats": {"total_uploads": total_uploads, "avg_severity": avg}, "history": history_docs}
            _dashboard_cache["data"] = payload
            _dashboard_cache["ts"] = now

        # Now always refresh latest from DB (authoritative)
        try:
            if use_mongo and collection is not None:
                latest_doc = collection.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
                if latest_doc:
                    payload["latest"] = latest_doc
                else:
                    payload["latest"] = last_result.copy()
            else:
                payload["latest"] = last_result.copy()
        except Exception:
            payload["latest"] = last_result.copy()

        return jsonify(payload)

# debug route to inspect runtime OpenSSL / Python version (temporary)
@app.route('/_debug_ssl', methods=['GET'])
def _debug_ssl():
    try:
        info = {
            "python_version": sys.version,
            "openssl_version": getattr(ssl, 'OPENSSL_VERSION', 'unknown'),
            "mongo_enabled": use_mongo
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----- Routes -----
@app.route('/')
def index():
    stats = compute_stats()
    return render_template('index.html',
                           severity=last_result['severity'],
                           disease=last_result['disease'],
                           timestamp=last_result.get('timestamp'),
                           total_uploads=stats['total_uploads'],
                           avg_severity=stats['avg_severity'],
                           top_diseases=stats['top_diseases'])

@app.route('/upload', methods=['POST'])
def upload():
    # Accept optional area and mode (from form, query or JSON)
    area = None
    mode = None
    if request.content_type and 'application/json' in request.content_type:
        body = request.get_json(silent=True) or {}
        area = body.get('area')
        mode = body.get('mode')
    else:
        area = request.form.get('area') or request.args.get('area')
        mode = request.form.get('mode') or request.args.get('mode')

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        allowed = ','.join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({'error': f'File extension not allowed. Allowed: {allowed}'}), 400

    filename = safe_filename_with_ts(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
    except Exception as e:
        logger.exception("Failed to save file: %s", e)
        return jsonify({'error': 'Failed to save file'}), 500

    img = load_image(filepath)
    if img is None:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({'error': 'Invalid or unreadable image format'}), 400

    # Analyze pest damage in image
    analysis = analyze_pest_damage(img)
    severity = int(analysis['severity'])
    predicted, probs = predict_pest_damage_with_probs(analysis)

    # server-set ISO timestamp in UTC (authoritative)
    timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    area_val = float(area) if area is not None else 100.0
    mode_val = mode if mode in ('knapsack', 'sprayer') else 'knapsack'

    pesticide_reco = compute_pesticide_recommendation(severity, area_val, mode_val)

    record = {
        "filename": filename,
        "severity": severity,
        "disease": predicted,
        "probabilities": probs,
        "chewing_damage_pct": analysis['chewing_damage_pct'],
        "sucking_damage_pct": analysis['sucking_damage_pct'],
        "discoloration_pct": analysis['discoloration_pct'],
        "pesticide_recommendation": pesticide_reco,
        "timestamp": timestamp
    }

    # Persist record and then read back authoritative saved doc (to avoid race)
    try:
        if use_mongo and collection is not None:
            # insert and then fetch the exact inserted document
            res = collection.insert_one(record)
            # fetch the inserted doc (without _id)
            saved = collection.find_one({"_id": res.inserted_id}, {"_id": 0})
            if saved:
                authoritative = saved
            else:
                # as fallback, read latest by timestamp
                authoritative = collection.find_one({}, {"_id": 0}, sort=[("timestamp", -1)]) or record
        else:
            # in-memory append and treat this process as authoritative
            in_memory_data.append(record)
            authoritative = record
    except Exception as e:
        logger.exception("Persistence failed; storing in-memory fallback. Error: %s", e)
        in_memory_data.append(record)
        authoritative = record

    # Update per-process last_result from authoritative record
    last_result.update({
        "severity": authoritative.get("severity", severity),
        "disease": authoritative.get("disease", predicted),
        "probabilities": authoritative.get("probabilities", probs),
        "chewing_damage_pct": authoritative.get("chewing_damage_pct", analysis['chewing_damage_pct']),
        "sucking_damage_pct": authoritative.get("sucking_damage_pct", analysis['sucking_damage_pct']),
        "discoloration_pct": authoritative.get("discoloration_pct", analysis['discoloration_pct']),
        "timestamp": authoritative.get("timestamp", timestamp),
        "pesticide_recommendation": authoritative.get("pesticide_recommendation", pesticide_reco)
    })

    # Clear dashboard cache (stats/history) so next dashboard-data returns fresh values
    with _dashboard_lock:
        _dashboard_cache["ts"] = 0
        _dashboard_cache["data"] = None

    response = {
        "severity": last_result["severity"],
        "disease": last_result["disease"],
        "probabilities": last_result["probabilities"],
        "chewing_damage_pct": last_result["chewing_damage_pct"],
        "sucking_damage_pct": last_result["sucking_damage_pct"],
        "discoloration_pct": last_result["discoloration_pct"],
        "pesticide_recommendation": last_result["pesticide_recommendation"],
        "filename": filename,
        "timestamp": last_result["timestamp"]
    }
    return jsonify(response)

@app.route('/latest-severity', methods=['GET'])
def latest():
    # Return DB-backed latest if possible for consistency across instances
    try:
        if use_mongo and collection is not None:
            latest_doc = collection.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
            if latest_doc:
                return jsonify(latest_doc)
    except Exception:
        logger.exception("Error fetching latest from DB; returning in-memory last_result")
    # fallback
    return jsonify(last_result)

@app.route('/pesticide-recommendation', methods=['GET'])
def pesticide_recommendation():
    """
    Compute recommendation using current latest severity and provided area/mode.
    Query params:
    - area (m2) default 100
    - mode: knapsack | sprayer (default knapsack)
    """
    try:
        area = float(request.args.get('area', 100.0))
    except Exception:
        area = 100.0
    mode = request.args.get('mode', 'knapsack')
    if mode not in ('knapsack', 'sprayer'):
        mode = 'knapsack'

    # authoritative severity from DB if possible
    severity = 0
    try:
        if use_mongo and collection is not None:
            latest_doc = collection.find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
            if latest_doc and 'severity' in latest_doc:
                severity = int(latest_doc['severity'])
            else:
                severity = int(last_result.get('severity', 0) or 0)
        else:
            severity = int(last_result.get('severity', 0) or 0)
    except Exception:
        severity = int(last_result.get('severity', 0) or 0)

    reco = compute_pesticide_recommendation(severity, area, mode)

    return jsonify({
        "severity": severity,
        "area_m2": reco['area_m2'],
        "mode": reco['mode'],
        "pesticide_recommendation": reco
    })

@app.route('/stats', methods=['GET'])
def stats():
    stats = compute_stats()
    return jsonify(stats)

def compute_stats():
    try:
        if use_mongo and collection is not None:
            total = collection.count_documents({})
            if total > 0:
                agg = list(collection.aggregate([
                    {"$group": {"_id": None, "avgSeverity": {"$avg": "$severity"}}}
                ]))
                avg = round(float(agg[0]['avgSeverity']) if agg else 0.0, 2)
                top = list(collection.aggregate([
                    {"$group": {"_id": "$disease", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 5}
                ]))
                top_diseases = {d['_id']: d['count'] for d in top}
            else:
                total = 0
                avg = 0.0
                top_diseases = {}
        else:
            total = len(in_memory_data)
            avg = round(sum(d['severity'] for d in in_memory_data) / total, 2) if total > 0 else 0.0
            counts = {}
            for d in in_memory_data:
                label = d.get('disease', 'Unknown')
                counts[label] = counts.get(label, 0) + 1
            top_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_diseases = {k: v for k, v in top_sorted}
        return {"total_uploads": int(total), "avg_severity": float(avg), "top_diseases": top_diseases}
    except Exception as e:
        logger.exception("compute_stats failed: %s", e)
        return {"total_uploads": 0, "avg_severity": 0.0, "top_diseases": {}}

@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    safe = secure_filename(filename)
    fullpath = os.path.join(app.config['UPLOAD_FOLDER'], safe)
    if not os.path.exists(fullpath):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe)

@app.route('/history', methods=['GET'])
def history():
    try:
        n = int(request.args.get('n', 20))
    except Exception:
        n = 20
    n = max(1, min(200, n))
    try:
        if use_mongo and collection is not None:
            docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(n))
        else:
            docs = list(reversed(in_memory_data))[-n:]
    except Exception:
        docs = list(reversed(in_memory_data))[-n:]
    return jsonify({"count": len(docs), "results": docs})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "use_mongo": use_mongo})

if __name__ == '__main__':
    # IMPORTANT: set debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=False)
