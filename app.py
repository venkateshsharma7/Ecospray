from flask import Flask, request, jsonify, render_template, send_from_directory, abort
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

# Optional pymongo
try:
    from pymongo import MongoClient
    mongo_available = True
except Exception:
    mongo_available = False

# ----- Config -----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# Keep MONGO_URI in env for safety; fallback if needed
MONGO_URI = os.environ.get('MONGO_URI',
                           'mongodb+srv://venkateshsharma:Vvs%402005@cluster0.ie4uxy6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

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
if mongo_available and MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client.get_database('ecospray')
        collection = db.get_collection('disease_data')
        use_mongo = True
        logger.info("MongoDB connected.")
    except Exception as e:
        logger.warning("MongoDB unavailable; using in-memory. Error: %s", e)
        use_mongo = False
else:
    logger.info("MongoDB not configured; using in-memory storage.")

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
DASHBOARD_TTL = 10  # seconds

@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    now = time.time()
    with _dashboard_lock:
        if _dashboard_cache["data"] and now - _dashboard_cache["ts"] < DASHBOARD_TTL:
            return jsonify(_dashboard_cache["data"])

        latest = last_result.copy()

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

        payload = {"latest": latest, "stats": {"total_uploads": total_uploads, "avg_severity": avg}, "history": history_docs}
        _dashboard_cache["data"] = payload
        _dashboard_cache["ts"] = now
        return jsonify(payload)


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
    timestamp = datetime.now(timezone.utc).isoformat()

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

    # Persist record
    try:
        if use_mongo and collection is not None:
            collection.insert_one(record)
        else:
            in_memory_data.append(record)
    except Exception:
        logger.exception("Persistence failed; storing in-memory fallback.")
        in_memory_data.append(record)

    # Update last_result
    last_result.update({
        "severity": severity,
        "disease": predicted,
        "probabilities": probs,
        "chewing_damage_pct": analysis['chewing_damage_pct'],
        "sucking_damage_pct": analysis['sucking_damage_pct'],
        "discoloration_pct": analysis['discoloration_pct'],
        "timestamp": timestamp,
        "pesticide_recommendation": pesticide_reco
    })

    # Clear dashboard cache
    with _dashboard_lock:
        _dashboard_cache["ts"] = 0
        _dashboard_cache["data"] = None

    response = {
        "severity": severity,
        "disease": predicted,
        "probabilities": probs,
        "chewing_damage_pct": analysis['chewing_damage_pct'],
        "sucking_damage_pct": analysis['sucking_damage_pct'],
        "discoloration_pct": analysis['discoloration_pct'],
        "pesticide_recommendation": pesticide_reco,
        "filename": filename,
        "timestamp": timestamp
    }
    return jsonify(response)


@app.route('/latest-severity', methods=['GET'])
def latest():
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
    if use_mongo and collection is not None:
        docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(n))
    else:
        docs = list(reversed(in_memory_data))[-n:]
    return jsonify({"count": len(docs), "results": docs})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "use_mongo": use_mongo})


if __name__ == '__main__':
    # IMPORTANT: set debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=False)
