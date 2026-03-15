from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# .env file-la irukkura secrets-ah load panrom
load_dotenv()

app = Flask(__name__)

# 1. Load the Model & Scaler
model = joblib.load('models/health_edge_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# 2. MongoDB Atlas (Cloud) Connection Setup
atlas_uri = os.getenv("MONGO_URI")
client = MongoClient(atlas_uri)
db = client['healthedge_db']
patients_collection = db['patients']

@app.route('/')
def start():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form.get('username'), request.form.get('password')
        if u == 'admin' and p == 'admin123':
            return redirect(url_for('dashboard'))
    return render_template('login.html', error='Invalid Credentials' if request.method == 'POST' else None)

@app.route('/dashboard')
def dashboard():
    total = patients_collection.count_documents({})
    urgent = patients_collection.count_documents({"status": {"$regex": "URGENT", "$options": "i"}})
    stable = total - urgent
    latest_records = list(patients_collection.find({}, {'_id': 0}).sort('_id', -1).limit(5))
    return render_template('dashboard.html', total=total, urgent=urgent, stable=stable, records=latest_records)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/logout')
def logout():
    return redirect(url_for('start'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name', 'Patient')
        gender = request.form.get('gender', 'Not Specified')
        age = request.form.get('age', '0')
        glucose = request.form.get('glucose', '0')
        bp = request.form.get('bp', '0')
        bmi = request.form.get('bmi', '0')
        pregnancies = request.form.get('pregnancies', 0)
        
        raw = [pregnancies, glucose, bp, 20, 80, bmi, 0.47, age]
        final = [float(x) if x else 0.0 for x in raw]
        prob_val = model.predict_proba(scaler.transform(np.array([final])))[0][1] * 100
        prob_str = f"{prob_val:.2f}%"

        gv, bv = float(glucose), float(bp)
        if gv > 200 or bv > 110:
            status, style = "URGENT CASE (Clinical Risk)", "danger"
            advice = "CRITICAL: Vitals are at dangerous levels. Immediate medical consultation is required."
        elif prob_val > 70:
            status, style = "URGENT CASE (AI Risk)", "danger"
            advice = "High AI-detected probability of diabetes. Please consult a specialist."
        else:
            status, style = "STABLE / LOW RISK", "success"
            advice = "Vitals are within acceptable range. Maintain a healthy diet and active lifestyle."

        diet = {"Breakfast": "Oats or Whole-wheat Idli", "Lunch": "Leafy greens with Brown Rice", "Dinner": "Clear vegetable soup"}
        
        new_rec = {
            "name": name, 
            "gender": gender, 
            "age": age, 
            "glucose": glucose, 
            "bp": bp,  
            "bmi": bmi,  
            "prob": prob_str, 
            "status": status
        }
        
        patients_collection.insert_one(new_rec)

        return render_template('analysis.html', name=name, gender=gender, age=age, 
                               glucose=glucose, bp=bp, bmi=bmi, status=status, 
                               prob=prob_str, style=style, advice=advice, diet=diet)
    except Exception as e:
        return render_template('analysis.html', prediction_text=f"Error: {str(e)}")

@app.route('/records')
def records():
    all_records = list(patients_collection.find({}, {'_id': 0}).sort('_id', -1))
    return render_template('records.html', records=all_records)

# --- HOSTING CONFIG UPDATE ---
if __name__ == "__main__":
    # Render or matha cloud platforms-ku dynamic port set panrom
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)