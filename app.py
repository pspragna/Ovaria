import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, TrackingEntry  # use models.py for SQLAlchemy models
from utils import prepare_input_from_form, predict_from_df, get_recommendations
import json
import io
import csv
from datetime import date


# ------------------------------
# Flask App Initialization
# ------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('OVARIA_SECRET', 'replace_this_with_a_strong_secret')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'ovaria.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# Bind SQLAlchemy to app
db.init_app(app)


# ------------------------------
# Flask-Login Setup
# ------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ------------------------------
# Database Initialization
# ------------------------------
with app.app_context():
    db.create_all()
    # Ensure at least one admin exists
    if not User.query.filter_by(is_admin=True).first():
        admin = User(
            username="admin",
            password=generate_password_hash("admin123"),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created: username=admin, password=admin123")


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return redirect(url_for('register'))


# -------- Register --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        role = request.form.get("role", "user")  # default to 'user'

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

        new_user = User(username=username, is_admin=(role == "admin"))
        new_user.password = generate_password_hash(password)

        db.session.add(new_user)
        db.session.commit()

        flash(f"{role.capitalize()} registered successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# -------- Login --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


# -------- Logout --------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'info')
    return redirect(url_for('login'))


# -------- Dashboard --------
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    return render_template('dashboard.html')


# -------- About / Awareness Page --------
@app.route('/about')
@login_required
def about():
    return render_template('about.html')


# -------- Questionnaire --------
@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    import numpy as np
    import pandas as pd

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    features_path = os.path.join(BASE_DIR, 'features.json')

    # Load features.json
    features_json = []
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features_json = json.load(f)

    # REMOVE BMI FEATURE IF PRESENT
    features_json = [f for f in features_json if f["name"] != "bmi"]

    if request.method == 'POST':
        try:
            ENCODING_MAP = {
                "Yes": 1,
                "No": 0,
                "Light": 0,
                "Normal": 1,
                "Heavy": 2,
                "Rarely": 0,
                "Sometimes": 1,
                "Often": 2
            }

            features_numeric = []
            for feature in features_json:
                value = request.form.get(feature["name"])
                if feature["type"] == "select":
                    value = ENCODING_MAP.get(value, 0)
                else:
                    try:
                        value = float(value)
                    except:
                        value = 0
                features_numeric.append(value)

            # --- Calculate BMI and add to feature list for prediction ---
            height = float(request.form.get("height") or 0)
            weight = float(request.form.get("weight") or 0)
            if height > 0:
                bmi_value = weight / ((height / 100) ** 2)
            else:
                bmi_value = 0
            features_numeric.append(bmi_value)
            features_json.append({"name": "bmi", "type": "number"})

            # --- Create DataFrame ---
            df = pd.DataFrame(np.array(features_numeric).reshape(1, -1),
                              columns=[f['name'] for f in features_json])

            prob = predict_from_df(df)  # returns float 0-1
            percent = round(prob * 100, 2)

            # Clinical rule override for high risk (Temporary workaround for demo)
            high_risk_features = [
                request.form.get("regular_period") == "No",
                request.form.get("periods_skipped") == "Yes",
                request.form.get("acne") == "Yes",
                request.form.get("facial_hair") == "Yes",
                request.form.get("hair_thinning") == "Yes",
                request.form.get("history_pcos") == "Yes",
                request.form.get("weight_gain") == "Yes",
                request.form.get("thyroid") == "Yes"
            ]
            high_risk_count = sum(high_risk_features)

            if bmi_value > 30 and high_risk_count >= 3:
                # Override percent for demo high risk
                percent = max(percent, 80.0)
                diagnosis = "High risk PCOD"
            else:
                if percent < 15:
                    diagnosis = "Low/No risk PCOD"
                elif 15 <= percent <= 55:
                    diagnosis = "Moderate/Likely to have PCOD"
                else:
                    diagnosis = "High risk PCOD"

            # Only tips in advice
            advice_dict = get_recommendations(prob)
            tips = []
            if 'advice' in advice_dict:
                tips = advice_dict['advice']

            return render_template('result.html',
                                   percent=percent,
                                   diagnosis=diagnosis,
                                   tips=tips)

        except Exception as e:
            flash(f'Prediction failed: {e}', 'danger')
            return redirect(url_for('questionnaire'))

    # GET request: render form
    return render_template('questionnaire.html', features=features_json)


# -------- Tracking --------
@app.route('/tracking', methods=['GET', 'POST'])
@login_required
def tracking():
    if request.method == 'POST':
        entry = TrackingEntry(
            user_id=current_user.id,
            date=request.form.get('date') or str(date.today()),
            step_count=int(request.form.get('step_count') or 0),
            sleep_hours=float(request.form.get('sleep') or 0.0),
            balanced_meal=int(request.form.get('balanced_meal') or 0),
            junk_food=int(request.form.get('junk_food') or 0),
            notes=request.form.get('notes') or ''
        )
        db.session.add(entry)
        db.session.commit()
        flash('Tracking entry saved.', 'success')
        return redirect(url_for('tracking'))

    entries = TrackingEntry.query.filter_by(user_id=current_user.id).order_by(TrackingEntry.date.desc()).all()
    return render_template('tracking.html', entries=entries)


# -------- Admin Dashboard --------
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Admins only', 'danger')
        return redirect(url_for('dashboard'))

    users = User.query.filter_by(is_admin=False).all()
    entries = TrackingEntry.query.order_by(TrackingEntry.date.desc()).all()
    return render_template('admin_dashboard.html', users=users, entries=entries)


# -------- Admin Export CSV --------
@app.route('/admin/export_csv')
@login_required
def admin_export_csv():
    if not current_user.is_admin:
        flash('Admins only', 'danger')
        return redirect(url_for('dashboard'))

    rows = []
    for e in TrackingEntry.query.order_by(TrackingEntry.date.desc()).all():
        user = User.query.get(e.user_id)
        rows.append([
            e.id, user.username if user else '', e.date,
            e.step_count, e.sleep_hours, e.balanced_meal,
            e.junk_food, e.notes
        ])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['entry_id', 'username', 'date', 'steps', 'sleep_hours', 'balanced_meal', 'junk_food', 'notes'])
    writer.writerows(rows)

    resp = make_response(output.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=tracking_export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
