import os
import re
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from app.xai_utils import explain_diabetes, explain_heart
from app.models.kidney_shap import explain_kidney
from app.models.liver_shap import explain_liver
from pdf_report import generate_health_report

import joblib
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
)
from flask_wtf.csrf import CSRFProtect

from werkzeug.security import check_password_hash, generate_password_hash
from auth_db import (
    initialize_database,
    create_user,
    get_user_by_email,
    get_user_by_id,
    verify_password,
    update_profile,
    change_password,
    get_all_users,
    create_password_reset_token,
    get_password_reset_token,
    mark_password_reset_token_used,
    reset_user_password,
    get_connection,
    save_prediction,
    get_user_prediction_history,
)

app = Flask(__name__)

from uuid import uuid4
from threading import Lock

# Temporary server-side storage for generated health reports.
health_reports = {}
health_reports_lock = Lock()

# Development configuration.
# Set HEALTHGUARD_SECRET_KEY to a persistent, random secret before deployment.
app.config["SECRET_KEY"] = os.environ.get(
    "HEALTHGUARD_SECRET_KEY",
    "development-only-change-this-secret-before-deployment",
)
csrf = CSRFProtect(app)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("FLASK_ENV") == "production"
)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize the authentication database
initialize_database()

def login_required(view_function):
    """Require an authenticated user before accessing a route."""

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        user_id = session.get("user_id")

        if not user_id or not get_user_by_id(user_id):
            session.clear()
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view

# Load the trained diabetes model from the saved dictionary
diabetes_bundle = joblib.load(
    BASE_DIR / "app" / "models" / "diabetes_random_forest.joblib"
)

diabetes_model = diabetes_bundle["model"]

# Load the trained heart disease model
heart_model = joblib.load(
    BASE_DIR / "app" / "models" / "heart_logistic_regression.joblib"
)

kidney_model = joblib.load(
    BASE_DIR / "app" / "models" / "kidney_logistic_regression.joblib"
)

liver_model = joblib.load(
    BASE_DIR / "app" / "models" / "liver_random_forest.joblib"
)


@app.route("/")
def home():
    return render_template("index.html")

# ---------------- AUTHENTICATION ROUTES ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or len(name) > 100:
            flash("Enter a name containing 1–100 characters.", "danger")
            return render_template("register.html")

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.", "danger")
            return render_template("register.html")

        if len(email) > 254:
            flash("Email address is too long.", "danger")
            return render_template("register.html")

        if len(password) < 8 or len(password) > 128:
            flash(
                "Password must contain between 8 and 128 characters.",
                "danger",
            )
            return render_template("register.html")

        if not any(char.isalpha() for char in password) or not any(
            char.isdigit() for char in password
        ):
            flash(
                "Password must contain at least one letter and one number.",
                "danger",
            )
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if get_user_by_email(email):
            flash("An account with this email already exists.", "danger")
            return render_template("register.html")

        user_id = create_user(name, email, password)

        if user_id is None:
            flash("Account creation failed. Please try again.", "danger")
            return render_template("register.html")

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        current_user = get_user_by_id(session["user_id"])

        if current_user and current_user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))

        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)

        if not user or not verify_password(user, password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        # Start a fresh authenticated session.
        session.clear()
        session["user_id"] = user["id"]

        flash("Login successful.", "success")

        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# ---------------- PROFILE ROUTES ----------------

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)

    if not user:
        session.clear()
        flash("Please log in again.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()

        if not name or len(name) > 100:
            flash("Enter a name containing 1–100 characters.", "danger")
            return render_template("profile.html", user=user)

        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.", "danger")
            return render_template("profile.html", user=user)

        if len(email) > 254:
            flash("Email address is too long.", "danger")
            return render_template("profile.html", user=user)

        if update_profile(user_id, name, email):
            flash("Profile updated successfully.", "success")
            return redirect(url_for("profile"))

        flash("That email address may already be registered.", "danger")
        return render_template("profile.html", user=user)

    return render_template("profile.html", user=user)


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password_page():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        user = get_user_by_email(
            get_user_by_id(session["user_id"])["email"]
        )

        if not user or not verify_password(user, current_password):
            flash("Your current password is incorrect.", "danger")
            return render_template("change_password.html")

        if len(new_password) < 8:
            flash("The new password must contain at least 8 characters.", "danger")
            return render_template("change_password.html")

        if new_password != confirm_password:
            flash("The new passwords do not match.", "danger")
            return render_template("change_password.html")

        if current_password == new_password:
            flash("Choose a password different from your current password.", "danger")
            return render_template("change_password.html")

        change_password(session["user_id"], new_password)

        # Invalidate the current login after changing credentials.
        session.clear()
        flash("Password changed successfully. Please log in again.", "success")
        return redirect(url_for("login"))

    return render_template("change_password.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = get_user_by_email(email)

        # Use the same message whether or not the account exists.
        flash(
            "If an account exists for that email, a reset link "
            "will be generated for it.",
            "info",
        )

        if user:
            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(
                raw_token.encode("utf-8")
            ).hexdigest()

            expires_at = (
                datetime.now(timezone.utc) + timedelta(minutes=15)
            ).isoformat()

            create_password_reset_token(
                user["id"],
                token_hash,
                expires_at,
            )

            # LOCAL DEVELOPMENT ONLY:
            # Print the link in the Flask terminal instead of emailing it.
            reset_url = url_for(
                "reset_password",
                token=raw_token,
                _external=True,
            )
            app.logger.info("Password reset link: %s", reset_url)

        return redirect(url_for("forgot_password"))

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    reset_record = get_password_reset_token(token_hash)

    if not reset_record or reset_record["used"]:
        flash("This password reset link is invalid or has already been used.", "danger")
        return redirect(url_for("forgot_password"))

    try:
        expires_at = datetime.fromisoformat(reset_record["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        flash("This password reset link is invalid.", "danger")
        return redirect(url_for("forgot_password"))

    if datetime.now(timezone.utc) >= expires_at:
        flash("This password reset link has expired. Please request another.", "warning")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            flash("Password must contain at least 8 characters, including a letter and a number.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            # Recheck and consume the token atomically to prevent reuse.
            with get_connection() as connection:
                current_token = connection.execute(
                    """
                    SELECT id, user_id, expires_at, used
                    FROM password_reset_tokens
                    WHERE id = ? AND used = 0
                    """,
                    (reset_record["id"],),
                ).fetchone()

                if not current_token:
                    flash("This reset link has already been used.", "danger")
                    return redirect(url_for("forgot_password"))

                try:
                    current_expiry = datetime.fromisoformat(
                        current_token["expires_at"]
                    )
                    if current_expiry.tzinfo is None:
                        current_expiry = current_expiry.replace(
                            tzinfo=timezone.utc
                        )
                except (ValueError, TypeError):
                    flash("This reset link is invalid.", "danger")
                    return redirect(url_for("forgot_password"))

                if datetime.now(timezone.utc) >= current_expiry:
                    flash("This reset link has expired.", "warning")
                    return redirect(url_for("forgot_password"))

                password_hash = generate_password_hash(password)

                connection.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (password_hash, current_token["user_id"]),
                )
                connection.execute(
                    "UPDATE password_reset_tokens SET used = 1 WHERE id = ?",
                    (current_token["id"],),
                )

                # Invalidate any other outstanding reset links for this account.
                connection.execute(
                    """
                    UPDATE password_reset_tokens
                    SET used = 1
                    WHERE user_id = ? AND used = 0
                    """,
                    (current_token["user_id"],),
                )

            flash("Your password has been reset. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)

@app.route("/admin")
@login_required
def admin_dashboard():
    user = get_user_by_id(session["user_id"])

    if not user or user["role"] != "admin":
        flash("Administrator access is required.", "danger")
        return redirect(url_for("dashboard"))

    users = get_all_users()

    return render_template(
        "admin_dashboard.html",
        users=users,
    )
# ---------------------------------------------------------
# MODEL PERFORMANCE DASHBOARD
# ---------------------------------------------------------

MODEL_METRIC_FILES = {
    "Diabetes": {
        "comparison": "diabetes_model_comparison.csv",
        "cross_validation": "diabetes_cross_validation.csv",
    },
    "Heart Disease": {
        "comparison": "heart_model_comparison.csv",
        "cross_validation": "heart_cross_validation.csv",
    },
    "Kidney Disease": {
        "comparison": "kidney_model_comparison.csv",
        "cross_validation": "kidney_cross_validation.csv",
    },
    "Liver Disease": {
        "comparison": "liver_model_comparison.csv",
        "cross_validation": None,
    },
}


def normalize_metric_name(name):
    """Normalize metric column names across evaluation CSV files."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def load_model_metrics(csv_path):
    """Load and normalize model metrics from a CSV file."""
    if not os.path.isfile(csv_path):
        return []

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        record = {}

        for column in df.columns:
            normalized = normalize_metric_name(column)
            value = row[column]

            if normalized == "model":
                record["model"] = str(value)

            elif normalized in {
                "accuracy", "accuracymean",
            }:
                record["accuracy"] = float(value)

            elif normalized in {
                "precision", "precisionmean",
            }:
                record["precision"] = float(value)

            elif normalized in {
                "recall", "recallmean",
            }:
                record["recall"] = float(value)

            elif normalized in {
                "f1score", "f1", "f1mean",
            }:
                record["f1"] = float(value)

            elif normalized in {
                "rocauc", "rocaucmean",
            }:
                record["roc_auc"] = float(value)

            elif normalized in {
                "accuracystd",
            }:
                record["accuracy_std"] = float(value)

            elif normalized in {
                "precisionstd",
            }:
                record["precision_std"] = float(value)

            elif normalized in {
                "recallstd",
            }:
                record["recall_std"] = float(value)

            elif normalized in {
                "f1std",
            }:
                record["f1_std"] = float(value)

            elif normalized in {
                "rocaucstd",
            }:
                record["roc_auc_std"] = float(value)

        if record.get("model"):
            records.append(record)

    return records


@app.route("/admin/model-performance")
@login_required
def model_performance():
    # Restrict the page to administrators.
    current_user = get_user_by_id(session["user_id"])

    if not current_user or current_user["role"] != "admin":
        flash("Administrator access is required.", "danger")
        return redirect(url_for("dashboard"))

    base_dir = os.path.join(
        app.root_path,
        "app",
        "models",
    )

    performance_data = {}

    for disease, filenames in MODEL_METRIC_FILES.items():
        comparison_path = os.path.join(
            base_dir,
            filenames["comparison"],
        )

        comparison_metrics = load_model_metrics(comparison_path)

        cross_validation_metrics = []

        if filenames["cross_validation"]:
            cv_path = os.path.join(
                base_dir,
                filenames["cross_validation"],
            )

            cross_validation_metrics = load_model_metrics(cv_path)

        performance_data[disease] = {
            "comparison": comparison_metrics,
            "cross_validation": cross_validation_metrics,
            "evaluation_type": (
                "Cross-validation mean"
                if disease == "Liver Disease"
                else "Evaluation results"
            ),
        }

    return render_template(
        "model_performance.html",
        performance_data=performance_data,
    )   

@app.route("/dashboard")
@login_required
def dashboard():
    diseases = [
        {
            "name": "Diabetes Prediction",
            "description": "Estimate diabetes risk using health measurements and machine learning.",
            "available": True,
            "url": "/predict/diabetes",
        },
        {
            "name": "Heart Disease Prediction",
            "description": "Assess heart disease risk using health indicators and machine learning.",
            "available": True,
            "url": "/predict/heart",
        },
        {
            "name": "Kidney Disease Prediction",
            "description": "Assess chronic kidney disease risk using clinical health indicators.",
            "available": True,
            "url": "/predict/kidney",
        },
        {
        "name": "Liver Disease Prediction",
        "description": "Analyze liver-related laboratory values using machine learning.",
        "available": True,
        "url": "/predict/liver",
        },
        {
            "name": "Hypertension Assessment",
            "description": "Explore blood pressure readings and hypertension risk indicators.",
            "available": True,
            "url": "/predict/hypertension",
        },
    ]

    user = get_user_by_id(session["user_id"])

    return render_template(
        "dashboard.html",
        diseases=diseases,
        user=user
    )

@app.route("/history")
@login_required
def prediction_history():
    user_id = session["user_id"]

    history = get_user_prediction_history(user_id)

    return render_template(
        "history.html",
        history=history,
    )

@app.route("/predict/diabetes", methods=["GET", "POST"])
@login_required
def predict_diabetes():
    result = None
    error = None
    xai_result = None
    report_id = None

    fields = [
        ("Pregnancies", "pregnancies", "0", "0"),
        ("Glucose level", "glucose", "120", "1"),
        ("Blood pressure", "blood_pressure", "70", "1"),
        ("Skin thickness", "skin_thickness", "20", "1"),
        ("Insulin level", "insulin", "80", "1"),
        ("BMI", "bmi", "25", "0.1"),
        ("Diabetes pedigree function", "pedigree", "0.5", "0.01"),
        ("Age", "age", "30", "1"),
    ]

    if request.method == "POST":
        try:
            values = {
                "Pregnancies": float(request.form["pregnancies"]),
                "Glucose": float(request.form["glucose"]),
                "BloodPressure": float(request.form["blood_pressure"]),
                "SkinThickness": float(request.form["skin_thickness"]),
                "Insulin": float(request.form["insulin"]),
                "BMI": float(request.form["bmi"]),
                "DiabetesPedigreeFunction": float(request.form["pedigree"]),
                "Age": float(request.form["age"]),
            }

            if any(value < 0 for value in values.values()):
                raise ValueError("Values must not be negative.")

            if (
                values["Glucose"] <= 0
                or values["BMI"] <= 0
                or values["Age"] <= 0
            ):
                raise ValueError(
                    "Glucose, BMI, and age must be greater than zero."
                )

            input_data = pd.DataFrame([values])
        

            prediction = int(diabetes_model.predict(input_data)[0])

            # Generate an Explainable AI explanation for this assessment.
            xai_result = explain_diabetes(values)

            result = (
                "The model classified this input in the higher-risk category."
                if prediction == 1
                else "The model classified this input in the lower-risk category."
            )
            save_prediction(
                session["user_id"],
                "Diabetes",
                result,
            )

            # Prepare a server-side PDF report after a successful prediction.
            report_id = str(uuid4())

            user = get_user_by_id(session["user_id"])
            
            shap_explanation = []

            if isinstance(xai_result, dict):
                shap_explanation = [
                    f"{feature}: {details}"
                    for feature, details in xai_result.items()
                ]
            elif isinstance(xai_result, (list, tuple)):
                shap_explanation = [str(item) for item in xai_result]
            elif xai_result is not None:
                shap_explanation = [str(xai_result)]

            with health_reports_lock:
                health_reports[report_id] = {
                    "user_id": session["user_id"],
                    "disease": "Diabetes",
                    "prediction": result,
                    "inputs": values.copy(),
                    "patient_name": user["name"] if user else "Not provided",
                    "shap_explanation": shap_explanation,
                }

        except (ValueError, KeyError, TypeError) as exc:
            app.logger.warning("Diabetes input validation failed: %s", exc)
            error = "Please enter valid numeric values in all eight fields."

    return render_template(
    "diabetes.html",
    fields=fields,
    result=result,
    error=error,
    xai_result=xai_result if error is None else None,
    report_id=report_id if error is None else None,
)

@app.route("/predict/heart", methods=["GET", "POST"])
@login_required
def predict_heart():
    fields = [
        {"name": "age", "label": "Age", "type": "number", "value": "45", "step": "1"},
        {"name": "sex", "label": "Sex", "type": "select", "options": [("0", "Female"), ("1", "Male")]},
        {"name": "cp", "label": "Chest pain type", "type": "select", "options": [
            ("1", "Typical angina"),
            ("2", "Atypical angina"),
            ("3", "Non-anginal pain"),
            ("4", "Asymptomatic"),
        ]},
        {"name": "trestbps", "label": "Resting blood pressure (mm Hg)", "type": "number", "value": "130", "step": "1"},
        {"name": "chol", "label": "Serum cholesterol (mg/dL)", "type": "number", "value": "220", "step": "1"},
        {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dL", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
        {"name": "restecg", "label": "Resting ECG result", "type": "select", "options": [
            ("0", "Normal"),
            ("1", "ST-T wave abnormality"),
            ("2", "Left ventricular hypertrophy"),
        ]},
        {"name": "thalach", "label": "Maximum heart rate achieved", "type": "number", "value": "150", "step": "1"},
        {"name": "exang", "label": "Exercise-induced angina", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
        {"name": "oldpeak", "label": "ST depression (oldpeak)", "type": "number", "value": "1.0", "step": "0.1"},
        {"name": "slope", "label": "Slope of peak exercise ST segment", "type": "select", "options": [
            ("1", "Upsloping"),
            ("2", "Flat"),
            ("3", "Downsloping"),
        ]},
        {"name": "ca", "label": "Number of major vessels (0–3)", "type": "select", "options": [
            ("0", "0"),
            ("1", "1"),
            ("2", "2"),
            ("3", "3"),
        ]},
        {"name": "thal", "label": "Thalassemia category", "type": "select", "options": [
            ("3", "Normal"),
            ("6", "Fixed defect"),
            ("7", "Reversible defect"),
        ]},
    ]

    result = None
    error = None
    xai_result = None
    report_id = None

    if request.method == "POST":
        try:
            values = {}
            for field in fields:
                raw_value = request.form.get(field["name"], "").strip()
                if not raw_value:
                    raise ValueError(f"Please enter or select {field['label']}.")

                values[field["name"]] = float(raw_value)

            if not 1 <= values["age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")
            if not 50 <= values["trestbps"] <= 300:
                raise ValueError("Check the resting blood pressure value.")
            if not 50 <= values["chol"] <= 700:
                raise ValueError("Check the cholesterol value.")
            if not 50 <= values["thalach"] <= 250:
                raise ValueError("Check the maximum heart rate value.")
            if not 0 <= values["oldpeak"] <= 10:
                raise ValueError("ST depression must be between 0 and 10.")

            input_data = pd.DataFrame([values], columns=[
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak",
                "slope", "ca", "thal"
            ])

            prediction = int(heart_model.predict(input_data)[0])
            # Generate an Explainable AI explanation for this assessment.
            xai_result = explain_heart(values)

            if prediction == 1:
                result = (
                    "The model classified this input in the higher-risk "
                    "category for heart disease."
                )
            else:
                result = (
                    "The model classified this input in the lower-risk "
                    "category for heart disease."
                )
            save_prediction(
                session["user_id"],
                "Heart Disease",
                result,
            )
            # Prepare the PDF report after a successful prediction.
            report_id = str(uuid4())

            user = get_user_by_id(session["user_id"])

            patient_name = "Not provided"

            if user:
                # sqlite3.Row supports column-name indexing, not .get()
                available_columns = user.keys()

                for column in ("full_name", "name", "username"):
                    if column in available_columns and user[column]:
                        patient_name = str(user[column])
                        break

            # Convert the SHAP explanation into report-friendly text.
            shap_explanation = None

            if xai_result:
                if isinstance(xai_result, dict):
                    shap_explanation = [
                        f"{key}: {value}"
                        for key, value in xai_result.items()
                    ]
                elif isinstance(xai_result, (list, tuple)):
                    shap_explanation = [
                        str(item) for item in xai_result
                    ]
                else:
                    shap_explanation = [str(xai_result)]

            # Store the report data for the authenticated download route.
            with health_reports_lock:
                health_reports[report_id] = {
                    "user_id": session["user_id"],
                    "disease": "Heart Disease",
                    "prediction": result,
                    "inputs": values.copy(),
                    "patient_name": patient_name,
                    "shap_explanation": shap_explanation,
                }   

        except (ValueError, TypeError) as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Heart disease prediction failed")
            error = (
                "The prediction could not be completed. "
                "Check the application terminal for details."
            )

    return render_template(
        "heart.html",
        fields=fields,
        result=result,
        error=error,
        xai_result=xai_result,
        report_id=report_id if error is None else None,

    )

@app.route("/predict/kidney", methods=["GET", "POST"])
@login_required
def predict_kidney():
    result = None
    error = None
    xai_result = None
    shap_result = None
    report_id = None

    numeric_features = [
        "age", "bp", "bgr", "bu", "sc",
        "sod", "pot", "hemo", "pcv", "wc", "rc"
    ]

    categorical_features = [
        "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "htn", "dm", "cad", "appet", "pe", "ane"
    ]

    feature_names = numeric_features + categorical_features

    if request.method == "POST":
        try:
            values = {}

            # Read numerical inputs.
            for field in numeric_features:
                raw_value = request.form.get(field, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter a value for {field}."
                    )

                values[field] = float(raw_value)

                if not pd.notna(values[field]):
                    raise ValueError(
                        f"Please enter a valid number for {field}."
                    )

            # Read categorical inputs.
            for field in categorical_features:
                raw_value = request.form.get(field, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please select a value for {field}."
                    )

                values[field] = raw_value

            # Basic input validation.
            if not 1 <= values["age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")

            for field in numeric_features:
                if values[field] < 0:
                    raise ValueError(
                        f"{field} cannot be negative."
                    )

            # Build the DataFrame in the exact training feature order.
            input_data = pd.DataFrame(
                [[values[field] for field in feature_names]],
                columns=feature_names
            )

            # Run the trained pipeline.
            prediction = int(kidney_model.predict(input_data)[0])
            shap_result = explain_kidney(values)

            if prediction == 1:
                result = (
                    "The model classified this input in the "
                    "CKD category. This is an educational model "
                    "output, not a medical diagnosis."
                )
            else:
                result = (
                    "The model classified this input in the "
                    "Not CKD category. This is an educational "
                    "model output, not a medical diagnosis."
                )
            save_prediction(
                session["user_id"],
                "Kidney Disease",
                result,
            )
            # Prepare the PDF report after a successful prediction.
            report_id = str(uuid4())

            user = get_user_by_id(session["user_id"])

            patient_name = "Not provided"

            if user:
                available_columns = user.keys()

                for column in ("full_name", "name", "username"):
                    if column in available_columns and user[column]:
                        patient_name = str(user[column])
                        break

            # Prepare the SHAP explanation for the PDF.
            shap_explanation = None

            if shap_result:
                if isinstance(shap_result, dict):
                    shap_explanation = [
                        f"{key}: {value}"
                        for key, value in shap_result.items()
                    ]
                elif isinstance(shap_result, (list, tuple)):
                    shap_explanation = [
                        str(item) for item in shap_result
                    ]
                else:
                    shap_explanation = [str(shap_result)]

            # Store the report data for the authenticated download route.
            with health_reports_lock:
                health_reports[report_id] = {
                    "user_id": session["user_id"],
                    "disease": "Kidney Disease",
                    "prediction": result,
                    "inputs": values.copy(),
                    "patient_name": patient_name,
                    "shap_explanation": shap_explanation,
                }

        except (ValueError, TypeError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Kidney disease prediction failed")
            error = (
                "The prediction could not be completed. "
                "Check the application terminal for details."
            )

    return render_template(
        "kidney.html",
        result=result,
        error=error,
        shap_result=shap_result,
        report_id=report_id if error is None else None,
    )

@app.route("/predict/liver", methods=["GET", "POST"])
@login_required
def predict_liver():
    result = None
    error = None
    shap_result = None
    top_features = None
    feature_values = None
    probability = None
    report_id = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"]

            feature_values = {
                "Age": age,
                "Gender": gender,
                "TB": float(request.form["tb"]),
                "DB": float(request.form["db"]),
                "Alkphos": float(request.form["alkphos"]),
                "Sgpt": float(request.form["sgpt"]),
                "Sgot": float(request.form["sgot"]),
                "TP": float(request.form["tp"]),
                "ALB": float(request.form["alb"]),
                "A/G Ratio": float(request.form["ag_ratio"]),
            }

            # Validate age.
            if age < 1 or age > 120:
                raise ValueError("Age must be between 1 and 120.")

            # Validate numerical inputs.
            numeric_values = [
                value
                for key, value in feature_values.items()
                if key != "Gender"
            ]

            if any(
                not pd.notna(value)
                or not __import__("math").isfinite(value)
                or value < 0
                for value in numeric_values
            ):
                raise ValueError(
                    "Numeric values must be finite and non-negative."
                )

            if gender.strip().lower() not in {
                "male", "female", "m", "f"
            }:
                raise ValueError("Please select a valid gender.")

            # Prepare model input.
            input_df = pd.DataFrame([feature_values])

            # Generate prediction.
            prediction = int(liver_model.predict(input_df)[0])

            if prediction == 1:
                result = (
                    "The model classified this input in the "
                    "liver disease category."
                )
            else:
                result = (
                    "The model classified this input in the "
                    "no-liver-disease category."
                )

            # Generate SHAP explanation.
            shap_result = explain_liver(feature_values)
            top_features = shap_result["features"][:5]
            probability = shap_result["probability"] * 100

            # Save prediction history.
            save_prediction(
                session["user_id"],
                "Liver Disease",
                result,
            )

            # Retrieve the logged-in user's name.
            user = get_user_by_id(session["user_id"])
            patient_name = "Not provided"

            if user:
                available_columns = user.keys()

                for column in ("full_name", "name", "username"):
                    if column in available_columns and user[column]:
                        patient_name = str(user[column])
                        break

            # Prepare the SHAP explanation for the PDF.
            shap_explanation = None

            if isinstance(shap_result, dict):
                shap_explanation = [
                    f"Estimated probability for class 1: "
                    f"{probability:.2f}%"
                ]

                for feature in top_features:
                    if isinstance(feature, dict):
                        shap_explanation.append(
                            ", ".join(
                                f"{key}: {value}"
                                for key, value in feature.items()
                            )
                        )
                    else:
                        shap_explanation.append(str(feature))

            # Store report data for the authenticated download route.
            report_id = str(uuid4())

            with health_reports_lock:
                health_reports[report_id] = {
                    "user_id": session["user_id"],
                    "disease": "Liver Disease",
                    "prediction": result,
                    "inputs": feature_values.copy(),
                    "patient_name": patient_name,
                    "shap_explanation": shap_explanation,
                }

        except (ValueError, KeyError) as exc:
            error = f"Please check your inputs: {exc}"

        except Exception:
            app.logger.exception(
                "Error generating liver prediction, SHAP explanation, "
                "or PDF report data."
            )

            error = (
                "The prediction or explanation could not be generated. "
                "Please check the application terminal."
            )

    return render_template(
        "liver.html",
        result=result,
        error=error,
        probability=probability,
        shap_result=shap_result,
        top_features=top_features,
        feature_values=feature_values,
        report_id=report_id if error is None else None,
    )

    return render_template("liver.html")

@app.route("/predict/hypertension", methods=["GET", "POST"])
@login_required
def predict_hypertension():
    result = None
    error = None
    report_id = None

    if request.method == "POST":
        try:
            systolic = int(request.form["systolic"])
            diastolic = int(request.form["diastolic"])

            if not (1 <= systolic <= 300):
                raise ValueError(
                    "Systolic pressure must be between 1 and 300."
                )

            if not (1 <= diastolic <= 200):
                raise ValueError(
                    "Diastolic pressure must be between 1 and 200."
                )

            if systolic > 180 or diastolic > 120:
                result = (
                    "This reading is in a severely elevated range. "
                    "Repeat the measurement after at least one minute. "
                    "If it remains this high, contact a healthcare professional "
                    "immediately. If you have chest pain, shortness of breath, "
                    "weakness, numbness, vision changes, or difficulty speaking, "
                    "seek emergency medical care."
                )

            elif systolic < 90 or diastolic < 60:
                result = (
                    "This reading is below the usual adult blood-pressure range. "
                    "Interpret it alongside symptoms and seek medical advice if concerned."
                )

            elif systolic >= 140 or diastolic >= 90:
                result = (
                    "This reading is in a high blood-pressure range. "
                    "Repeat measurements and consult a healthcare professional."
                )

            elif systolic >= 130 or diastolic >= 80:
                result = (
                    "This reading falls in the 130–139 systolic or 80–89 diastolic "
                    "range. Repeated measurements and professional assessment are needed."
                )

            elif systolic >= 120 and diastolic < 80:
                result = (
                    "This reading falls in the elevated systolic range. "
                    "Continue monitoring and discuss repeated elevated readings "
                    "with a healthcare professional."
                )

            else:
                result = (
                    "This reading is within the usual normal adult blood-pressure "
                    "range. One reading alone cannot establish overall blood-pressure status."
                )

            # Save prediction history.
            save_prediction(
                session["user_id"],
                "Hypertension",
                result,
            )

            # Retrieve the logged-in user's name.
            user = get_user_by_id(session["user_id"])
            patient_name = "Not provided"

            if user:
                available_columns = user.keys()

                for column in ("full_name", "name", "username"):
                    if column in available_columns and user[column]:
                        patient_name = str(user[column])
                        break

            # Store the assessment data for the PDF report.
            report_id = str(uuid4())

            with health_reports_lock:
                health_reports[report_id] = {
                    "user_id": session["user_id"],
                    "disease": "Hypertension Assessment",
                    "prediction": result,
                    "inputs": {
                        "Systolic blood pressure (mm Hg)": systolic,
                        "Diastolic blood pressure (mm Hg)": diastolic,
                    },
                    "patient_name": patient_name,
                    "shap_explanation": None,
                }

        except (ValueError, KeyError) as exc:
            error = f"Please check your readings: {exc}"

        except Exception:
            app.logger.exception(
                "Hypertension assessment or report preparation failed."
            )

            error = (
                "The assessment could not be completed. "
                "Please check the application terminal."
            )

    return render_template(
        "hypertension.html",
        result=result,
        error=error,
        report_id=report_id if error is None else None,
    )

@app.route("/download-health-report/<report_id>")
@login_required
def download_health_report(report_id):
    user_id = session["user_id"]

    # Retrieve only a report belonging to the logged-in user.
    with health_reports_lock:
        report_data = health_reports.get(report_id)

        if not report_data or report_data["user_id"] != user_id:
            flash("Report not found or access denied.", "danger")
            return redirect(url_for("dashboard"))

        report_data = dict(report_data)

    pdf_buffer = generate_health_report(
        disease=report_data["disease"],
        prediction=report_data["prediction"],
        inputs=report_data["inputs"],
        patient_name=report_data["patient_name"],
        shap_explanation=report_data.get("shap_explanation"),
    )

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"HealthGuard_{report_data['disease'].replace(' ', '_')}_Report.pdf",
    )

if __name__ == "__main__":
    app.run(debug=True)