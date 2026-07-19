"""
app.py
======
Purpose: Flask web server — receives browser requests, calls explain.py,
         returns HTML pages or JSON data.

Route map:
    Page routes  (return HTML):
        GET  /                    → dashboard.html  (Module 1 + 2)
        GET  /predict             → index.html      (Module 3–8)

    API routes  (return JSON — called by frontend JavaScript):
        GET  /api/dashboard-stats     → Module 1 KPI cards
        GET  /api/analytics-data      → Module 2 chart data
        GET  /api/feature-importance  → Module 5 global importance
        POST /api/predict             → Module 3 prediction
        POST /api/explain             → Module 4 SHAP explanation
        POST /api/recommend           → Module 6 recommendations
        POST /api/cost-calculator     → Module 7 cost estimate
        POST /api/download-report     → Module 8 PDF download

Run:
    python app.py
"""

import os
import json
import warnings
import io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify, send_file

# Import our explanation module — loads model + explainer once at startup
import explain

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load shared artifacts at startup
# ─────────────────────────────────────────────
# These are loaded once when the server starts, not on every request.
# Keeps response times fast.

# FIXED — always resolves relative to app.py location 

# CODE CHANGE HERE AFTER SOME TIME OR IN SECOAND 
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
pipeline         = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
feature_defaults = json.load(open(os.path.join(BASE_DIR, "models", "feature_defaults.json")))
dashboard_stats  = json.load(open(os.path.join(BASE_DIR, "reports", "dashboard_stats.json")))
model_metrics    = json.load(open(os.path.join(BASE_DIR, "reports", "model_metrics.json")))
RAW_DATA_PATH    = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")


#RAW_DATA_PATH    = "data/HR-Employee-Attrition.csv"

# import os
# BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "HR-Employee-Attrition.csv")





# Education and WorkLifeBalance use numeric codes in the dataset.
# We map them to readable labels for the dashboard charts.
EDUCATION_MAP = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor"
}

WORKLIFE_MAP = {
    1: "Bad",
    2: "Good",
    3: "Better",
    4: "Best"
}


# 
# Remove this — nothing reads it-----modify it after folder modification
SATISFACTION_MAP = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
}


# ─────────────────────────────────────────────
# Utility: Build input DataFrame for model
# ─────────────────────────────────────────────
# The prediction form sends 11 fields.
# The model needs 31 (after dropping 4 noise cols).
# This function fills the missing 20 with dataset medians/modes.
#
# Why medians and not zeros?
# Zeros are meaningless and dangerous — a Monthly Income of 0 or
# an Age of 0 would wildly skew the prediction. Medians represent
# the "average employee" — a safe neutral baseline.

def build_input_df(form_data: dict) -> pd.DataFrame:
    """
    Converts form data (11 fields) into a full feature DataFrame (31 fields)
    that the pipeline can process.

    Binary fields (Gender, OverTime) must be label-encoded here
    because we did that in train_model.py before building the pipeline.
    """

    # Start with all feature defaults (medians for numerical, modes for binary)
    row = feature_defaults.copy()

    # ── Numerical fields from form ──
    row["Age"]                     = int(form_data.get("age", row["Age"]))
    row["MonthlyIncome"]           = int(form_data.get("monthly_income", row["MonthlyIncome"]))
    row["JobSatisfaction"]         = int(form_data.get("job_satisfaction", row["JobSatisfaction"]))
    row["EnvironmentSatisfaction"] = int(form_data.get("environment_satisfaction", row["EnvironmentSatisfaction"]))
    row["WorkLifeBalance"]         = int(form_data.get("work_life_balance", row["WorkLifeBalance"]))
    row["YearsAtCompany"]          = int(form_data.get("years_at_company", row["YearsAtCompany"]))
    row["DistanceFromHome"]        = int(form_data.get("distance_from_home", row["DistanceFromHome"]))

    # ── Binary fields — must be label-encoded (same as train_model.py) ──
    # Gender:  Female → 0, Male → 1
    # OverTime: No → 0, Yes → 1
    gender_raw   = form_data.get("gender", "Male")
    overtime_raw = form_data.get("overtime", "No")
    row["Gender"]  = 1 if gender_raw == "Male" else 0
    row["OverTime"] = 1 if overtime_raw == "Yes" else 0

    # ── Categorical fields — passed as-is (ColumnTransformer handles OHE) ──
    row["Department"]  = form_data.get("department", "Research & Development")
    row["JobRole"]     = form_data.get("job_role", "Research Scientist")

    # BusinessTravel, EducationField, MaritalStatus — not in form, use mode defaults
    row["BusinessTravel"]  = "Travel_Rarely"
    row["EducationField"]  = "Life Sciences"
    row["MaritalStatus"]   = "Married"

    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Main dashboard page — Module 1 (KPIs) + Module 2 (Analytics charts)"""
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    """Prediction page — Module 3 (Form) + 4 (SHAP) + 5 (FI) + 6 (Recs) + 7 (Cost) + 8 (Report)"""
    return render_template("index.html")


# ─────────────────────────────────────────────
# API ROUTE — Module 1: Dashboard Stats
# ─────────────────────────────────────────────

@app.route("/api/dashboard-stats")
def api_dashboard_stats():
    """
    Returns pre-computed KPI stats for the dashboard cards.
    Data was computed in train_model.py and saved to reports/dashboard_stats.json.
    No computation happens here — just a file read cached at startup.
    """
    return jsonify({
        "status"           : "ok",
        "stats"            : dashboard_stats,
        "model_metrics"    : model_metrics
    })


# ─────────────────────────────────────────────
# API ROUTE — Module 2: Analytics Chart Data
# ─────────────────────────────────────────────

@app.route("/api/analytics-data")
def api_analytics_data():
    """
    Computes all chart data from the raw dataset.
    Returns a single JSON object with data for all 9 charts.

    Why compute here instead of saving to JSON in train_model.py?
    These charts are read-only views of the raw data — no ML involved.
    Computing them on-demand keeps train_model.py focused on ML only.
    We could cache this for performance, but at 1470 rows it's instant.
    """
    df = pd.read_csv(RAW_DATA_PATH)

    # Helper: attrition rate = Yes / total for a group
    def attrition_rate(group):
        return round((group == "Yes").mean() * 100, 1)

    # 1. Attrition by Department
    dept = df.groupby("Department")["Attrition"].apply(attrition_rate).reset_index()

    # 2. Attrition by Gender
    gender = df.groupby("Gender")["Attrition"].apply(attrition_rate).reset_index()

    # 3. Attrition by Job Role
    role = df.groupby("JobRole")["Attrition"].apply(attrition_rate).reset_index()

    # 4. Attrition by Education
    df["EducationLabel"] = df["Education"].map(EDUCATION_MAP)
    edu = df.groupby("EducationLabel")["Attrition"].apply(attrition_rate).reset_index()

    # 5. Attrition by Age Group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )
    age = df.groupby("AgeGroup", observed=True)["Attrition"].apply(attrition_rate).reset_index()

    # 6. Attrition by Overtime
    ot = df.groupby("OverTime")["Attrition"].apply(attrition_rate).reset_index()

    # 7. Monthly Income Distribution
    # We return mean income by attrition group — shows income gap clearly
    income_dist = df.groupby("Attrition")["MonthlyIncome"].mean().round(0).reset_index()

    # 8. Work-Life Balance Distribution
    df["WLBLabel"] = df["WorkLifeBalance"].map(WORKLIFE_MAP)
    wlb = df.groupby("WLBLabel")["Attrition"].apply(attrition_rate).reset_index()
    # sort by WLB level
    wlb_order = ["Bad", "Good", "Better", "Best"]
    wlb["WLBLabel"] = pd.Categorical(wlb["WLBLabel"], categories=wlb_order, ordered=True)
    wlb = wlb.sort_values("WLBLabel")

    # 9. Correlation with Attrition (numerical features only)
    df_num = df.copy()
    df_num["AttritionNum"] = (df_num["Attrition"] == "Yes").astype(int)
    num_cols = [
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome",
        "NumCompaniesWorked", "PercentSalaryHike",
        "RelationshipSatisfaction", "StockOptionLevel",
        "TotalWorkingYears", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    corr = (df_num[num_cols + ["AttritionNum"]]
            .corr()["AttritionNum"]
            .drop("AttritionNum")
            .sort_values()
            .round(3))

    return jsonify({
        "status": "ok",
        "charts": {
            "attrition_by_dept": {
                "labels": dept["Department"].tolist(),
                "values": dept["Attrition"].tolist()
            },
            "attrition_by_gender": {
                "labels": gender["Gender"].tolist(),
                "values": gender["Attrition"].tolist()
            },
            "attrition_by_role": {
                "labels": role["JobRole"].tolist(),
                "values": role["Attrition"].tolist()
            },
            "attrition_by_education": {
                "labels": edu["EducationLabel"].tolist(),
                "values": edu["Attrition"].tolist()
            },
            "attrition_by_age": {
                "labels": age["AgeGroup"].astype(str).tolist(),
                "values": age["Attrition"].tolist()
            },
            "attrition_by_overtime": {
                "labels": ot["OverTime"].tolist(),
                "values": ot["Attrition"].tolist()
            },
            "income_distribution": {
                "labels": income_dist["Attrition"].tolist(),
                "values": income_dist["MonthlyIncome"].tolist()
            },
            "worklife_balance": {
                "labels": wlb["WLBLabel"].astype(str).tolist(),
                "values": wlb["Attrition"].tolist()
            },
            "correlation": {
                "labels": corr.index.tolist(),
                "values": corr.values.tolist()
            }
        }
    })


# ─────────────────────────────────────────────
# API ROUTE — Module 3: Prediction
# ─────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Receives form data, builds full feature row, returns prediction.

    POST body (JSON):
        age, gender, department, job_role, monthly_income,
        overtime, job_satisfaction, environment_satisfaction,
        work_life_balance, years_at_company, distance_from_home

    Returns:
        prediction   : 0 (Stay) or 1 (Leave)
        probability  : float — % chance of leaving
        label        : "Stay" or "Leave"
        risk_level   : "Low" / "Medium" / "High"
    """
    try:
        data = request.get_json()
        input_df = build_input_df(data)

        prediction  = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1]) * 100

        # Risk level bucketing — for UI colour coding
        if probability < 30:
            risk_level = "Low"
        elif probability < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return jsonify({
            "status"      : "ok",
            "prediction"  : prediction,
            "probability" : round(probability, 1),
            "label"       : "Leave" if prediction == 1 else "Stay",
            "risk_level"  : risk_level
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# API ROUTE — Module 4: SHAP Explanation
# ─────────────────────────────────────────────

@app.route("/api/explain", methods=["POST"])
def api_explain():
    """
    Returns SHAP-based explanation for a single prediction.
    Called after /api/predict — uses the same form data.

    Returns top 5 SHAP factors with direction + magnitude.
    """
    try:
        data     = request.get_json()
        input_df = build_input_df(data)
        result   = explain.get_shap_explanation(input_df, top_n=5)

        return jsonify({
            "status" : "ok",
            "result" : result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# API ROUTE — Module 5: Feature Importance
# ─────────────────────────────────────────────

@app.route("/api/feature-importance")
def api_feature_importance():
    """
    Returns global feature importance (top 15 features).
    GET request — no input needed, computed from the trained model.
    """
    try:
        fi = explain.get_feature_importance(top_n=15)
        return jsonify({
            "status" : "ok",
            "data"   : fi
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# API ROUTE — Module 6: Recommendations
# ─────────────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Takes SHAP values from the frontend (already computed by /api/explain)
    and returns HR action recommendations.

    Why pass shap_values from frontend instead of recomputing?
    Avoids running SHAP twice for the same employee.
    The frontend stores the SHAP result from /api/explain and
    sends it directly here.
    """
    try:
        data        = request.get_json()
        shap_values = data.get("shap_values", [])
        recs        = explain.get_recommendations(shap_values, top_n=3)

        return jsonify({
            "status"          : "ok",
            "recommendations" : recs
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# API ROUTE — Module 7: Attrition Cost Calculator
# ─────────────────────────────────────────────

@app.route("/api/cost-calculator", methods=["POST"])
def api_cost_calculator():
    """
    Estimates the cost of replacing an employee who leaves.

    Formula (industry standard):
        Replacement cost = Monthly Salary × 12 × multiplier
        where multiplier = 0.5 to 2.0 depending on role level

    Research basis:
        SHRM (Society for Human Resource Management) estimates
        replacement costs at 50%–200% of annual salary depending
        on role complexity and seniority.

    POST body:
        monthly_income : int
        job_level      : int (1–5, from form or default)
        department     : str
    """
    try:
        data           = request.get_json()
        monthly_income = float(data.get("monthly_income", 5000))
        job_level      = int(data.get("job_level", 2))

        annual_salary  = monthly_income * 12

        # Multiplier increases with seniority
        # Entry level (1–2): 0.5× annual salary
        # Mid level   (3)  : 1.0× annual salary
        # Senior      (4–5): 1.5× annual salary
        multiplier_map = {1: 0.5, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}
        multiplier     = multiplier_map.get(job_level, 1.0)

        replacement_cost = annual_salary * multiplier

        # Cost breakdown — shows where the cost comes from
        breakdown = {
            "recruitment"    : round(replacement_cost * 0.30, 0),  # Job postings, agency fees
            "onboarding"     : round(replacement_cost * 0.20, 0),  # Training, orientation
            "productivity"   : round(replacement_cost * 0.35, 0),  # Lost productivity ramp-up
            "overtime_cover" : round(replacement_cost * 0.15, 0),  # Existing team overtime
        }

        return jsonify({
            "status"              : "ok",
            "monthly_income"      : monthly_income,
            "annual_salary"       : round(annual_salary, 0),
            "multiplier"          : multiplier,
            "replacement_cost"    : round(replacement_cost, 0),
            "breakdown"           : breakdown,
            "note": (f"Based on SHRM guidelines: {int(multiplier*100)}% "
                     f"of annual salary for Job Level {job_level}")
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# API ROUTE — Module 8: Download Report (PDF)
# ─────────────────────────────────────────────

@app.route("/api/download-report", methods=["POST"])
def api_download_report():
    """
    Generates a PDF report combining:
        - Employee input summary
        - Prediction result
        - SHAP top factors
        - HR recommendations
        - Cost estimate

    Uses reportlab for PDF generation.
    Returns the PDF as a file download (not rendered in browser).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)

        data         = request.get_json()
        form_data    = data.get("form_data", {})
        prediction   = data.get("prediction", {})
        shap_result  = data.get("shap_result", {})
        recs         = data.get("recommendations", [])
        cost         = data.get("cost", {})

        # Build PDF in memory
        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                   rightMargin=0.75*inch, leftMargin=0.75*inch,
                                   topMargin=0.75*inch, bottomMargin=0.75*inch)

        styles  = getSampleStyleSheet()
        story   = []

        # ── Colour palette ──
        PRIMARY    = colors.HexColor("#4F46E5")  # Indigo
        SUCCESS    = colors.HexColor("#10B981")  # Green
        DANGER     = colors.HexColor("#EF4444")  # Red
        WARNING    = colors.HexColor("#F59E0B")  # Amber
        LIGHT_GREY = colors.HexColor("#F3F4F6")
        DARK       = colors.HexColor("#111827")

        # ── Custom styles ──
        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                     fontSize=20, textColor=PRIMARY,
                                     spaceAfter=4, fontName="Helvetica-Bold")

        subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                        fontSize=10, textColor=colors.grey,
                                        spaceAfter=16)

        section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                       fontSize=13, textColor=DARK,
                                       spaceBefore=14, spaceAfter=6,
                                       fontName="Helvetica-Bold")

        body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                    fontSize=10, leading=16, textColor=DARK)

        # ── Title Block ──
        story.append(Paragraph("Employee Attrition Risk Report", title_style))
        story.append(Paragraph("Generated by HR Analytics Dashboard", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=PRIMARY, spaceAfter=16))

        # ── Section 1: Employee Summary ──
        story.append(Paragraph("1. Employee Profile", section_style))

        profile_data = [
            ["Field", "Value"],
            ["Age",                  str(form_data.get("age", "—"))],
            ["Gender",               str(form_data.get("gender", "—"))],
            ["Department",           str(form_data.get("department", "—"))],
            ["Job Role",             str(form_data.get("job_role", "—"))],
            ["Monthly Income",       f"₹ {int(form_data.get('monthly_income', 0)):,}"],
            ["Overtime",             str(form_data.get("overtime", "—"))],
            ["Job Satisfaction",     f"{form_data.get('job_satisfaction', '—')} / 4"],
            ["Environment Sat.",     f"{form_data.get('environment_satisfaction', '—')} / 4"],
            ["Work-Life Balance",    f"{form_data.get('work_life_balance', '—')} / 4"],
            ["Years at Company",     str(form_data.get("years_at_company", "—"))],
            ["Distance From Home",   f"{form_data.get('distance_from_home', '—')} km"],
        ]

        profile_table = Table(profile_data, colWidths=[2.5*inch, 4*inch])
        profile_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("BACKGROUND",   (0, 1), (-1, -1), LIGHT_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        story.append(profile_table)
        story.append(Spacer(1, 16))

        # ── Section 2: Prediction Result ──
        story.append(Paragraph("2. Prediction Result", section_style))

        label       = prediction.get("label", "—")
        probability = prediction.get("probability", 0)
        risk_level  = prediction.get("risk_level", "—")
        risk_color  = DANGER if label == "Leave" else SUCCESS

        pred_data = [
            ["Prediction", "Attrition Probability", "Risk Level"],
            [label, f"{probability}%", risk_level]
        ]
        pred_table = Table(pred_data, colWidths=[2*inch, 2.5*inch, 2*inch])
        pred_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND",    (0, 1), (-1, 1), LIGHT_GREY),
            ("TEXTCOLOR",     (0, 1), (0, 1),  risk_color),
            ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 11),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 16))

        # ── Section 3: SHAP Explanation ──
        story.append(Paragraph("3. Key Attrition Drivers (SHAP Analysis)", section_style))
        story.append(Paragraph(
            "SHAP values show how much each factor increased (+) or decreased (–) "
            "the attrition probability for this specific employee.",
            body_style
        ))
        story.append(Spacer(1, 8))

        shap_vals = shap_result.get("shap_values", [])
        if shap_vals:
            shap_data = [["Factor", "Impact Direction", "SHAP Value"]]
            for s in shap_vals:
                direction_symbol = "▲ Increases Risk" if s["shap_value"] > 0 else "▼ Decreases Risk"
                shap_data.append([
                    s.get("label", s.get("feature", "")),
                    direction_symbol,
                    f"{s['shap_value']:+.4f}"
                ])

            shap_table = Table(shap_data, colWidths=[2.8*inch, 2.2*inch, 1.5*inch])
            shap_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("FONTSIZE",      (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ]))
            story.append(shap_table)
        story.append(Spacer(1, 16))

        # ── Section 4: Recommendations ──
        story.append(Paragraph("4. HR Recommendations", section_style))
        if recs:
            for i, rec in enumerate(recs, 1):
                story.append(Paragraph(
                    f"<b>{i}. {rec.get('title', '')}</b>  "
                    f"<font color='grey'>[{rec.get('feature', '')}]</font>",
                    body_style
                ))
                story.append(Paragraph(rec.get("detail", ""), body_style))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No high-risk factors requiring immediate action.", body_style))
        story.append(Spacer(1, 8))

        # ── Section 5: Cost Estimate ──
        story.append(Paragraph("5. Estimated Replacement Cost", section_style))
        if cost:
            cost_data = [
                ["Cost Component", "Amount (₹)"],
                ["Recruitment & Hiring",  f"₹ {int(cost.get('breakdown', {}).get('recruitment', 0)):,}"],
                ["Onboarding & Training", f"₹ {int(cost.get('breakdown', {}).get('onboarding', 0)):,}"],
                ["Lost Productivity",     f"₹ {int(cost.get('breakdown', {}).get('productivity', 0)):,}"],
                ["Overtime Coverage",     f"₹ {int(cost.get('breakdown', {}).get('overtime_cover', 0)):,}"],
                ["TOTAL REPLACEMENT COST",f"₹ {int(cost.get('replacement_cost', 0)):,}"],
            ]
            cost_table = Table(cost_data, colWidths=[3.5*inch, 3*inch])
            cost_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, LIGHT_GREY]),
                ("BACKGROUND",    (0, -1), (-1, -1), DARK),
                ("TEXTCOLOR",     (0, -1), (-1, -1), colors.white),
                ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("FONTSIZE",      (0, 0), (-1, -1), 10),
                ("TOPPADDING",    (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ]))
            story.append(cost_table)
            story.append(Spacer(1, 8))
            story.append(Paragraph(
                f"<font color='grey'><i>{cost.get('note', '')}</i></font>",
                body_style
            ))

        # ── Footer ──
        story.append(Spacer(1, 24))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "This report was generated by the Employee Attrition Prediction System. "
            "Predictions are probabilistic and should be used alongside human judgement.",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey, alignment=1)
        ))

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="attrition_report.pdf"
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────
# Health check — useful for deployment on Render
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
