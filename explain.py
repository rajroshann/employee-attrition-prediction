"""
explain.py
==========
Purpose: Everything related to explaining WHY the model made a prediction.

Three public functions used by app.py:
    get_shap_explanation(input_df)   → per-prediction SHAP reasons
    get_feature_importance()         → global feature importance (pre-computed)
    get_recommendations(shap_reasons)→ rule-based HR action suggestions

This file has NO Flask code. It's pure Python — testable independently.
"""

import joblib
import json
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Load artifacts once at import time
# ─────────────────────────────────────────────
# These load when app.py does `import explain` — not on every request.
# This is important: loading a 1.3MB pkl on every request would
# make your app slow. Load once, reuse forever.

# pipeline         = joblib.load("models/model.pkl")
# explainer        = joblib.load("models/shap_explainer.pkl")
# feature_names    = json.load(open("models/feature_names.json"))

# FIXED ----------------after modifying the folder structure
import os 
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
pipeline      = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
explainer     = joblib.load(os.path.join(BASE_DIR, "models", "shap_explainer.pkl"))
feature_names = json.load(open(os.path.join(BASE_DIR, "models", "feature_names.json")))

preprocessor     = pipeline.named_steps['preprocessor']
model            = pipeline.named_steps['model']


# ─────────────────────────────────────────────
# Human-readable labels for encoded feature names
# ─────────────────────────────────────────────
# After OneHotEncoding, Department becomes Department_Sales,
# Department_Research & Development, etc.
# This map converts those back to something a user can read.

FEATURE_LABELS = {
    # Numerical
    "Age"                      : "Age",
    "DailyRate"                : "Daily Rate",
    "DistanceFromHome"         : "Distance From Home",
    "Education"                : "Education Level",
    "EnvironmentSatisfaction"  : "Environment Satisfaction",
    "HourlyRate"               : "Hourly Rate",
    "JobInvolvement"           : "Job Involvement",
    "JobLevel"                 : "Job Level",
    "JobSatisfaction"          : "Job Satisfaction",
    "MonthlyIncome"            : "Monthly Income",
    "MonthlyRate"              : "Monthly Rate",
    "NumCompaniesWorked"       : "No. of Companies Worked",
    "PercentSalaryHike"        : "Salary Hike %",
    "PerformanceRating"        : "Performance Rating",
    "RelationshipSatisfaction" : "Relationship Satisfaction",
    "StockOptionLevel"         : "Stock Option Level",
    "TotalWorkingYears"        : "Total Working Years",
    "TrainingTimesLastYear"    : "Training Times Last Year",
    "WorkLifeBalance"          : "Work-Life Balance",
    "YearsAtCompany"           : "Years at Company",
    "YearsInCurrentRole"       : "Years in Current Role",
    "YearsSinceLastPromotion"  : "Years Since Last Promotion",
    "YearsWithCurrManager"     : "Years With Manager",
    # Binary
    "Gender"                   : "Gender",
    "OverTime"                 : "Overtime",
    # One-hot (partial — OHE columns contain _ separator)
    "BusinessTravel_Travel_Rarely"     : "Travel Rarely",
    "BusinessTravel_Travel_Frequently" : "Frequent Travel",
    "BusinessTravel_Non-Travel"        : "No Travel",
    "Department_Sales"                 : "Sales Dept",
    "Department_Research & Development": "R&D Dept",
    "Department_Human Resources"       : "HR Dept",
    "JobRole_Sales Executive"          : "Sales Executive Role",
    "JobRole_Research Scientist"       : "Research Scientist Role",
    "JobRole_Laboratory Technician"    : "Lab Technician Role",
    "JobRole_Manager"                  : "Manager Role",
    "JobRole_Sales Representative"     : "Sales Rep Role",
    "MaritalStatus_Single"             : "Single",
    "MaritalStatus_Married"            : "Married",
    "MaritalStatus_Divorced"           : "Divorced",
}


# ─────────────────────────────────────────────
# Recommendation rules
# ─────────────────────────────────────────────
# Maps a raw feature name → HR action recommendation.
# This is rule-based — no ML involved.
# In a real product this would be a database table.
# We keep top-level feature name (before OHE suffix) as the key
# so both "OverTime" and OHE variants like "Department_Sales" resolve.

RECOMMENDATIONS = {
    "OverTime": {
        "title"  : "Reduce Overtime Load",
        "detail" : "This employee's overtime hours are significantly contributing "
                   "to attrition risk. Consider redistributing workload, hiring "
                   "additional headcount, or offering overtime compensation and comp-off days."
    },
    "JobSatisfaction": {
        "title"  : "Address Job Satisfaction",
        "detail" : "Low job satisfaction is a key risk factor. Schedule a 1-on-1 "
                   "with the employee to understand role-specific frustrations. "
                   "Consider role enrichment, skill development opportunities, or project rotation."
    },
    "MonthlyIncome": {
        "title"  : "Review Compensation",
        "detail" : "Monthly income is below the retention threshold for this role. "
                   "Benchmark against market rates and consider a structured salary "
                   "review or performance bonus."
    },
    "EnvironmentSatisfaction": {
        "title"  : "Improve Work Environment",
        "detail" : "Low environment satisfaction suggests issues with workplace "
                   "conditions, team dynamics, or physical workspace. Conduct an "
                   "anonymous team survey to identify root causes."
    },
    "WorkLifeBalance": {
        "title"  : "Improve Work-Life Balance",
        "detail" : "Poor work-life balance is flagged as a risk driver. "
                   "Consider flexible work hours, remote work options, or "
                   "reducing non-essential meeting load."
    },
    "YearsSinceLastPromotion": {
        "title"  : "Evaluate Promotion Eligibility",
        "detail" : "It has been a significant time since last promotion. "
                   "Review performance records and assess whether a promotion "
                   "or title change is warranted to recognise contribution."
    },
    "DistanceFromHome": {
        "title"  : "Offer Remote/Hybrid Options",
        "detail" : "Long commute distance is contributing to attrition risk. "
                   "Consider offering remote or hybrid work arrangements to "
                   "reduce daily commute burden."
    },
    "NumCompaniesWorked": {
        "title"  : "Strengthen Retention Programme",
        "detail" : "Employee has worked at multiple companies, indicating a "
                   "higher baseline tendency to switch. Focus on engagement "
                   "programmes, mentorship, and clear career path communication."
    },
    "Age": {
        "title"  : "Tailor Retention to Career Stage",
        "detail" : "Age-related factors influence retention differently. "
                   "Younger employees respond well to growth opportunities; "
                   "senior employees value recognition and autonomy."
    },
    "JobLevel": {
        "title"  : "Clarify Career Progression",
        "detail" : "Job level is a contributing factor. Ensure the employee "
                   "has a visible, achievable career path with defined milestones."
    },
    "StockOptionLevel": {
        "title"  : "Review Stock Option Allocation",
        "detail" : "Low stock option level reduces long-term financial incentive "
                   "to stay. Review ESOP allocation for this role band."
    },
    "TotalWorkingYears": {
        "title"  : "Leverage Experience for Retention",
        "detail" : "Experience level suggests the employee may be looking for "
                   "senior responsibility or leadership opportunities. Explore "
                   "mentorship or team lead roles."
    },
    "BusinessTravel": {
        "title"  : "Reduce Travel Burden",
        "detail" : "Frequent business travel is a known attrition driver. "
                   "Evaluate whether travel requirements can be reduced through "
                   "virtual meetings or workload rebalancing."
    },
    "MaritalStatus": {
        "title"  : "Offer Family-Friendly Benefits",
        "detail" : "Marital status influences attrition patterns. Ensure "
                   "family-friendly policies (parental leave, flexible hours, "
                   "childcare support) are available and well-communicated."
    },
    "JobRole": {
        "title"  : "Assess Role Fit",
        "detail" : "The employee's current job role is contributing to risk. "
                   "Evaluate whether a role change or skill-based project "
                   "assignment could re-engage them."
    },
    "Department": {
        "title"  : "Investigate Department Culture",
        "detail" : "Department is a contributing factor. Review team dynamics, "
                   "manager effectiveness, and workload distribution within "
                   "this department."
    },
    # Default fallback
    "_default": {
        "title"  : "Conduct Retention Review",
        "detail" : "Schedule an employee engagement review to identify and "
                   "address the specific factors driving this attrition risk."
    }
}


def _get_feature_root(feature_name: str) -> str:
    """
    Maps an encoded feature name back to its root for recommendation lookup.
    e.g. 'Department_Sales' → 'Department'
         'OverTime'         → 'OverTime'
         'MonthlyIncome'    → 'MonthlyIncome'
    """
    for root in RECOMMENDATIONS:
        if feature_name.startswith(root):
            return root
    return "_default"


# ─────────────────────────────────────────────
# PUBLIC FUNCTION 1: Per-prediction SHAP explanation
# ─────────────────────────────────────────────

def get_shap_explanation(input_df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Given a single-row DataFrame (in the same format as training data),
    returns SHAP-based explanation of the prediction.

    Parameters:
        input_df : pd.DataFrame — one row, same columns as X_train
        top_n    : int — how many top factors to return (default 5)

    Returns:
        dict with keys:
            prediction     : int (0 or 1)
            probability    : float (0.0 – 1.0, probability of leaving)
            shap_values    : list of {feature, label, shap_value, direction}
            base_value     : float (model's average prediction)
    """

    # Step A: Get prediction and probability from full pipeline
    prediction  = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    # Step B: Transform input using just the preprocessor
    # The SHAP explainer was built on the XGBoost model directly,
    # so it needs transformed (preprocessed) data — not raw input.
    input_transformed = preprocessor.transform(input_df)

    # Step C: Compute SHAP values
    # shap_values shape: (1, n_features)
    # Each value = how much that feature pushed the prediction
    # above (+) or below (-) the base rate.
    shap_vals = explainer.shap_values(input_transformed)[0]  # first (only) row

    # Step D: Pair each SHAP value with its feature name and sort by |magnitude|
    shap_pairs = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Step E: Build output — top_n features with readable labels
    shap_output = []
    for feat_name, shap_val in shap_pairs[:top_n]:
        shap_output.append({
            "feature"    : feat_name,
            "label"      : FEATURE_LABELS.get(feat_name, feat_name),
            "shap_value" : round(float(shap_val), 4),
            "direction"  : "increases risk" if shap_val > 0 else "decreases risk"
        })

    return {
        "prediction"  : prediction,
        "probability" : round(probability * 100, 1),  # as percentage
        "base_value"  : round(float(explainer.expected_value), 4),
        "shap_values" : shap_output
    }


# ─────────────────────────────────────────────
# PUBLIC FUNCTION 2: Global Feature Importance
# ─────────────────────────────────────────────

def get_feature_importance(top_n: int = 15) -> list:
    """
    Returns global feature importance from the trained XGBoost model.
    This is NOT per-prediction — it shows which features matter most
    across the entire dataset.

    Uses model.feature_importances_ (gain-based by default in XGBoost).
    Gain = average improvement in accuracy a feature brings when used
    in a tree split. Higher = more important.

    Returns:
        list of {feature, label, importance} sorted descending
    """

    importances = model.feature_importances_

    paired = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    result = []
    for feat_name, importance in paired[:top_n]:
        result.append({
            "feature"    : feat_name,
            "label"      : FEATURE_LABELS.get(feat_name, feat_name),
            "importance" : round(float(importance), 4)
        })

    return result


# ─────────────────────────────────────────────
# PUBLIC FUNCTION 3: Recommendation Engine
# ─────────────────────────────────────────────

def get_recommendations(shap_values: list, top_n: int = 3) -> list:
    """
    Takes the SHAP output from get_shap_explanation() and returns
    HR action recommendations for the top risk-increasing features.

    Only recommends based on features that INCREASE attrition risk
    (positive SHAP values) — no point recommending action on features
    that are already helping retain the employee.

    Parameters:
        shap_values : list — output from get_shap_explanation()['shap_values']
        top_n       : int  — max recommendations to return

    Returns:
        list of {title, detail, feature} — HR actions to take
    """

    recommendations = []
    seen_roots = set()  # avoid duplicate recommendations from OHE siblings

    for item in shap_values:
        if item["shap_value"] <= 0:
            # This feature is actually helping retention — skip it
            continue

        root = _get_feature_root(item["feature"])

        if root in seen_roots:
            # Already have a recommendation for this feature group
            continue

        seen_roots.add(root)
        rec = RECOMMENDATIONS.get(root, RECOMMENDATIONS["_default"]).copy()
        rec["feature"] = item["label"]
        recommendations.append(rec)

        if len(recommendations) >= top_n:
            break

    return recommendations


# ─────────────────────────────────────────────
# Standalone test — run this file directly to verify
# python explain.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Testing explain.py with a sample employee...\n")

    # Simulate a high-risk employee
    # (Young, high overtime, low job satisfaction, low income)
    sample = pd.DataFrame([{
        'Age': 28, 'DailyRate': 500, 'DistanceFromHome': 25,
        'Education': 2, 'EnvironmentSatisfaction': 1,
        'HourlyRate': 45, 'JobInvolvement': 2, 'JobLevel': 1,
        'JobSatisfaction': 1, 'MonthlyIncome': 2500,
        'MonthlyRate': 8000, 'NumCompaniesWorked': 4,
        'PercentSalaryHike': 11, 'PerformanceRating': 3,
        'RelationshipSatisfaction': 2, 'StockOptionLevel': 0,
        'TotalWorkingYears': 3, 'TrainingTimesLastYear': 1,
        'WorkLifeBalance': 1, 'YearsAtCompany': 1,
        'YearsInCurrentRole': 0, 'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 0,
        'Gender': 1,    # Male (label-encoded)
        'OverTime': 1,  # Yes (label-encoded)
        'BusinessTravel': 'Travel_Frequently',
        'Department': 'Sales',
        'EducationField': 'Marketing',
        'JobRole': 'Sales Representative',
        'MaritalStatus': 'Single'
    }])

    # ── Test 1: SHAP Explanation ──
    result = get_shap_explanation(sample)
    print(f"Prediction  : {'LEAVE' if result['prediction'] == 1 else 'STAY'}")
    print(f"Probability : {result['probability']}% chance of leaving")
    print(f"Base value  : {result['base_value']}")
    print(f"\nTop {len(result['shap_values'])} factors:")
    for i, s in enumerate(result['shap_values'], 1):
        direction = "▲" if s['shap_value'] > 0 else "▼"
        print(f"  {i}. {direction} {s['label']:35s} SHAP={s['shap_value']:+.4f}  [{s['direction']}]")

    # ── Test 2: Recommendations ──
    recs = get_recommendations(result['shap_values'])
    print(f"\nRecommendations ({len(recs)}):")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. [{r['feature']}] {r['title']}")
        print(f"     {r['detail'][:80]}...")

    # ── Test 3: Feature Importance ──
    fi = get_feature_importance(top_n=10)
    print(f"\nTop 10 Global Feature Importances:")
    for i, f in enumerate(fi, 1):
        bar = "█" * int(f['importance'] * 200)
        print(f"  {i:2}. {f['label']:35s} {f['importance']:.4f}  {bar}")

    print("\n✓ explain.py working correctly")
