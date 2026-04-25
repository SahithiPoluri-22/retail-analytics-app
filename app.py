from flask import Flask, render_template, request
from db import query_df
import pandas as pd
import plotly.express as px
import plotly.utils
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

app = Flask(__name__)

# ── HOME / HOUSEHOLD LOOKUP ──────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    data, hshd_num, error = None, None, None
    if request.method == "POST":
        hshd_num = request.form.get("hshd_num", "").strip()
        if hshd_num:
            try:
                sql = """
                    SELECT
                        CAST(t.HSHD_NUM AS INT)     AS HSHD_NUM,
                        CAST(t.BASKET_NUM AS INT)    AS BASKET_NUM,
                        RTRIM(t.PURCHASE_)           AS PURCHASE_,
                        CAST(t.PRODUCT_NUM AS INT)   AS PRODUCT_NUM,
                        RTRIM(p.DEPARTMENT)          AS DEPARTMENT,
                        RTRIM(p.COMMODITY)           AS COMMODITY,
                        CAST(t.SPEND AS FLOAT)       AS SPEND,
                        CAST(t.UNITS AS INT)         AS UNITS,
                        RTRIM(t.STORE_R)             AS STORE_R,
                        CAST(t.WEEK_NUM AS INT)      AS WEEK_NUM,
                        CAST(t.YEAR AS INT)          AS YEAR,
                        RTRIM(h.L)                   AS LOYALTY_FLAG,
                        RTRIM(h.AGE_RANGE)           AS AGE_RANGE,
                        RTRIM(h.MARITAL)             AS MARITAL,
                        RTRIM(h.INCOME_RANGE)        AS INCOME_RANGE,
                        RTRIM(h.HOMEOWNER)           AS HOMEOWNER,
                        RTRIM(h.HSHD_COMPOSITION)    AS HSHD_COMPOSITION,
                        RTRIM(h.HH_SIZE)             AS HH_SIZE,
                        RTRIM(h.CHILDREN)            AS CHILDREN
                    FROM transactions t
                    JOIN households h ON CAST(t.HSHD_NUM AS INT) = CAST(h.HSHD_NUM AS INT)
                    JOIN products   p ON CAST(t.PRODUCT_NUM AS INT) = CAST(p.PRODUCT_NUM AS INT)
                    WHERE CAST(t.HSHD_NUM AS INT) = :hshd_num
                    ORDER BY t.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_,
                             t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY
                """
                df = query_df(sql, {"hshd_num": int(hshd_num)})
                if df.empty:
                    error = f"No data found for Household #{hshd_num}"
                else:
                    data = df.to_dict(orient="records")
                    return render_template("index.html", data=data,
                                           columns=df.columns.tolist(),
                                           hshd_num=hshd_num)
            except Exception as e:
                error = str(e)
    return render_template("index.html", data=data, hshd_num=hshd_num, error=error)


# ── DASHBOARD ────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    charts = {}

    def make_layout(fig):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            font_family="DM Sans",
            title_font_size=16,
            margin=dict(t=50, l=50, r=20, b=60),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 1. Spend by Department
    df1 = query_df("""
        SELECT RTRIM(p.DEPARTMENT) AS DEPARTMENT,
               SUM(CAST(t.SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions t
        JOIN products p ON CAST(t.PRODUCT_NUM AS INT) = CAST(p.PRODUCT_NUM AS INT)
        GROUP BY p.DEPARTMENT ORDER BY TOTAL_SPEND DESC
    """)
    charts["dept_spend"] = make_layout(px.bar(df1, x="DEPARTMENT", y="TOTAL_SPEND",
        title="Total Spend by Department", color="TOTAL_SPEND",
        color_continuous_scale="Viridis"))

    # 2. Spend over time
    df2 = query_df("""
        SELECT TRY_CAST(YEAR AS INT) AS YEAR,
               SUM(CAST(SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions
        WHERE TRY_CAST(YEAR AS INT) IS NOT NULL
          AND TRY_CAST(YEAR AS INT) > 2000
        GROUP BY TRY_CAST(YEAR AS INT)
        ORDER BY TRY_CAST(YEAR AS INT)
    """)
    df2["YEAR"] = df2["YEAR"].astype(str)   # ← string, not int
    fig2 = px.line(df2, x="YEAR", y="TOTAL_SPEND",
                   title="Total Spend Over Time", markers=True)
    fig2.update_traces(line_color="#00f5d4", line_width=3, marker_size=10)
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        font_family="DM Sans",
        title_font_size=16,
        margin=dict(t=50, l=50, r=20, b=60),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    charts["spend_time"] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Loyalty vs Non-loyalty
    df3 = query_df("""
        SELECT RTRIM(h.L) AS LOYALTY,
               SUM(CAST(t.SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions t
        JOIN households h ON CAST(t.HSHD_NUM AS INT) = CAST(h.HSHD_NUM AS INT)
        WHERE h.L IS NOT NULL
        GROUP BY h.L
    """)
    df3["LOYALTY"] = df3["LOYALTY"].str.strip().map({"Y": "Loyal", "N": "Non-Loyal"}).fillna("Unknown")
    charts["loyalty_spend"] = make_layout(px.pie(df3, names="LOYALTY", values="TOTAL_SPEND",
        title="Loyal vs Non-Loyal Spend",
        color_discrete_sequence=["#00f5d4", "#f15bb5"]))

    # 4. Spend by income range
    df4 = query_df("""
        SELECT RTRIM(h.INCOME_RANGE) AS INCOME_RANGE,
               SUM(CAST(t.SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions t
        JOIN households h ON CAST(t.HSHD_NUM AS INT) = CAST(h.HSHD_NUM AS INT)
        WHERE h.INCOME_RANGE NOT LIKE '%null%' AND h.INCOME_RANGE IS NOT NULL
        GROUP BY h.INCOME_RANGE ORDER BY TOTAL_SPEND DESC
    """)
    charts["income_spend"] = make_layout(px.bar(df4, x="INCOME_RANGE", y="TOTAL_SPEND",
        title="Spend by Income Range", color="TOTAL_SPEND",
        color_continuous_scale="Plasma"))

    # 5. Brand type
    df5 = query_df("""
        SELECT RTRIM(p.BRAND_TY) AS BRAND_TY,
               SUM(CAST(t.SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions t
        JOIN products p ON CAST(t.PRODUCT_NUM AS INT) = CAST(p.PRODUCT_NUM AS INT)
        WHERE p.BRAND_TY IS NOT NULL
        GROUP BY p.BRAND_TY
    """)
    charts["brand_spend"] = make_layout(px.pie(df5, names="BRAND_TY", values="TOTAL_SPEND",
        title="Private vs National Brand",
        color_discrete_sequence=["#fee440", "#9b5de5"]))

    # 6. Spend by region
    df6 = query_df("""
        SELECT RTRIM(STORE_R) AS STORE_R,
               SUM(CAST(SPEND AS FLOAT)) AS TOTAL_SPEND
        FROM transactions GROUP BY STORE_R ORDER BY TOTAL_SPEND DESC
    """)
    charts["region_spend"] = make_layout(px.bar(df6, x="STORE_R", y="TOTAL_SPEND",
        title="Spend by Store Region", color="TOTAL_SPEND",
        color_continuous_scale="Turbo"))

    return render_template("dashboard.html", charts=charts)


# ── ML MODEL — CLV PREDICTION ────────────────────────────────
@app.route("/ml")
def ml():
    clv_df = query_df("""
        SELECT CAST(t.HSHD_NUM AS INT) AS HSHD_NUM,
               SUM(CAST(t.SPEND AS FLOAT)) AS CLV,
               RTRIM(h.AGE_RANGE)    AS AGE_RANGE,
               RTRIM(h.INCOME_RANGE) AS INCOME_RANGE,
               RTRIM(h.L)            AS LOYALTY,
               RTRIM(h.MARITAL)      AS MARITAL,
               RTRIM(h.HH_SIZE)      AS HH_SIZE,
               RTRIM(h.CHILDREN)     AS CHILDREN
        FROM transactions t
        JOIN households h ON CAST(t.HSHD_NUM AS INT) = CAST(h.HSHD_NUM AS INT)
        GROUP BY t.HSHD_NUM, h.AGE_RANGE, h.INCOME_RANGE,
                 h.L, h.MARITAL, h.HH_SIZE, h.CHILDREN
    """)

    clv_df = clv_df.replace("null", np.nan).dropna()
    cat_cols = ["AGE_RANGE", "INCOME_RANGE", "LOYALTY", "MARITAL", "HH_SIZE", "CHILDREN"]
    le = LabelEncoder()
    for col in cat_cols:
        clv_df[col] = le.fit_transform(clv_df[col].astype(str).str.strip())

    X, y = clv_df[cat_cols], clv_df["CLV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = round(r2_score(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

    importance_df = pd.DataFrame({
        "Feature": cat_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance — CLV Prediction",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#ffffff", font_family="DM Sans")
    chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    clv_df["PREDICTED_CLV"] = model.predict(X)
    top10 = clv_df.nlargest(10, "PREDICTED_CLV")[["HSHD_NUM", "CLV", "PREDICTED_CLV"]].round(2)
    top10.columns = ["Household #", "Actual CLV ($)", "Predicted CLV ($)"]

    return render_template("ml.html", r2=r2, rmse=rmse, chart=chart,
                           top10=top10.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)