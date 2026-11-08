"""
Bakery Sales Forecasting
------------------------
Uses real sales data from a French bakery (Kaggle, Jan 2021 - Sep 2022)
to predict how many baguettes, croissants and pain au chocolat
the bakery will sell each day, so it can bake the right amount.

Steps:
  1. Load the data
  2. Clean it
  3. Explore it with SQL
  4. Make charts
  5. Train a simple forecasting model and check its accuracy
"""

import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
df = pd.read_csv("data/Bakery sales.csv")
print("Rows loaded:", len(df))

# ---------------------------------------------------------------
# 2. Clean the data
# ---------------------------------------------------------------
# Remove the extra index column that came with the file
df = df.drop(columns=["Unnamed: 0"])

# Price is text like "0,90 €" -> turn it into a number 0.90
df["unit_price"] = (df["unit_price"]
                    .str.replace("€", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.strip()
                    .astype(float))

# Negative quantities are refunds or corrections, not sales
negative_rows = (df["Quantity"] <= 0).sum()
df = df[df["Quantity"] > 0]

# Zero-price items are not real sales
df = df[df["unit_price"] > 0]

df["date"] = pd.to_datetime(df["date"])
df["revenue"] = df["Quantity"] * df["unit_price"]
print(f"Removed {negative_rows} refund/correction rows. Rows left: {len(df)}")

# ---------------------------------------------------------------
# 3. Explore with SQL
# ---------------------------------------------------------------
conn = sqlite3.connect(":memory:")
df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_sql("sales", conn, index=False)

print("\nTop 5 products by units sold:")
print(pd.read_sql("""
    SELECT article, SUM(Quantity) AS units_sold, ROUND(SUM(revenue), 0) AS revenue_eur
    FROM sales
    GROUP BY article
    ORDER BY units_sold DESC
    LIMIT 5
""", conn))

print("\nBusiest hour of the day:")
print(pd.read_sql("""
    SELECT SUBSTR(time, 1, 2) AS hour, COUNT(DISTINCT ticket_number) AS customers
    FROM sales
    GROUP BY hour
    ORDER BY customers DESC
    LIMIT 3
""", conn))

# ---------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------
products = ["TRADITIONAL BAGUETTE", "CROISSANT", "PAIN AU CHOCOLAT"]

# Daily sales table: one row per day, one column per product
daily = (df[df["article"].isin(products)]
         .groupby(["date", "article"])["Quantity"].sum()
         .unstack(fill_value=0))

monthly = daily.resample("MS").sum()
monthly.plot(figsize=(10, 5), marker="o")
plt.title("Units sold per month")
plt.ylabel("Units sold")
plt.xlabel("Month")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("charts/monthly_sales.png", dpi=120)
plt.close()

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_weekday = daily.groupby(daily.index.dayofweek).mean()
by_weekday.index = [days[i] for i in by_weekday.index]
by_weekday.plot(kind="bar", figsize=(10, 5))
plt.title("Average units sold by day of the week")
plt.ylabel("Average units sold")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("charts/sales_by_weekday.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 5. Forecasting model
# ---------------------------------------------------------------
# The model predicts a day's sales from:
#   - the day of the week (weekends are busier)
#   - the month (summer is busier for this bakery)
#   - sales on the same weekday one week earlier
split_date = "2022-08-01"   # train before August 2022, test on Aug-Sep 2022
results = []

for product in products:
    data = daily[[product]].rename(columns={product: "sales"})
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    # .shift(freq="7D") looks up the date exactly 7 days earlier
    data["sales_last_week"] = data["sales"].shift(freq="7D").reindex(data.index)
    data = data.dropna()   # first week, and days after a closed day, have no "last week" value

    X = pd.get_dummies(data[["day_of_week", "month"]].astype(str)).astype(int)
    X["sales_last_week"] = data["sales_last_week"]
    y = data["sales"]

    train = data.index < split_date
    model = LinearRegression()
    model.fit(X[train], y[train])
    predictions = model.predict(X[~train]).round()

    # Simple baseline: "same as last week"
    baseline_error = mean_absolute_error(y[~train], data["sales_last_week"][~train])
    model_error = mean_absolute_error(y[~train], predictions)

    results.append({
        "product": product,
        "avg_daily_sales": round(y[~train].mean(), 1),
        "model_error": round(model_error, 1),
        "baseline_error": round(baseline_error, 1),
        "improvement_%": round((baseline_error - model_error) / baseline_error * 100, 1),
    })

    if product == "TRADITIONAL BAGUETTE":
        plt.figure(figsize=(10, 5))
        plt.plot(y[~train].index, y[~train], label="Actual", marker="o", markersize=3)
        plt.plot(y[~train].index, predictions, label="Forecast", linestyle="--")
        plt.title("Traditional baguette: actual vs forecast (Aug-Sep 2022)")
        plt.ylabel("Units sold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("charts/forecast_vs_actual.png", dpi=120)
        plt.close()

results = pd.DataFrame(results)
print("\nModel accuracy on Aug-Sep 2022 (error = average units off per day):")
print(results.to_string(index=False))
results.to_csv("charts/model_results.csv", index=False)
