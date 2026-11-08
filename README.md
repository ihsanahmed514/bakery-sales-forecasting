# Bakery Sales Forecasting

A beginner data project using **real sales data from a French bakery** to predict how many traditional baguettes, croissants and pain au chocolat it will sell each day, so it can bake the right amount and waste less.

Built with **Python (pandas, scikit-learn, matplotlib)** and **SQL (SQLite)**.

## Data

[French bakery daily sales](https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales) by Matthieu Gimbert on Kaggle: 234,005 sales lines from January 2021 to September 2022.

The data file is not included in this repository. Download it from Kaggle and save it as `data/Bakery sales.csv`.

## What the project does

1. **Cleans** the data: removes an extra index column, converts prices from text like `"0,90 €"` into numbers, and removes 1,295 refund or correction rows with negative quantities.
2. **Explores** it with SQL: best-selling products and the busiest hours.
3. **Visualises** monthly and weekday sales patterns.
4. **Forecasts** daily sales with a Linear Regression model using the day of the week, the month and last week's sales.

## Results

Trained on January 2021 – July 2022 and tested on August – September 2022, compared with the simple guess "the same as last week":

| Product | Avg daily sales | Model error (units/day) | Baseline error | Improvement |
|---------|----------------:|------------------------:|---------------:|------------:|
| Traditional baguette | 294.1 | 58.2 | 69.8 | 17% |
| Croissant | 72.5 | 19.7 | 25.3 | 22% |
| Pain au chocolat | 59.9 | 15.0 | 20.6 | 27% |

The model is **17–27% more accurate** than the simple guess.

![Forecast vs actual](charts/forecast_vs_actual.png)

## Key findings

- The traditional baguette is by far the best seller, with over 118,000 sold.
- 11am is the busiest hour, followed by 10am and 12pm.
- Sales are much higher in summer, and weekends are busier than weekdays.
- The bakery is often closed on Wednesdays.
- The model was too high in late August, when the summer rush ended suddenly. Adding school holiday dates would likely fix this.

![Monthly sales](charts/monthly_sales.png)
![Sales by weekday](charts/sales_by_weekday.png)

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Author

Ihsan Ahmed – Software Engineering graduate interested in data and AI.
[LinkedIn](https://linkedin.com/in/balochihsan1996)
