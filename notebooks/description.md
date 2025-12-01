

# Source Code Directory

This directory contains the Python implementation of an earthquake and tsunami risk dashboard with predictive modeling and interactive visualization using Dash.

## Modules

### 📦 `data_loading.py`

Data loading and basic preparation:

* `load_earthquake_data(file_eq)` - Load earthquake dataset from CSV
* `load_city_data(file_cities)` - Load world cities data
* `prepare_cities()` - Rename and select relevant columns
* `prepare_earthquake_features(df)` - Convert features to numeric and prepare target variable

### 🤖 `model_training.py`

Machine learning model training:

* `train_random_forest(X_train, y_train)` - Train Random Forest classifier
* `train_gradient_boosting(X_train, y_train)` - Train Gradient Boosting classifier
* `evaluate_models(X_test, y_test, models)` - Compute Accuracy, Classification Report, AUC for models

### 📊 `risk_analysis.py`

Risk calculation and helper functions:

* `calculate_earthquake_radius(magnitude, cdi, mmi, sig, depth, tsunami, nst=None)` - Compute affected radius
* `haversine_distance(lat1, lon1, lat2, lon2)` - Calculate distance between two coordinates
* `calculate_risk_level(distance, radius)` - Map distance to risk level
* `find_cities_at_risk(earthquake_lat, earthquake_lon, radius_km, cities_df)` - Identify cities within radius

### 📈 `visualization.py`

Visualization and plotting:

* `plot_scatter_geo(df)` - Global scatter plot of earthquakes
* `plot_heatmap(df)` - Heatmap simulation of earthquake intensity
* `plot_pred_risk_timelapse(df_pred_risk)` - Predicted cities at risk animation
* `plot_fact_risk_timelapse(df_fact_risk)` - Actual affected cities animation

### 🛠️ `dashboard.py`

Dash web application setup:

* `create_dash_app()` - Initialize Dash app layout with:

  * Global scatter plot
  * Heatmap
  * Predicted risk timelapse
  * Actual risk timelapse
  * Model metrics (Accuracy, AUC, classification report)
  * Data preview table
* `run_app()` - Run server and open browser

### 📁 `data/`

* `earthquake_data_tsunami.csv` - Raw earthquake and tsunami dataset
* `worldcities.csv` - City coordinates and population

## Usage

### As a Package

```python
from data_loading import load_earthquake_data, load_city_data, prepare_cities
from model_training import train_random_forest, train_gradient_boosting, evaluate_models
from risk_analysis import find_cities_at_risk, calculate_earthquake_radius
from dashboard import create_dash_app

df = load_earthquake_data('earthquake_data_tsunami.csv')
cities_df = load_city_data('worldcities.csv')
cities = prepare_cities(cities_df)

X_train, X_test, y_train, y_test = prepare_earthquake_features(df)

rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

metrics = evaluate_models(X_test, y_test, [rf_model, gb_model])
app = create_dash_app(df, cities, rf_model, gb_model)
```

### Run Complete Dashboard

```bash
python main.py
```

## Dependencies

All modules require packages listed in `requirements.txt`:

* pandas
* numpy
* scikit-learn
* plotly
* dash
* math
* webbrowser

## Design Principles

1. **Modularity**: Separation of data loading, modeling, risk analysis, visualization, and dashboard
2. **Interactivity**: Dash used for live, interactive visualizations
3. **Predictive Analytics**: Random Forest and Gradient Boosting models to predict tsunami risk
4. **Visualization**: Geospatial plots, heatmaps, and timelapse animations
5. **Error Handling**: File existence checks and numeric conversion

## Testing

```python
# Test data loading
df = load_earthquake_data('earthquake_data_tsunami.csv')
cities = prepare_cities(load_city_data('worldcities.csv'))
print(df.head())
print(cities.head())

# Test risk calculation
radius = calculate_earthquake_radius(6.5, 5.0, 6.0, 200, 10, 1)
cities_at_risk = find_cities_at_risk(38.3, 142.4, radius, cities)
print(cities_at_risk)
```

## Extending the Project

* To add new models: implement in `model_training.py` and update `evaluate_models()`
* To add new visualizations: add functions in `visualization.py` and call from `dashboard.py`
* To support new datasets: add preprocessing functions in `data_loading.py`

---

