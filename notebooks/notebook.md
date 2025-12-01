
### 1️⃣ `data_loading.py`

```python
import os
import pandas as pd

def load_earthquake_data(file_eq='earthquake_data_tsunami.csv'):
    if not os.path.exists(file_eq):
        raise FileNotFoundError(f"Файл {file_eq} не найден")
    df = pd.read_csv(file_eq)
    return df

def load_city_data(file_cities='worldcities.csv'):
    if not os.path.exists(file_cities):
        raise FileNotFoundError(f"Файл {file_cities} не найден")
    df = pd.read_csv(file_cities)
    return df

def prepare_cities(df):
    cities = df[['city', 'lat', 'lng', 'population', 'country']].copy()
    cities.rename(columns={'lat': 'city_lat', 'lng': 'city_lng', 'city': 'name'}, inplace=True)
    return cities

def prepare_earthquake_features(df):
    df = df.copy()
    df['tsunami'] = df['tsunami'].astype(int)
    features = ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude', 'Year', 'Month']
    for c in features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    X = df[['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']].copy()
    y = df['tsunami']
    return X, y, df
```

---

### 2️⃣ `model_training.py`

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

def preprocess_data(X):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_processed = pd.DataFrame(numeric_transformer.fit_transform(X.fillna(0)), columns=X.columns)
    return X_processed

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_random_forest(X_train, y_train, n_estimators=200):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf

def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.1):
    gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    gb.fit(X_train, y_train)
    return gb

def evaluate_models(X_test, y_test, models):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        report = classification_report(y_test, y_pred, zero_division=0)
        results[name] = {"accuracy": acc, "auc": auc, "report": report}
    return results
```

---

### 3️⃣ `risk_analysis.py`

```python
import math
import numpy as np
import pandas as pd

RISK_COLORS = {
    "ОЧЕНЬ ВЫСОКИЙ": "red",
    "ВЫСОКИЙ": "orange",
    "УМЕРЕННЫЙ": "yellow",
    "НИЗКИЙ": "green"
}

def calculate_earthquake_radius(magnitude, cdi, mmi, sig, depth, tsunami, nst=None):
    base_radius = 10 ** (0.43 * magnitude - 1.2)
    depth_factor = 1.8 - (depth / 100) if depth <= 70 else 1.1 - (depth / 500)
    depth_factor = max(0.5, min(2.0, depth_factor))

    intensity = mmi if not np.isnan(mmi) else cdi
    if not np.isnan(intensity):
        if intensity >= 7: intensity_factor = 1.4
        elif intensity >= 5: intensity_factor = 1.2
        else: intensity_factor = 0.8
    else:
        intensity_factor = 1.0

    significance_factor = 1.0 + (sig / 5000)
    tsunami_factor = 1.3 if tsunami == 1 else 1.0
    data_quality_factor = 1.0
    if nst is not None and nst < 20: data_quality_factor = 0.9
    elif nst is not None and nst > 100: data_quality_factor = 1.1

    final_radius = (base_radius * depth_factor * intensity_factor *
                    significance_factor * tsunami_factor * data_quality_factor)
    return final_radius

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_risk_level(distance, radius):
    if distance <= radius * 0.3: return "ОЧЕНЬ ВЫСОКИЙ"
    elif distance <= radius * 0.6: return "ВЫСОКИЙ"
    elif distance <= radius: return "УМЕРЕННЫЙ"
    else: return "НИЗКИЙ"

def find_cities_at_risk(earthquake_lat, earthquake_lon, radius_km, cities):
    affected = []
    for _, city in cities.iterrows():
        distance = haversine_distance(
            earthquake_lat, earthquake_lon,
            city['city_lat'], city['city_lng']
        )
        if distance <= radius_km:
            risk_level = calculate_risk_level(distance, radius_km)
            affected.append({
                'city': city['name'],
                'lat': city['city_lat'],
                'lng': city['city_lng'],
                'distance_km': distance,
                'risk_level': risk_level,
                'color': RISK_COLORS[risk_level]
            })
    return pd.DataFrame(affected)
```

---

### 4️⃣ `visualization.py`

```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_scatter_geo(df):
    fig = px.scatter_geo(
        df,
        lat='latitude', lon='longitude',
        size='magnitude',
        color='magnitude',
        color_continuous_scale=['green','yellow','red'],
        projection='orthographic',
        hover_data=['magnitude','depth','tsunami'],
        title='Распределение землетрясений'
    )
    fig.update_geos(showland=True, landcolor="LightGray", showcountries=True)
    fig.update_layout(height=750, margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def plot_heatmap(df):
    size_heat = np.clip((df['magnitude'].fillna(df['magnitude'].median()) - df['magnitude'].min()) /
                        max(1e-6, df['magnitude'].max() - df['magnitude'].min()) * 20 + 10, 5, 25)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers',
        marker=dict(size=size_heat, opacity=0.5,
                    color=df['magnitude'], colorscale='YlOrRd', showscale=True, colorbar=dict(title='Magnitude')),
        hoverinfo='text',
        hovertext=df.apply(lambda r: f"mag: {r['magnitude']}<br>tsunami: {r['tsunami']}", axis=1)
    ))
    fig.update_geos(projection_type="orthographic", showland=True, landcolor="LightGray", showcountries=True)
    fig.update_layout(title="Имитация тепловой карты", height=750, margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def plot_risk_timelapse(df_risk, title, color_map):
    fig = px.scatter_geo(
        df_risk,
        lat="lat",
        lon="lng",
        color="risk_level",
        color_discrete_map=color_map,
        hover_data=["city", "distance_km", "risk_level"],
        animation_frame="time",
        title=title
    )
    fig.update_geos(projection_type="orthographic", showland=True, showcountries=True)
    fig.update_layout(height=750, margin={"r":0,"t":40,"l":0,"b":0})
    return fig
```

---

### 5️⃣ `dashboard.py`

```python
from dash import Dash, dcc, html, dash_table
import webbrowser

def create_dash_app(df, cities, rf_model, gb_model, df_pred_risk, df_fact_risk, results):
    app = Dash(__name__, suppress_callback_exceptions=True)
    server = app.server
    df_head = df.head(50)

    fig_points = df['scatter_geo'] if 'scatter_geo' in df else None
    fig_heat = df['heatmap'] if 'heatmap' in df else None

    app.layout = html.Div([
        html.H1("Earthquakes Dashboard", style={'textAlign': 'center', 'color': '#f5f5f5'}),
        html.Div([dcc.Graph(figure=fig_points)], style={'padding': '10px'}),
        html.Div([dcc.Graph(figure=fig_heat)], style={'padding': '10px'}),
        html.Div([dcc.Graph(figure=df_pred_risk)], style={'padding': '10px'}),
        html.Div([dcc.Graph(figure=df_fact_risk)], style={'padding': '10px'}),
        html.H2("Метрики моделей", style={'color': '#f5f5f5'}),
        html.Div([
            html.P(f"Random Forest: Accuracy={results['Random Forest']['accuracy']:.3f}, AUC={results['Random Forest']['auc']:.3f}"),
            html.Pre(results['Random Forest']['report'], style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#2e2e2e', 'color': '#f5f5f5'}),
            html.P(f"Gradient Boosting: Accuracy={results['Gradient Boosting']['accuracy']:.3f}, AUC={results['Gradient Boosting']['auc']:.3f}"),
            html.Pre(results['Gradient Boosting']['report'], style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#2e2e2e', 'color': '#f5f5f5'})
        ], style={'padding': '10px'}),
        html.H3("Превью данных", style={'color': '#f5f5f5'}),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df_head.columns],
            data=df_head.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#2e2e2e', 'color': '#f5f5f5'},
            style_cell={'backgroundColor': '#1e1e1e', 'color': '#f5f5f5'}
        )
    ], style={'backgroundColor': '#1e1e1e', 'color': '#f5f5f5', 'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1400px'})

    return app

def run_app(app, url="http://127.0.0.1:8050"):
    print(f"Запуск Dash приложения на {url}")
    try:
        webbrowser.open(url)
    except:
        pass
    app.run(debug=False, port=8050)
```

---

### 6️⃣ `main.py`

```python
from data_loading import load_earthquake_data, load_city_data, prepare_cities, prepare_earthquake_features
from model_training import preprocess_data, split_data, train_random_forest, train_gradient_boosting, evaluate_models
from risk_analysis import calculate_earthquake_radius, find_cities_at_risk, RISK_COLORS
from visualization import plot_scatter_geo, plot_heatmap, plot_risk_timelapse
from dashboard import create_dash_app, run_app

df_raw = load_earthquake_data()
cities_df = load_city_data()
cities = prepare_cities(cities_df)
X, y, df = prepare_earthquake_features(df_raw)

X_processed = preprocess_data(X)
X_train, X_test, y_train, y_test = split_data(X_processed, y)

rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

results = evaluate_models(X_test, y_test, {'Random Forest': rf_model, 'Gradient Boosting': gb_model})

fig_points = plot_scatter_geo(df)
fig_heat = plot_heatmap(df)

# Predicted risk timelapse
frames_pred = []
for _, row in df.iterrows():
    radius = calculate_earthquake_radius(row['magnitude'], row['cdi'], row['mmi'], row['sig'], row['depth'], row['tsunami'])
    cities_hit = find_cities_at_risk(row['latitude'], row['longitude'], radius, cities)
    if not cities_hit.empty:
        cities_hit['time'] = row['Year']
        frames_pred.append(cities_hit)
df_pred_risk = plot_risk_timelapse(pd.concat(frames_pred, ignore_index=True), "Таймлапс (ПРЕДСКАЗАНИЕ): города в зоне риска", RISK_COLORS)

# Actual risk timelapse
frames_fact = []
for _, row in df.iterrows():
    if row['tsunami'] == 1:
        radius = calculate_earthquake_radius(row['magnitude'], row['cdi'], row['mmi'], row['sig'], row['depth'], row['tsunami'])
        cities_hit = find_cities_at_risk(row['latitude'], row['longitude'], radius, cities)
        if not cities_hit.empty:
            cities_hit['time'] = row['Year']
            frames_fact.append(cities_hit)
df_fact_risk = plot_risk_timelapse(pd.concat(frames_fact, ignore_index=True), "Таймлапс (ФАКТ): города, попавшие в зону риска", RISK_COLORS)

app = create_dash_app(df, cities, rf_model, gb_model, df_pred_risk, df_fact_risk, results)

if __name__ == "__main__":
    run_app(app)
```

---


