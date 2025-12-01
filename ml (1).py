import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
import webbrowser

# ---------------------- 0. Файлы ----------------------
file_eq = 'earthquake_data_tsunami.csv'
file_cities = 'worldcities.csv'

if not os.path.exists(file_eq) or not os.path.exists(file_cities):
    raise FileNotFoundError("Убедитесь, что файлы earthquake_data_tsunami.csv и worldcities.csv находятся в текущей папке")

# ---------------------- 1. Загрузка данных ----------------------
df_raw = pd.read_csv(file_eq)
df = df_raw.copy()

cities_df = pd.read_csv(file_cities)
cities = cities_df[['city', 'lat', 'lng', 'population', 'country']].copy()
cities.rename(columns={'lat': 'city_lat', 'lng': 'city_lng', 'city': 'name'}, inplace=True)

# ---------------------- 2. Подготовка целевой переменной ----------------------
df['tsunami'] = df['tsunami'].astype(int)
features = ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude', 'Year', 'Month']

for c in features:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------------------- 3. Обучение моделей ----------------------
X = df[['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']].copy()
y = df['tsunami']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_processed = pd.DataFrame(numeric_transformer.fit_transform(X.fillna(0)), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2,
    random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Метрики
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_gb = accuracy_score(y_test, y_pred_gb)
report_rf = classification_report(y_test, y_pred_rf, zero_division=0)
report_gb = classification_report(y_test, y_pred_gb, zero_division=0)
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
auc_gb = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

# ---------------------- 4. Вспомогательные функции для радиуса и риска ----------------------
def calculate_earthquake_radius(magnitude, cdi, mmi, sig, depth, tsunami, nst=None):
    base_radius = 10 ** (0.43 * magnitude - 1.2)
    depth_factor = 1.8 - (depth / 100) if depth <= 70 else 1.1 - (depth / 500)
    depth_factor = max(0.5, min(2.0, depth_factor))

    intensity = mmi if not np.isnan(mmi) else cdi
    if not np.isnan(intensity):
        if intensity >= 7:
            intensity_factor = 1.4
        elif intensity >= 5:
            intensity_factor = 1.2
        else:
            intensity_factor = 0.8
    else:
        intensity_factor = 1.0

    significance_factor = 1.0 + (sig / 5000)
    tsunami_factor = 1.3 if tsunami == 1 else 1.0
    data_quality_factor = 1.0
    if nst is not None and nst < 20:
        data_quality_factor = 0.9
    elif nst is not None and nst > 100:
        data_quality_factor = 1.1

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
    if distance <= radius * 0.3:
        return "ОЧЕНЬ ВЫСОКИЙ"
    elif distance <= radius * 0.6:
        return "ВЫСОКИЙ"
    elif distance <= radius:
        return "УМЕРЕННЫЙ"
    else:
        return "НИЗКИЙ"

RISK_COLORS = {
    "ОЧЕНЬ ВЫСОКИЙ": "red",
    "ВЫСОКИЙ": "orange",
    "УМЕРЕННЫЙ": "yellow",
    "НИЗКИЙ": "green"
}

def find_cities_at_risk(earthquake_lat, earthquake_lon, radius_km):
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

# ---------------------- 5. Старые глобусы ----------------------
fig_points = px.scatter_geo(
    df,
    lat='latitude', lon='longitude',
    size='magnitude',
    color='magnitude',
    color_continuous_scale=['green','yellow','red'],
    projection='orthographic',
    hover_data=['magnitude','depth','tsunami'],
    title='Распределение землетрясений'
)
fig_points.update_geos(showland=True, landcolor="LightGray", showcountries=True)
fig_points.update_layout(height=750, margin={"r":0,"t":40,"l":0,"b":0})

size_heat = np.clip((df['magnitude'].fillna(df['magnitude'].median()) - df['magnitude'].min()) /
                    max(1e-6, df['magnitude'].max() - df['magnitude'].min()) * 20 + 10, 5, 25)
fig_heat = go.Figure()
fig_heat.add_trace(go.Scattergeo(
    lon=df['longitude'],
    lat=df['latitude'],
    mode='markers',
    marker=dict(size=size_heat, opacity=0.5,
                color=df['magnitude'], colorscale='YlOrRd', showscale=True, colorbar=dict(title='Magnitude')),
    hoverinfo='text',
    hovertext=df.apply(lambda r: f"mag: {r['magnitude']}<br>tsunami: {r['tsunami']}", axis=1)
))
fig_heat.update_geos(projection_type="orthographic", showland=True, landcolor="LightGray", showcountries=True)
fig_heat.update_layout(title="Имитация тепловой карты", height=750, margin={"r":0,"t":40,"l":0,"b":0})

# ---------------------- 6. Таймлапсы ----------------------
# Сортируем данные по году
df_sorted = df[df['Year'] >= 2015].sort_values(by="Year", ascending=True)

# === PREDS ===

frames_pred = []
for _, row in df_sorted.iterrows():
    radius = calculate_earthquake_radius(row['magnitude'], row['cdi'], row['mmi'], row['sig'], row['depth'], row['tsunami'])
    cities_hit = find_cities_at_risk(row['latitude'], row['longitude'], radius)
    if not cities_hit.empty:
        cities_hit['time'] = row['Year']
        frames_pred.append(cities_hit)

df_pred_risk = pd.concat(frames_pred, ignore_index=True) if frames_pred else pd.DataFrame()

fig_pred_risk = px.scatter_geo(
    df_pred_risk,
    lat="lat",
    lon="lng",
    color="risk_level",
    color_discrete_map=RISK_COLORS,
    hover_data=["city", "distance_km", "risk_level"],
    animation_frame="time",
    title="Таймлапс (ПРЕДСКАЗАНИЕ): города в зоне риска"
)
fig_pred_risk.update_geos(projection_type="orthographic", showland=True, showcountries=True)
fig_pred_risk.update_layout(height=750, margin={"r":0,"t":40,"l":0,"b":0})


# === FACT ===
frames_fact = []
for _, row in df_sorted.iterrows():
    radius = calculate_earthquake_radius(row['magnitude'], row['cdi'], row['mmi'], row['sig'], row['depth'], row['tsunami'])
    if row['tsunami'] == 1:
        cities_hit = find_cities_at_risk(row['latitude'], row['longitude'], radius)
        if not cities_hit.empty:
            cities_hit['time'] = row['Year']
            frames_fact.append(cities_hit)

df_fact_risk = pd.concat(frames_fact, ignore_index=True) if frames_fact else pd.DataFrame()

fig_fact_risk = px.scatter_geo(
    df_fact_risk,
    lat="lat",
    lon="lng",
    color="risk_level",
    color_discrete_map=RISK_COLORS,
    hover_data=["city", "distance_km", "risk_level"],
    animation_frame="time",
    title="Таймлапс (ФАКТ): города, попавшие в зону риска"
)
fig_fact_risk.update_geos(projection_type="orthographic", showland=True, showcountries=True)
fig_fact_risk.update_layout(height=750, margin={"r":0,"t":40,"l":0,"b":0})


# ---------------------- 7. Dash приложение ----------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
df_head = df.head(50)

app.layout = html.Div([
    html.H1("Earthquakes Dashboard", style={'textAlign': 'center', 'color': '#f5f5f5'}),

    # Старые глобусы
    html.Div([dcc.Graph(figure=fig_points)], style={'padding': '10px'}),
    html.Div([dcc.Graph(figure=fig_heat)], style={'padding': '10px'}),

    # Таймлапсы
    html.Div([dcc.Graph(figure=fig_pred_risk)], style={'padding': '10px'}),
    html.Div([dcc.Graph(figure=fig_fact_risk)], style={'padding': '10px'}),

    # Метрики моделей
    html.H2("Метрики моделей", style={'color': '#f5f5f5'}),
    html.Div([
        html.P(f"Random Forest: Accuracy={acc_rf:.3f}, AUC={auc_rf:.3f}"),
        html.Pre(report_rf, style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#2e2e2e', 'color': '#f5f5f5'}),
        html.P(f"Gradient Boosting: Accuracy={acc_gb:.3f}, AUC={auc_gb:.3f}"),
        html.Pre(report_gb, style={'whiteSpace': 'pre-wrap', 'backgroundColor': '#2e2e2e', 'color': '#f5f5f5'})
    ], style={'padding': '10px'}),

    # Превью данных
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

# ---------------------- 8. Запуск ----------------------
if __name__ == '__main__':
    url = "http://127.0.0.1:8050"
    print(f"Запуск Dash приложения на {url}")
    try:
        webbrowser.open(url)
    except:
        pass
    app.run(debug=False, port=8050)
