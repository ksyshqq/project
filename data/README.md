

# 📁 Data Directory — Earthquake Impact & Tsunami Risk Dashboard

[![Dataset](https://img.shields.io/badge/Datasets-Earthquake%20%2B%20Cities-blue.svg)]()
[![USGS](https://img.shields.io/badge/Source-USGS-orange.svg)]()
[![Cities](https://img.shields.io/badge/World%20Cities-41k%2B-green.svg)]()
[![Time-lapse](https://img.shields.io/badge/Frames-Generated-purple.svg)]()

> **This directory contains all datasets required for earthquake analysis, tsunami prediction, and city-level risk assessment.**

---

## 📦 Dataset Files

* **earthquake_data_tsunami.csv** — main earthquake dataset (USGS)
* **worldcities.csv** — global cities dataset
* **time_lapse_predicted.csv** — generated risk prediction frames
* **time_lapse_actual_tsunami.csv** — real tsunami event frames

---

## 📥 How to Get the Data

1. Download the **earthquake dataset** (CSV) from the official USGS Earthquake Catalog.
2. Download the **world cities dataset** (public open-source CSV).
3. Place both files into this directory.
4. Run preprocessing scripts to generate:

   * explosion radius values
   * ML features
   * time-lapse datasets (predicted & actual)

---

## 📊 Dataset Description

### 🌋 earthquake_data_tsunami.csv

Global earthquake records including:

* latitude, longitude
* magnitude, depth
* intensity metrics: MMI, CDI
* tsunami flag (0 = no tsunami, 1 = tsunami)
* sig, gap, dmin, nst
* engineered features for ML models

---

### 🏙️ worldcities.csv

City-level dataset containing:

* city name
* country
* latitude, longitude
* population

---

### 🕒 time_lapse_predicted.csv

Generated frames used for **predicted time-lapse animation** of cities affected by all earthquakes since 2015.

### 🌊 time_lapse_actual_tsunami.csv

Generated frames for the **historical time-lapse** of real tsunami-affected cities.

---

## 📝 Notes

* Raw CSV files are **not included** in Git (large size).
* All datasets must be stored in **UTF-8** encoding.
* Preprocessing scripts must be run before launching the dashboard.



