# 🌍 Earthquake Risk Assessment & Tsunami Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-Web%20Dashboard-purple.svg)](https://dash.plotly.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-green.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Lines](https://img.shields.io/badge/Code-400%2B%20lines-brightgreen.svg)]()

> **An interactive web dashboard for earthquake analysis, tsunami prediction, and real-time risk assessment for cities worldwide.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Model Performance](#-model-performance)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Architecture & Design](#️-architecture--design)
- [Tech Stack](#-tech-stack)
- [Risk Assessment Logic](#-risk-assessment-logic)
- [Future Improvements](#-future-improvements)
- [Usage Examples](#-usage-examples)
- [Academic Context](#-academic-context)
- [Contact](#-contact)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🧠 Overview
A comprehensive machine learning and geospatial analysis project for **earthquake monitoring, tsunami prediction, and city risk assessment**. The system combines ML models with interactive visualizations to provide insights into seismic activity and potential impact zones.

### 🎯 Project Highlights
- 🌐 **Interactive Global Dashboard:** 4 distinct geo-visualizations
- 🤖 **Dual ML Models:** Random Forest & Gradient Boosting for tsunami prediction
- 🏙️ **City Risk Assessment:** Real-time calculation of risk levels for 41,000+ cities
- 📅 **Time-lapse Analysis:** Historical earthquake impact visualization (2015+)
- 📊 **Comprehensive Metrics:** Accuracy, AUC, classification reports
- 🎨 **Professional UI:** Dark-themed Dash application with responsive design

---

## 🚀 Key Features

### Machine Learning
- **Dual Classification Models:** Random Forest and Gradient Boosting
- **Stratified Training:** Balanced class handling for tsunami prediction
- **Pipeline Processing:** Median imputation and feature scaling
- **Performance Metrics:** Accuracy, ROC-AUC, and detailed classification reports

### Geospatial Analysis
- **Earthquake Radius Calculation:** Multi-factor formula incorporating magnitude, depth, intensity, and tsunami presence
- **Haversine Distance:** Accurate calculation of distances between earthquake epicenters and cities
- **Risk Level Classification:** 4-tier system (VERY HIGH, HIGH, MODERATE, LOW)

### Interactive Visualizations
1. **Earthquake Distribution Map:** Scatter plot with magnitude-based sizing
2. **Heatmap Simulation:** Density visualization of seismic activity
3. **Prediction Time-lapse:** Cities at risk based on all earthquakes (2015+)
4. **Factual Time-lapse:** Actual tsunami-affected cities (2015+)

---

## 📊 Dataset

### Earthquake Data (`earthquake_data_tsunami.csv`)
- Seismic parameters: magnitude, depth, latitude, longitude, etc.
- Tsunami indicator (binary target variable)
- Temporal features: Year, Month

### World Cities Data (`worldcities.csv`)
- 41,000+ cities worldwide
- Geographic coordinates (latitude/longitude)
- Population and country information

---

## 🔬 Workflow
1. **Data Loading & Preparation**
   - Load earthquake and city datasets
   - Convert tsunami indicator to binary
   - Handle missing values and type conversions

2. **Feature Engineering**
   - Select 10 key seismic features
   - Create processed dataset with imputation and scaling

3. **Model Training**
   - 80/20 train-test split with stratification
   - Train Random Forest (balanced class weights)
   - Train Gradient Boosting

4. **Risk Assessment System**
   - Calculate earthquake impact radius using multi-factor formula
   - Compute distances to all cities using Haversine formula
   - Classify risk levels based on distance-to-radius ratio

5. **Visualization Creation**
   - Generate static global earthquake maps
   - Create animated time-lapse visualizations
   - Build interactive Dash dashboard

---

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Class Weight Strategy |
|:------|:---------:|:--------:|:----------------------:|
| Random Forest | Variable* | Variable* | Balanced |
| Gradient Boosting | Variable* | Variable* | Standard |

*Note: Actual values depend on dataset characteristics and are calculated at runtime.*

---

## 🗺️ Visualizations

### 1. Earthquake Distribution Map
- Global orthographic projection
- Magnitude-based point sizing and coloring
- Interactive hover information

### 2. Heatmap Simulation
- Color intensity based on magnitude
- Opacity-adjusted markers for density effect
- YlOrRd color scale

### 3. Prediction Time-lapse
- All earthquakes since 2015
- Affected cities with risk level coloring
- Year-by-year animation

### 4. Factual Time-lapse
- Only tsunami-generating earthquakes
- Actual impacted cities
- Historical risk assessment

---

## 📁 Project Structure
```
earthquake-risk-dashboard/
│
├── 📄 ml (1).py                    # Main application script (400+ lines)
├── earthquake_data_tsunami.csv     # Earthquake dataset
├── worldcities.csv                 # Cities dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation file
```

### Core Components in `ml (1).py`:
- **Data Loading & Preprocessing** (Lines 1-60)
- **ML Model Training** (Lines 60-120)
- **Risk Calculation Functions** (Lines 120-180)
- **Visualization Creation** (Lines 180-280)
- **Dash Application** (Lines 280-400)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required Python packages
pip install numpy pandas scikit-learn plotly dash
```

### Running the Application
```bash
# 1. Ensure data files are in the same directory
#    - earthquake_data_tsunami.csv
#    - worldcities.csv

# 2. Run the main script
python "ml (1).py"

# 3. Open browser (auto-opens to http://127.0.0.1:8050)
```

### Expected Output
- Local Dash server starts on port 8050
- Browser automatically opens the dashboard
- 4 interactive visualizations load
- Model metrics display in the interface
- Sample data table shows earthquake records

---

## 🏗️ Architecture & Design

### Modular Code Structure
```
Data Layer          → Data loading, cleaning, preprocessing
ML Layer            → Model training, prediction, evaluation
Geospatial Layer    → Distance calculation, risk assessment
Visualization Layer → Plot generation, dashboard creation
UI Layer            → Dash application layout and styling
```

### Key Design Decisions
- **Pipeline Processing:** Ensures consistent data transformation
- **Stratified Splitting:** Maintains class balance in training/testing
- **Multi-factor Radius:** Combines seismic parameters for realistic impact zones
- **Animation Frames:** Enables historical analysis through time-lapse
- **Responsive Layout:** Adapts to different screen sizes

---

## 🧰 Tech Stack

### Core Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-0C2340?style=for-the-badge&logo=dash&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Python Libraries Used
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn (RandomForest, GradientBoosting, metrics, preprocessing)
- **Visualization:** plotly.express, plotly.graph_objects
- **Web Dashboard:** dash, dash_table
- **Utilities:** os, math, webbrowser

---

## 🧮 Risk Assessment Logic

### Earthquake Impact Radius Formula
```
base_radius = 10^(0.43 * magnitude - 1.2)
depth_factor = function(depth)  # [0.5, 2.0]
intensity_factor = function(mmi/cdi)  # [0.8, 1.4]
significance_factor = 1.0 + (sig / 5000)
tsunami_factor = 1.3 if tsunami else 1.0
data_quality_factor = function(nst)  # [0.9, 1.1]

final_radius = base_radius × all_factors
```

### Risk Level Classification
| Distance to Radius Ratio | Risk Level | Color |
|--------------------------|------------|-------|
| ≤ 0.3 | VERY HIGH | Red |
| ≤ 0.6 | HIGH | Orange |
| ≤ 1.0 | MODERATE | Yellow |
| > 1.0 | LOW | Green |

---

## 🚧 Future Improvements

### Data & Features
- [ ] Integrate real-time earthquake data feeds (USGS API)
- [ ] Add geological features (fault lines, tectonic plates)
- [ ] Include building codes and infrastructure data
- [ ] Incorporate sea floor topography for tsunami modeling

### Models & Algorithms
- [ ] Time-series prediction of earthquake occurrences
- [ ] Deep learning models for pattern recognition
- [ ] Ensemble methods combining multiple risk factors
- [ ] Uncertainty quantification for predictions

### Visualization
- [ ] 3D visualizations of seismic activity
- [ ] Population density overlay for risk assessment
- [ ] Historical comparison tools
- [ ] Mobile-responsive dashboard design

### Deployment & Scalability
- [ ] Deploy as cloud service (Heroku, AWS, GCP)
- [ ] Add user authentication and data persistence
- [ ] Implement caching for faster load times
- [ ] Create REST API for programmatic access

### User Features
- [ ] Custom city/country filtering
- [ ] Risk notification system
- [ ] Historical event database
- [ ] Export functionality for reports

---

## 📚 Usage Examples

### Running with Custom Data
```python
# Modify these paths in the script
file_eq = 'your_earthquake_data.csv'
file_cities = 'your_cities_data.csv'
```

### Adjusting Model Parameters
```python
# Modify in the training section
rf = RandomForestClassifier(
    n_estimators=300,  # Increase number of trees
    random_state=42,
    class_weight='balanced',
    max_depth=15
)

gb = GradientBoostingClassifier(
    n_estimators=300,  # Increase number of trees
    learning_rate=0.05,  # Adjust learning rate
    random_state=42,
    max_depth=5
)
```

### Customizing Risk Calculation
```python
def custom_radius_calculation(magnitude, depth, intensity):
    # Your custom formula
    custom_radius = magnitude * 100 / (depth + 1)
    return custom_radius
```

---

## 🎓 Academic Context

### Skills Demonstrated
This project showcases comprehensive capabilities in:

#### Data Science & Machine Learning
✅ End-to-end ML pipeline development  
✅ Geospatial data analysis and processing  
✅ Feature engineering for seismic data  
✅ Model evaluation and comparison  
✅ Real-world risk assessment system  

#### Software Engineering
✅ Interactive web application development  
✅ Clean, modular code architecture  
✅ Algorithm implementation (Haversine, risk formulas)  
✅ Professional visualization creation  
✅ User-friendly interface design  

#### Domain Knowledge
✅ Seismology and earthquake parameter understanding  
✅ Tsunami generation mechanisms  
✅ Geographic information systems (GIS)  
✅ Risk assessment methodologies  
✅ Data-driven decision support systems  

---

## 📬 Contact

**Author:** Rogova Ksenia,Musanap Aidemir
**GitHub:**  
**Email:** kkseniaky@gmail.com

*Note: Update contact information as needed*

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Sources:** Earthquake and city datasets
- **Libraries:** Plotly/Dash for interactive visualizations, scikit-learn for ML
- **Geospatial Formulas:** Haversine distance calculation
- **Inspiration:** Seismic monitoring and disaster prevention initiatives

---

<div align="center">

### ⭐ If this project helps in earthquake risk awareness, consider giving it a star! ⭐

**Disclaimer:** This tool is for educational and research purposes. Always refer to official sources (like USGS) for actual earthquake warnings and safety information.

</div>
