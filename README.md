# wifi-ml-analysis
## Project Overview

This project presents a **measurement-based and machine learning-driven analysis of Wi-Fi performance** under varying distance and network load conditions at Habib University.

The study leverages real-world wireless data captured using monitor mode to analyze how factors such as **signal strength, channel conditions, and traffic load** affect Wi-Fi performance.

---

## Objectives

* Analyze Wi-Fi performance using real captured data
* Study the impact of **distance (Near/Far)** and **network load (Rush/Empty)**
* Apply machine learning to predict:

  * Data Rate
  * Network load conditions
* Understand relationships between:

  * RSSI (signal strength)
  * Data rate
  * MCS indices
  * Channel usage


##  Tools & Technologies

* Python (Pandas, Scikit-learn)
* Wireshark / TShark (packet capture)
* Monitor Mode (Aircrack-ng)
* CSV-based dataset processing


## Dataset Description

Each record represents a captured Wi-Fi frame with the following features:

* **Location** (encodes Near/Far + Rush/Empty)
* **Timestamp**
* **Transmitter_MAC / Receiver_MAC**
* **BSSID**
* **Frame_Type**
* **Signal_dBm (RSSI)**
* **Frequency_MHz**
* **PHY_Type**
* **MCS_Legacy / MCS_WiFi5 / MCS_WiFi6**
* **Data_Rate_Mbps**


## Data Processing Pipeline

```
Raw Capture → Channel Reduction → Labeling → Cleaning → Encoding → ML Models
```

### Key Steps:

* Removal of duplicate packets
* Handling missing MCS values
* Conversion of categorical features into numeric labels
* Creation of final ML-ready dataset


##  Machine Learning Tasks

## 1. Network Load Classification (ml_load.py)
 Goal: Categorize the environment into Empty, Moderate, or Rush states.
 Insight: Uses a Random Forest Classifier to identify congestion patterns. By training on Signal_dBm, MCS, and Data_Rate_Mbps, the model detects "Performance     Gaps"—where signals are strong but speeds are low due to high user density.

## 2. Throughput Regression (ml_data_rate.py)
Goal: Predict the actual Data Rate (Mbps) of a specific packet.
Insight: Implements a Random Forest Regressor. This model uses feature engineering (e.g; binary encoding for distance and load) to forecast transmission speeds.
Visual Validation: Features a scatter plot color-coded by "Load," proving that network congestion is a statistically significant predictor of speed variance.

## Key Observations

* Signal strength decreases significantly with distance
* Higher data rates and MCS values are observed in near conditions
* Network congestion (rush hours) introduces variability in performance
* Channel conditions influence achievable throughput

---

## Limitations

* Dataset imbalance (not all locations have both load conditions)
* Limited capture duration
* Modulation schemes were not mentioned sometimes due to limitation of capturing via tshark.


##  Future Work

* Expand dataset across more locations
* Real-time Wi-Fi performance prediction
* Incorporate additional PHY-layer features
* Extend analysis to Wi-Fi 6/6E environments

