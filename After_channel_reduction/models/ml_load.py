#Detect whether the network is in rush or empty state using signal, MCS, and data rate patterns.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("../preprocessing/wifi_preprocessed_clean.csv")
df = df.dropna(subset=["Load"])
df["WiFi_Version"] = df["WiFi_Version"].str.extract(r'(\d+)').astype(float)
df["Load"] = df["Load"].str.lower().map({
    "rush": 2,
    "moderate": 1,
    "empty": 0
})

features = [
    "Signal_dBm",
    "MCS",
    "Channel",
   # "Hour", #only keep hour to learn time-of-day patterns without overfitting to specific minutes/seconds
    "Data_Rate_Mbps",
    "WiFi_Version"
]

X = df[features]
y = df["Load"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Network Load Classification - Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

importances = model.feature_importances_
cols = X.columns
#plot feature importance to see which features are most influential in predicting load, which can help us understand the underlying patterns the model is learning and identify key factors that contribute to rush vs empty states:
plt.barh(cols, importances)
plt.title("Feature Importance - Load Prediction")
plt.show()

#plotting load distribution to see if the dataset is imbalanced, which could explain why the model might struggle to predict certain classes (e.g. if rush samples are much less common than empty samples, the model might be biased towards predicting empty):
df["Load"].value_counts().plot(kind="bar")
plt.title("Load Distribution")
plt.show()

#printing percentage distribution of load classes to confirm if there is an imbalance in the dataset that could be affecting model performance:
load_counts = df["Load"].value_counts(normalize=True) * 100

print("Network Load Distribution (%):")
print(load_counts)