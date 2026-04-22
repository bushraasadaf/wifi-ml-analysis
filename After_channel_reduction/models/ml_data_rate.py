import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("../preprocessing/wifi_preprocessed_clean.csv")

# drop missing target
df = df.dropna(subset=["Data_Rate_Mbps"])


# ENCODE ALL STRINGS

# Binary encoding
df["Distance"] = df["Distance"].map({"Near": 1, "Far": 0, "Unknown": -1})
df["Load"] = df["Load"].map({"Rush": 2, "Moderate": 1, "Empty": 0, "Unknown": -1})

# WiFi version encoding
df["WiFi_Version"] = df["WiFi_Version"].map({
    "WiFi4": 0,
    "WiFi5": 1,
    "WiFi6": 2,
    "Unknown": -1
})

# PHY type encoding to binary and drop original column
df = pd.get_dummies(df, columns=["PHY_Type"], drop_first=True)


# FEATURES & TARGET
#drop Location since it's just a label and won't help the model learn patterns for data rate prediction
#drop data rate from features since it's the target variable
X = df.drop(columns=["Data_Rate_Mbps", "Location","Minute","Second"])

y = df["Data_Rate_Mbps"]

# SPLIT: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MODEL
#n_estimators is the number of trees in the forest. More trees can improve performance but also increase training time and risk of overfitting. 200 is a common choice for a good balance.
#use n_estimators=200 for balance of performance and training time
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#MODEL EVALUATION:
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
# Create a DataFrame to compare actual vs predicted values
# results = pd.DataFrame({
#     "Actual": y_test.values,
#     "Predicted": y_pred
# })

print(results.head(20))

import matplotlib.pyplot as plt
# Scatter plot of actual vs predicted data rates
plt.figure()

colors = df.loc[X_test.index, "Load"].map({
    0: "blue",
    1: "green",
    2: "red" #rush = red, moderate = green, empty = blue
})
plt.scatter(y_test, y_pred, c=colors)
plt.xlabel("Actual Data Rate (Mbps)")
plt.ylabel("Predicted Data Rate (Mbps)")
plt.title("Actual vs Predicted (Colored by Load)")
 # Add a reference line for perfect predictions
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Empty'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Moderate'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Rush'),
])
plt.show()

# FEATURE IMPORTANCE:
importances = model.feature_importances_
cols = X.columns

plt.barh(cols, importances)
plt.title("Feature Importance - Data Rate Prediction")
plt.show()