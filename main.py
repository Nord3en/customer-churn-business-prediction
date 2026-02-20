import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
churn = pd.read_csv("data/churn.csv")
print("\n--- az elso 5 sor ---")
print(churn.head())

print("\n--- technikai infok ---")
print(churn.info())

print("\n--- leiro statisztika ---")
print(churn.describe())
print("\n--- chunrn count---")
print(churn['Churn'].value_counts())

churn.drop("customerID", axis='columns',inplace=True)
churn["TotalCharges"] = pd.to_numeric(churn["TotalCharges"], errors='coerce')
churn = churn.dropna()

churn = pd.get_dummies(churn)
churn.drop("Churn_No",axis="columns",inplace=True)
print("\n--- az elso 5 sor ---")
print(churn.head(5))

print("\n--- technikai infok ---")
print(churn.info())

print("\n--- leiro statisztika ---")
print(churn.describe())
print(churn["gender_Female"].head(5))
y= churn["Churn_Yes"]
X= churn.drop("Churn_Yes", axis='columns') 
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.30, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

model.fit(X_train,y_train)

predictions= model.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, predictions)

print("Accuracy: ",accuracy)

print(sklearn.metrics.classification_report(y_test, predictions))

probs = model.predict_proba(X_test)[:, 1]

# Húzzuk meg a határt 30%-nál (0.3) 0.5 helyett
new_predictions = (probs > 0.3).astype(int)

from sklearn.metrics import classification_report
print("Eredmény 30%-os küszöbnél:")
print(classification_report(y_test, new_predictions))

importances = pd.Series(model.feature_importances_, index=X.columns)

# Rendezzük sorba és nézzük a top 10-et
importances.nlargest(10).plot(kind='barh')
plt.title("Miért mennek el az ügyfelek? (Top 10 ok)")
plt.show()