# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# Set the path to the file you'd like to load
file_path = "C://Users//maxsc//OneDrive//Hochschule//5_Semester//KI//dataset//creditcard.csv"

# Load the latest version
df = pd.read_csv(file_path)

print("First 5 records:", df.head())


# 1. Daten laden und splitten
#df = pd.read_csv('creditcard.csv')
X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Modell OHNE Balancing trainieren
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Confusion Matrix erstellen
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix (Unbalanciert)')
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.show()

weights = model.coef_[0]

# 2. DataFrame zur besseren Darstellung erstellen
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Weight': weights
})

# 3. Nach absolutem Gewicht sortieren (Stärkster Einfluss oben)
feature_importance['Absolute_Weight'] = feature_importance['Weight'].abs()
feature_importance = feature_importance.sort_values(by='Absolute_Weight', ascending=False)

print(feature_importance.head(10))