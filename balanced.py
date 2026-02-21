import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Daten laden
file_path = "C://Users//maxsc//OneDrive//Hochschule//5_Semester//KI//dataset//creditcard.csv"
df = pd.read_csv(file_path)

# 2. Preprocessing: Time und Amount skalieren
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Originalspalten entfernen
X = df.drop(['Class', 'Time', 'Amount', 'V23', 'V27', 'V28', 'V26', 'V25', 'V22', 'V15', 'V13', 'V24'], axis=1)
y = df['Class']

# 3. Train-Test-Split (Stratified sorgt für gleiches Verhältnis im Testset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Modell MIT Balancing trainieren
# class_weight='balanced' gewichtet die seltene Klasse (Betrug) automatisch stärker
model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
model_balanced.fit(X_train, y_train)
y_pred = model_balanced.predict(X_test)

# 5. Confusion Matrix erstellen
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # Farbe auf Blau gewechselt für den Kontrast
plt.title('Confusion Matrix (Gebalanced mit Class Weights)')
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.show()

# 6. Detail-Check für deinen Vortrag
print(classification_report(y_test, y_pred))