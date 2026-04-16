import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load dataset
df = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
df.columns = df.columns.str.strip()

#remove rwows with missing or invalid values
df = df.replace([float("inf"), -float("inf")], pd.NA)
df = df.dropna()

#convert labels to binary classes
#BENIGN to 0 else 1
df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)

#keep onyl numeric features
x = df.select_dtypes(include=["number"]).copy()

#remove label column from features if still present
if "Label" in x.columns:
	x = x.drop(columns=["Label"])

y = df["Label"]

#split into traiing and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#train model
model = DecisionTreeClassifier(random_state=42, max_depth=10)
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)

#evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("\nConfusion Matrix:")
print(str(cm))
print("\nClassification Report:")
print(report)

#write to file
with open("ml_results.txt", "w") as f:
	f.write(f"Accuracy: {acc}\n\n")
	f.write("Confusion Matrix:\n")
	f.write(str(cm))
	f.write("\n\nClassification Report:\n")
	f.write(report)

print("Results saved to ml_results.txt")
