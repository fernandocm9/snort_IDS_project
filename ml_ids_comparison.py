import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#load dataset
df = pd.read_csv("Tuesday-workingHours.pcap_ISCX.csv")
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

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

models = {
	"Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
	"Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
	"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
	}

results = []

#train model
#model = DecisionTreeClassifier(random_state=42, max_depth=10)
#tree models
for name in ["Decision Tree", "Random Forest"]:
	model = models[name]
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	
	results.append({
		"Model": name,
		"Accuracy": accuracy_score(y_test, y_pred),
		"Precision": precision_score(y_test, y_pred),
		"Recall": recall_score(y_test, y_pred),
		"F1": f1_score(y_test, y_pred)
	})

#logistic regression on scaled data
lr = models["Logistic Regression"]
lr.fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)
results.append({
	"Model": "Logistic Regression",
	"Accuracy": accuracy_score(y_test, y_pred),
	"Precision": precision_score(y_test, y_pred),
	"Recall": recall_score(y_test, y_pred),
	"F1": f1_score(y_test, y_pred)
})

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("ml_model_comparison_tuesday.csv", index=False)
