import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from matplotlib import rcParams


def load_data():

    healthy_path = "E:\Health\Features_baseDate_1D+2D_Total.xlsx"
    patient_path = "E:\PAF\Features_baseDate_1D+2D_Total.xlsx"

    healthy_df = pd.read_excel(healthy_path, index_col=0)
    patient_df = pd.read_excel(patient_path, index_col=0)

    def process_df(df):
        df_t = df.T.reset_index()
        df_t = df_t.rename(columns={"index": "sample_name"})


        df_t["subject_id"] = df_t["sample_name"].apply(
            lambda x: x.split("-")[0].strip()
        )
        return df_t

    healthy_samples = process_df(healthy_df)
    patient_samples = process_df(patient_df)

    healthy_samples["label"] = 0
    patient_samples["label"] = 1

    full_df = pd.concat([healthy_samples, patient_samples], axis=0)

    sample_names = full_df["sample_name"].values
    X = full_df.drop(columns=["label", "subject_id", "sample_name"]).values
    y = full_df["label"].values
    subjects = full_df["subject_id"].values

    feature_names = full_df.drop(columns=["label", "subject_id", "sample_name"]).columns.tolist()
    return X, y, feature_names, subjects, sample_names


X, y, feature_names, subjects, sample_names = load_data()


print("[Data Validation]")
print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"First 100 subject IDs: {subjects[:100]}")
print(f"First 5 feature names: {feature_names[:20]}")
print(f"Label distribution: Healthy={sum(y == 0)}, Patients={sum(y == 1)}\n")

# Data splitting (grouped by subjects)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
train_idx, test_idx = next(splitter.split(X, y, groups=subjects))

# Get split data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_subjects, test_subjects = subjects[train_idx], subjects[test_idx]
test_sample_names = sample_names[test_idx]

# Split validation
print("[Data Split Validation]")
print(f"Number of unique subjects in training set: {len(np.unique(train_subjects))}")
print(f"Number of unique subjects in test set: {len(np.unique(test_subjects))}")
print(f"Training set samples: {len(train_idx)}, Test set samples: {len(test_idx)}")
print(f"Test set subjects: {np.unique(test_subjects)}\n")
print("Training set 0/1 ratio:", np.bincount(y_train))
print("Test set 0/1 ratio:", np.bincount(y_test))


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

robust_scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(5.0, 95.0)
).fit(X_train_imp)
X_train_robust = robust_scaler.transform(X_train_imp)
X_test_robust = robust_scaler.transform(X_test_imp)

scaler = StandardScaler().fit(X_train_robust)
X_train_scaled = scaler.transform(X_train_robust)
X_test_scaled = scaler.transform(X_test_robust)


def calculate_feature_importance(X_train, y_train):

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train.ravel())

    feature_scores = rf.feature_importances_

    sorted_indices = np.argsort(feature_scores)[::-1]
    return feature_scores,sorted_indices

mw_scores,sorted_indices = calculate_feature_importance(X_train_scaled, y_train)




from sklearn.metrics import (confusion_matrix, f1_score,
                             roc_curve, auc, precision_score, recall_score)

best_clf = XGBClassifier(random_state=42,n_estimators=1000,learning_rate=0.5, eval_metric='logloss')
best_k = 200
selected_indices = sorted_indices[:best_k]

X_train_best = X_train_scaled[:, selected_indices]
X_test_best = X_test_scaled[:, selected_indices]

best_clf.fit(X_train_best, y_train)

y_pred = best_clf.predict(X_test_best)
y_proba = best_clf.predict_proba(X_test_best)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
f1 = f1_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("\n" + "="*60)
print(f"Optimal number of features: {best_k}")
print("Default classification threshold confusion matrix:")
print(cm)
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value: {ppv:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {roc_auc:.4f}")

# Feature names output
print(f"\n{' Important Feature Analysis ':=^50}")
print("TOP 200 features:")
for idx in sorted_indices[:200]:
    print(f"{feature_names[idx]:<25} | Importance: {mw_scores[idx]*100:.4f}")
print(f"\nSelected {best_k} features:")
print([feature_names[i] for i in sorted_indices[:best_k]])

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# Plot confusion matrix heatmap
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Healthy', 'Predicted Patient'],
            yticklabels=['Actual Healthy', 'Actual Patient'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Default Classification Threshold Confusion Matrix', fontsize=14)
plt.show()










from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier

xgb = XGBClassifier(random_state=42,n_estimators=1000,learning_rate=0.5,eval_metric='logloss')
svm_model = svm.SVC(kernel='rbf', probability=True, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=2000,max_depth=5,min_samples_split=10,max_features=0.6,bootstrap=False,random_state=42)
meta = svm.SVC(C=1,kernel='linear',probability=True,random_state=42)


models = [
    ('SVM', svm.SVC(kernel='rbf', probability=True, random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42, n_estimators=400, learning_rate=0.1, eval_metric='logloss')),
    # ('GaussianNB',GaussianNB() ),
    ('ExtraTrees',ExtraTreesClassifier(n_estimators=300,max_depth=5,min_samples_split=10,max_features=0.6,bootstrap=False,random_state=42)),
    ('Stacking',StackingClassifier(classifiers=[xgb, svm_model, gb,et],meta_classifier=meta,use_probas=True, average_probas=False, verbose=1))
]


results_dict = {}
roc_data = {}


for name, model in models:
    print(f"\n{'=' * 30} {name} Model evaluation {'=' * 30}")


    X_train_model = X_train_scaled[:, selected_indices]
    X_test_model = X_test_scaled[:, selected_indices]


    model.fit(X_train_model, y_train)


    y_pred = model.predict(X_test_model)
    y_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
        X_test_model)


    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'PPV': tp / (tp + fp) if (tp + fp) != 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) != 0 else 0,
        'F1': f1_score(y_test, y_pred),
        'AUC': auc(*roc_curve(y_test, y_proba)[:2])
    }


    results_dict[name] = metrics
    roc_data[name] = roc_curve(y_test, y_proba)

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 20

results_df = pd.DataFrame(results_dict).T
print("\n Model performance comparison:")
print(results_df.round(4))


plt.figure(figsize=(7,6))
colors = ['orange', 'green', 'red', 'purple', 'blue']

for (name, (fpr, tpr, _)), color in zip(roc_data.items(), colors):
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {results_dict[name]["AUC"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlabel('Specificity', fontsize=20, fontname='Times New Roman')
# plt.ylabel('Sensitivity', fontsize=20, fontname='Times New Roman')
# plt.title('Receiver Operating Characteristic Curve', fontsize=20, fontname='Times New Roman')

plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')

plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 12})

plt.grid(linestyle='--', alpha=0.7)
plt.show()


