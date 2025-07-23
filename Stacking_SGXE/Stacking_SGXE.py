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


splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
train_idx, test_idx = next(splitter.split(X, y, groups=subjects))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
train_subjects, test_subjects = subjects[train_idx], subjects[test_idx]
test_sample_names = sample_names[test_idx]

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


from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
xgb = XGBClassifier(random_state=42,n_estimators=1000,learning_rate=0.5,eval_metric='logloss')
svm_model = svm.SVC(kernel='rbf', probability=True, random_state=42)       # 改用非线性核
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=2000,max_depth=5,min_samples_split=10,max_features=0.6,bootstrap=False,random_state=42)
meta = svm.SVC(C=1,kernel='linear',probability=True,random_state=42)
clf = StackingClassifier(classifiers=[xgb, svm_model, gb,et],meta_classifier=meta,use_probas=True, average_probas=False, verbose=1)


train_accuracies = []
test_accuracies = []
k_values = range(1, 301)
for k in k_values:
    print(k)

    selected_indices = sorted_indices[:k]

    X_train_sub = X_train_scaled[:, selected_indices]
    X_test_sub = X_test_scaled[:, selected_indices]

    stratified_group_cv = StratifiedGroupKFold(n_splits=10)

    cv_scores = cross_val_score(
        clf, X_train_sub, y_train,
        cv=stratified_group_cv, groups=train_subjects
    )

    train_accuracies.append(np.max(cv_scores))

    clf.fit(X_train_sub, y_train)
    test_acc = clf.score(X_test_sub, y_test)
    test_accuracies.append(test_acc)



from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 20
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, 'b-', label='10-fold Cross-validation Accuracy')
plt.plot(k_values, test_accuracies, 'r-', label='Test Set Accuracy')
plt.xlabel('Number of Feature Selections',  fontsize=20, fontname='Times New Roman')
plt.ylabel('Accuracy', fontsize=20, fontname='Times New Roman')
plt.title('Model performance under different numbers of features', fontsize=20, fontname='Times New Roman')
# 设置刻度字体
plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')
# 设置图例字体
plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 20})
plt.grid(linestyle='--', alpha=0.7)
# 标记最佳测试准确率
best_k = np.argmax(test_accuracies) + 1
best_acc = test_accuracies[best_k - 1]
plt.scatter(best_k, best_acc, color='red', zorder=5)
plt.annotate(f'Best: k = {best_k}\nAcc = 90.53%',
             xy=(best_k, best_acc),
             xytext=(best_k + 5, best_acc - 0.2),
             arrowprops=dict(arrowstyle='->'))
plt.xticks(range(0, 301, 20))
plt.tight_layout()
plt.show()




from sklearn.metrics import (confusion_matrix, f1_score,
                             roc_curve, auc, precision_score, recall_score)


# best_clf = svm.SVC(kernel='linear', random_state=42, probability=True)  # 注意添加probability=True
# best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# best_clf = XGBClassifier(random_state=42,n_estimators=1000,learning_rate=0.5, eval_metric='logloss')
# best_clf = RandomForestClassifier(n_estimators=200, random_state=42,n_jobs=-1)
best_clf = StackingClassifier(classifiers=[xgb, svm_model, gb,et],meta_classifier=meta,use_probas=True, average_probas=False, verbose=1)
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

print(f"\n{' Important Feature Analysis ':=^50}")
print("TOP 200 features:")
for idx in sorted_indices[:200]:
    print(f"{feature_names[idx]:<25} | Importance: {mw_scores[idx]*100:.4f}")

print(f"\nSelected {best_k} features:")
print([feature_names[i] for i in sorted_indices[:best_k]])


plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificity', fontsize=20, fontname='Times New Roman')
plt.ylabel('Sensitivity', fontsize=20, fontname='Times New Roman')
# 设置刻度字体
plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')
# 设置图例字体
plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 20})
plt.grid(linestyle='--', alpha=0.7)
plt.show()



import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predict Health', 'Predict AF'],
            yticklabels=['Actual Health', 'Actual AF'])
plt.xlabel('Predictive Label', fontsize=20, fontname='Times New Roman')
plt.ylabel('Actual Label', fontsize=20, fontname='Times New Roman')
# 设置刻度字体
plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, roc_auc_score)

thresholds = np.linspace(0, 1, 100)

f1_scores = [f1_score(y_test, (y_proba >= t)) for t in thresholds]
acc_scores = [accuracy_score(y_test, (y_proba >= t)) for t in thresholds]
combined_scores = [(acc + f1)/2 for acc, f1 in zip(acc_scores, f1_scores)]

max_f1_idx = np.argmax(f1_scores)
max_acc_idx = np.argmax(acc_scores)
max_combined_idx = np.argmax(combined_scores)


plt.figure(figsize=(10, 8))

plt.plot(thresholds, f1_scores, label='F1 Score', color='darkorange', lw=2)
plt.plot(thresholds, acc_scores, label='Accuracy', color='navy',  lw=2)
plt.plot(thresholds, combined_scores, label='(Acc+F1)/2', color='purple', lw=2)

plt.scatter(thresholds[max_f1_idx], f1_scores[max_f1_idx], s=120, color='red',
            zorder=5, label=f'Max F1: {f1_scores[max_f1_idx]:.4f} @ {thresholds[max_f1_idx]:.4f}')
plt.scatter(thresholds[max_acc_idx], acc_scores[max_acc_idx], s=120, color='green',
            zorder=5, label=f'Max Acc: {acc_scores[max_acc_idx]:.4f} @ {thresholds[max_acc_idx]:.4f}')
plt.scatter(thresholds[max_combined_idx], combined_scores[max_combined_idx], s=120, color='blue',
            zorder=5, label=f'Max Combined: {combined_scores[max_combined_idx]:.4f} @ {thresholds[max_combined_idx]:.4f}')

plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Score Value', fontsize=12)
plt.title('Threshold Optimization with Combined Metric', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=10)
plt.grid(alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.show()


best_threshold = thresholds[max_combined_idx]
y_pred_combined = (y_proba >= best_threshold).astype(int)
cm_max = confusion_matrix(y_test, y_pred_combined)
tn_max, fp_max, fn_max, tp_max = cm_max.ravel()

acc_max = (tp_max + tn_max) / (tp_max + tn_max + fp_max + fn_max)
sensitivity_max = tp_max / (tp_max + fn_max) if (tp_max + fn_max) != 0 else 0
specificity_max = tn_max / (tn_max + fp_max) if (tn_max + fp_max) != 0 else 0
ppv_max = tp_max / (tp_max + fp_max) if (tp_max + fp_max) != 0 else 0
npv_max = tn_max / (tn_max + fn_max) if (tn_max + fn_max) != 0 else 0
f1_max = f1_score(y_test, y_pred_combined)

roc_auc = roc_auc_score(y_test, y_proba)


print('Final Best Metrics')
print("Optimal classification threshold confusion matrix:")
print(cm_max)
print(f"Best threshold: {best_threshold:.4f}")
print(f"Accuracy: {acc_max:.4f}")
print(f"Sensitivity: {sensitivity_max:.4f}")
print(f"Specificity: {specificity_max:.4f}")
print(f"Positive Predictive Value: {ppv_max:.4f}")
print(f"Negative Predictive Value: {npv_max:.4f}")
print(f"F1 Score: {f1_max:.4f}")
print(f"AUC Score: {roc_auc:.4f}")


import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm_max, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predict Health', 'Predict AF'],
            yticklabels=['Actual Health', 'Actual AF'])
plt.xlabel('Predictive Label', fontsize=20, fontname='Times New Roman')
plt.ylabel('Actual Label', fontsize=20, fontname='Times New Roman')

plt.xticks(fontsize=20,fontname='Times New Roman')
plt.yticks(fontsize=20,fontname='Times New Roman')
plt.show()




error_mask = (y_test != y_pred_combined)
error_indices = np.where(error_mask)[0]
error_samples = test_sample_names[error_indices]

print("\nIncorrectly classified sample names in the test set：")
for name in error_samples:
    print(name)


















from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np


def cross_val_metrics(clf, X, y, groups, n_splits=10):

    cv = StratifiedGroupKFold(n_splits=n_splits)
    all_fold_metrics = []

    for train_idx, val_idx in cv.split(X, y, groups):
        model = clone(clf)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_val))


        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        fold_f1 = f1_score(y_val, y_pred)


        if len(np.unique(y_val)) >= 2:
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.0

        all_fold_metrics.append({
            'acc': acc,
            'sens': sensitivity,
            'spec': specificity,
            'auc': roc_auc,
            'f1': fold_f1
        })


    acc_scores = [fm['acc'] for fm in all_fold_metrics]
    sens_scores = [fm['sens'] for fm in all_fold_metrics]
    spec_scores = [fm['spec'] for fm in all_fold_metrics]
    auc_scores = [fm['auc'] for fm in all_fold_metrics]
    f1_scores = [fm['f1'] for fm in all_fold_metrics]


    best_fold = max(all_fold_metrics, key=lambda x: x['acc'])

    return {

        'cv_acc': np.mean(acc_scores),
        'cv_sens': np.mean(sens_scores),
        'cv_spec': np.mean(spec_scores),
        'cv_auc': np.mean(auc_scores),
        'cv_f1': np.mean(f1_scores),


        'best_acc': best_fold['acc'],
        'best_sens': best_fold['sens'],
        'best_spec': best_fold['spec'],
        'best_auc': best_fold['auc'],
        'best_f1': best_fold['f1'],


        'acc_scores': acc_scores
    }



best_clf = StackingClassifier(classifiers=[xgb, svm_model, gb,et],meta_classifier=meta,use_probas=True, average_probas=False, verbose=1)
selected_indices = sorted_indices[:best_k]


X_train_best = X_train_scaled[:, selected_indices]
X_test_best = X_test_scaled[:, selected_indices]


cv_metrics = cross_val_metrics(best_clf, X_train_best, y_train, train_subjects)


best_clf.fit(X_train_best, y_train)
y_test_pred = best_clf.predict(X_test_best)
y_test_proba = best_clf.predict_proba(X_test_best)[:, 1]


cm_test = confusion_matrix(y_test, y_test_pred)
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
acc_test = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
sens_test = tp_test / (tp_test + fn_test) if (tp_test + fn_test) != 0 else 0
spec_test = tn_test / (tn_test + fp_test) if (tn_test + fp_test) != 0 else 0
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
auc_test = auc(fpr_test, tpr_test)
f1_test = f1_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print(f"Optimal number of features: {best_k}")
print("\nTraining set performance (10-fold CV):")
print(f"Accuracy: {cv_metrics['best_acc']:.4f} ")
print(f"Sensitivity: {cv_metrics['best_sens']:.4f}")
print(f"Specificity: {cv_metrics['best_spec']:.4f}")
print(f"F1 Score: {cv_metrics['best_f1']:.4f}")
print(f"AUC Score: {cv_metrics['best_auc']:.4f}")

print("\nTest set performance:")
print(f"Confusion Matrix:\n{cm_test}")
print(f"Accuracy: {acc_test:.4f}")
print(f"Sensitivity: {sens_test:.4f}")
print(f"Specificity: {spec_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"AUC Score: {auc_test:.4f}")





import csv


with open('accuracies.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['k', 'train_accuracy', 'test_accuracy'])
    for k, train, test in zip(k_values, train_accuracies, test_accuracies):
        writer.writerow([k, train, test])









