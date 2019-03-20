# Store and organize output files
from pathlib import Path

# Manipulate data
import numpy as np
import pandas as pd

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical tests
import scipy.stats as stats

# Machine learning
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Ignore WARNING
import warnings

warnings.filterwarnings('ignore')

random_seed = 1
np.random.seed = random_seed

results_dir = Path('./results')
results_dir.mkdir(exist_ok=True)

experiment_name = 'linear_SVM_example'
experiment_dir = results_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

dataset_df = pd.read_csv("./Chapter_19_data.csv")

print("Number of features =", dataset_df.shape[1])
print("Number of participants =", dataset_df.shape[0])

dataset_df = dataset_df.dropna()
print("Number of participants =", dataset_df.shape[0])

# Create the contigency table
tab = pd.crosstab(dataset_df["Gender"], dataset_df["label"])
print(tab)
print("")

# Perform the homogeneity test
chi2, p_value, _, _ = stats.chi2_contingency(tab, correction=False)
print('chi2 = %.3f' % chi2)
print('p-value = %.3f' % p_value)

while p_value < 0.05:
    # select one female controls at random and get their indexes
    scz_women = dataset_df[(dataset_df['label'] == 0) & (dataset_df['Gender'] == 1)]
    indexes_to_remove = scz_women.sample(n=1, random_state=1).index
    print('Droping %s' % str(indexes_to_remove.values[0]))
    # remove them from the data
    dataset_df = dataset_df.drop(indexes_to_remove)
    tab = pd.crosstab(dataset_df["Gender"], dataset_df["label"])
    chi2, p_value, _, _ = stats.chi2_contingency(tab, correction=False)

print('chi2 = %.3f' % chi2)
print('p-value = %.3f' % p_value)

# Check new sampple size
tab = pd.crosstab(dataset_df["Gender"], dataset_df["label"])
print(tab)

# Plot normal curve
sns.kdeplot((dataset_df[dataset_df['label'] == 0]['Age']), color="#839098", label=("HC"), shade=True)
sns.kdeplot((dataset_df[dataset_df['label'] == 1]['Age']), color="#f7d842", label=("SZ"), shade=True)
plt.show()

# Shapiro test for normality
_, p_hc = stats.shapiro(dataset_df[dataset_df['label'] == 0]['Age'])
_, p_sz = stats.shapiro(dataset_df[dataset_df['label'] == 1]['Age'])

print('Healthy control - Shapiro-Wilk Normality test: p-value = %.4f' % p_hc)
print('Patients - Shapiro-Wilk Normality test: p-value = %.4f' % p_sz)

# Descriptives
mean_hc, sd_hc = (dataset_df[dataset_df['label'] == 0]['Age']).describe().loc[['mean', 'std']]
mean_sz, sd_sz = (dataset_df[dataset_df['label'] == 1]['Age']).describe().loc[['mean', 'std']]

age_sz = dataset_df[dataset_df['label'] == 0]['Age']
age_hc = dataset_df[dataset_df['label'] == 1]['Age']

statistic, p_value = stats.ttest_ind(age_sz, age_hc)

print('HC: Mean(SD) = %.2f(%.2f)' % (mean_hc, sd_hc))
print('SZ: Mean(SD) = %.2f(%.2f)' % (mean_sz, sd_sz))
print('t = %.2f, p = %.3f' % (statistic, p_value))

features_names = dataset_df.columns[2:]
features_df = dataset_df[features_names]
targets_df = dataset_df['label']

features_df.to_csv(experiment_dir / 'prepared_features.csv')
targets_df.to_csv(experiment_dir / 'prepared_targets.csv')

features = features_df.values.astype('float32')
targets = targets_df.values.astype('int')

predictions_df = dataset_df[['ID', 'label']].copy()
predictions_df['predictions'] = np.nan

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

cv_test_bac = np.zeros((n_folds,1))
cv_test_sens = np.zeros((n_folds,1))
cv_test_spec = np.zeros((n_folds,1))
cv_coefficients = np.zeros((n_folds, len(features_names)))

models_dir = experiment_dir / 'models'
models_dir.mkdir(exist_ok=True)

for i_fold, (train_index, test_index) in enumerate(skf.split(features, targets)):
    features_train, features_test = features[train_index], features[test_index]
    targets_train, targets_test = targets[train_index], targets[test_index]

    print("CV iteration: %d" % (i_fold + 1))
    print("Training set size: %d", len(targets_train))
    print("Test set size:", len(targets_test))

    scaler = StandardScaler()
    features_train_normalized = scaler.fit_transform(features_train)

    clf = LinearSVC(loss='hinge')

    # Hyperparameter seach space
    param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

    # Gridsearch
    internal_cv = StratifiedKFold(n_splits=10)
    grid_cv = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=internal_cv,
                           scoring='balanced_accuracy',
                           verbose=2)

    grid_result = grid_cv.fit(features_train_normalized, targets_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



    best_clf = grid_cv.best_estimator_

    coef_abs_value = np.abs(best_clf.coef_)
    cv_coefficients[i_fold, :] = coef_abs_value

    features_test_normalized = scaler.transform(features_test)
    target_test_predicted = best_clf.predict(features_test_normalized)

    print("Confusion matrix")
    cm = confusion_matrix(targets_test, target_test_predicted)
    print(cm)
    print("")

    tn, fp, fn, tp = cm.ravel()

    test_bac = balanced_accuracy_score(targets_test, target_test_predicted)
    test_sens = tp / (tp + fn)
    test_spec = tn / (tn + fp)

    print("Balanced accuracy: %.4f " % (test_bac))
    print("Sensitivity: %.4f " % (test_sens))
    print("Specificity: %.4f " % (test_spec))

    cv_test_bac[i_fold,:] = test_bac
    cv_test_sens[i_fold,:] = test_sens
    cv_test_spec[i_fold,:] = test_spec

    for row, value in zip(test_index, target_test_predicted):
        predictions_df.at[row, 'predictions'] = value

print("Cross-validation Balanced accuracy: %.4f +- %.4f" % (cv_test_bac.mean(), cv_test_bac.std()))
print("Cross-validation Sensitivity: %.4f +- %.4f" % (cv_test_sens.mean(), cv_test_sens.std()))
print("Cross-validation Specificity: %.4f +- %.4f" % (cv_test_spec.mean(), cv_test_spec.std()))

# Saving feature importance
mean_coeficients = np.mean(cv_coefficients, axis=0).reshape(1,-1)
model_coef_df = pd.DataFrame(data= mean_coeficients, columns=features_names.values)
model_coef_df.to_csv(experiment_dir / 'feature_importance.csv', index=False)

# Saving predictions
predictions_df.to_csv(experiment_dir / "predictions.csv", index=False)

# Saving metrics
metrics = np.concatenate((cv_test_bac, cv_test_sens, cv_test_spec), axis=1)
metrics_df = pd.DataFrame(data=metrics, columns=['bac','sens','spec'])
metrics_df.to_csv(experiment_dir/'metrics.csv', index=False)

# -----------------------------------------------------------------------------

permutation_dir = experiment_dir / 'permutation'
permutation_dir.mkdir(exist_ok=True)


bac = np.mean(cv_test_bac, axis=0)
sens = np.mean(cv_test_sens, axis=0)
spec = np.mean(cv_test_spec, axis=0)

n_permutations = 5

perm_test_bac = np.zeros((n_permutations,1))
perm_test_sens= np.zeros((n_permutations,1))
perm_test_spec= np.zeros((n_permutations,1))
perm_abs_coef= np.zeros((n_permutations, len(features_names) ))

for i_perm in range(n_permutations):
    print('Permutation: %d' % i_perm)
    np.random.seed = i_perm
    targets_permuted = np.random.permutation(targets)

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    cv_test_bac = np.zeros((n_folds, 1))
    cv_test_sens = np.zeros((n_folds, 1))
    cv_test_spec = np.zeros((n_folds, 1))
    cv_coefficients = np.zeros((n_folds, len(features_names)))

    for i_fold, (train_index, test_index) in enumerate(skf.split(features, targets_permuted)):
        features_train, features_test = features[train_index], features[test_index]
        targets_train, targets_test = targets_permuted[train_index], targets_permuted[test_index]

        scaler = StandardScaler()
        features_train_normalized = scaler.fit_transform(features_train)

        clf = LinearSVC(loss='hinge')

        param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

        internal_cv = StratifiedKFold(n_splits=10)
        grid_cv = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               cv=internal_cv,
                               scoring='balanced_accuracy',
                               verbose=0)

        grid_result = grid_cv.fit(features_train_normalized, targets_train)

        best_clf = grid_cv.best_estimator_

        coef_abs_value = np.abs(best_clf.coef_)
        cv_coefficients[i_fold, :] = coef_abs_value


        features_test_normalized = scaler.transform(features_test)
        target_test_predicted = best_clf.predict(features_test_normalized)

        cm = confusion_matrix(targets_test, target_test_predicted)

        tn, fp, fn, tp = cm.ravel()

        test_bac = balanced_accuracy_score(targets_test, target_test_predicted)
        test_sens = tp / (tp + fn)
        test_spec = tn / (tn + fp)

        cv_test_bac[i_fold, :] = test_bac
        cv_test_sens[i_fold, :] = test_sens
        cv_test_spec[i_fold, :] = test_spec

    test_bac = np.mean(cv_test_bac, axis=0)
    test_sens = np.mean(cv_test_sens, axis=0)
    test_spec = np.mean(cv_test_spec, axis=0)

    abs_coef = np.mean(cv_coefficients, axis=0)

    np.save(permutation_dir / ('perm_test_bac_%3d.npy'%i_perm), perm_test_bac)
    np.save(permutation_dir / ('perm_test_sens_%3d.npy'%i_perm), perm_test_sens)
    np.save(permutation_dir / ('perm_test_spec_%3d.npy'%i_perm), perm_test_spec)
    np.save(permutation_dir / ('perm_coef_%3d.npy' % i_perm), perm_abs_coef)

    perm_test_bac[i_perm,:] = test_bac
    perm_test_sens[i_perm,:] = test_sens
    perm_test_spec[i_perm,:] = test_spec
    perm_abs_coef[i_perm,:] = abs_coef

# Get p_values from metrics
pvalue_test_bac = (np.sum(perm_test_bac >= bac) + 1.0) / (n_permutations + 1)
pvalue_test_sens = (np.sum(perm_test_sens >= sens) + 1.0) / (n_permutations + 1)
pvalue_test_spec = (np.sum(perm_abs_coef >= spec) + 1.0) / (n_permutations + 1)

# Get p_values from coef
coef_pvalues = np.zeros((1, len(features_names)))
for i_feature in range(len(features_names)):
    coef_pvalue_temp = (np.sum(perm_abs_coef[:, i_feature] >= mean_coeficients[0, i_feature]) + 1.0) / (n_permutations + 1)
    coef_pvalues[0, i_feature] = coef_pvalue_temp



# Saving
perm_metrics_df = pd.DataFrame(data={'metric': ['bac', 'sens', 'spec'],
                                'value': [bac,sens,spec],
                                'p_value': [pvalue_test_bac,
                                            pvalue_test_sens,
                                            pvalue_test_spec]})

perm_metrics_df.to_csv(experiment_dir/'metrics_permutation_pvalue.csv', index=False)


coef_df = pd.DataFrame( index=['coefficients', 'p value'],
                        data=np.concatenate((mean_coeficients, coef_pvalues)),
                        columns=features_names)
coef_df.to_csv(experiment_dir/'coef_permutation_pvalue.csv', index=True)
