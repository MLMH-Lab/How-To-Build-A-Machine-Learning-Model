# SNIPPET 1

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
from sklearn.externals import joblib
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Ignore WARNING
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# SNIPPET 2

random_seed = 1
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# SNIPPET 3

results_dir = Path('./results')
results_dir.mkdir(exist_ok=True)

experiment_name = 'linear_SVM_example'
experiment_dir = results_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# SNIPPET 4

dataset_file = Path('./Chapter_19_data.csv')
dataset_df = pd.read_csv(dataset_file, index_col='ID')

# --------------------------------------------------------------------------
# SNIPPET 5

patient_str = 'sz'
healthy_str = 'hc'
male_str = 'M'
female_str = 'F'

# --------------------------------------------------------------------------
# SNIPPET 6

# >>> dataset_df[0:6]

# Output
#      Diagnosis Gender   Age  ...  rh temporalpole thickness  rh transversetemporal thickness  rh insula thickness
# ID                           ...
# c001        hc      M  22.0  ...                   2.235844                         2.300844             2.645844
# c002        hc      F  24.0  ...                   2.622699                         2.322699             2.673699
# c003        hc      F  22.0  ...                   2.232989                         2.267989             2.795989
# c004        hc      F  30.0  ...                   1.956654                         2.297654             2.731654
# c005        hc      M  31.0  ...                   3.162771                         2.081771             2.607771
# c006        hc      F   NaN  ...                   3.512643                         2.591643             2.606643
#
# [6 rows x 172 columns]

# --------------------------------------------------------------------------
# SNIPPET 7

# >>> dataset_df.columns.tolist()

# Output
# ['Diagnosis',
#  'Gender',
#  'Age',
#  'Left Lateral Ventricle',
#  'Left Inf Lat Vent',
#  'Left Cerebellum White Matter',
#  'Left Cerebellum Cortex',
#  'Left Thalamus Proper',
#  'Left Caudate',
#  'Left Putamen',
#  'Left Pallidum',
#  'rd Ventricle',
#  'th Ventricle',
#  'Brain Stem',
#  'Left Hippocampus',
#  'Left Amygdala',
#  'CSF',
#  'Left Accumbens area',
#  'Left VentralDC',
# ...
#  'rh pericalcarine thickness',
#  'rh postcentral thickness',
#  'rh posteriorcingulate thickness',
#  'rh precentral thickness',
#  'rh precuneus thickness',
#  'rh rostralanteriorcingulate thickness',
#  'rh rostralmiddlefrontal thickness',
#  'rh superiorfrontal thickness',
#  'rh superiorparietal thickness',
#  'rh superiortemporal thickness',
#  'rh supramarginal thickness',
#  'rh frontalpole thickness',
#  'rh temporalpole thickness',
#  'rh transversetemporal thickness',
#  'rh insula thickness']
# --------------------------------------------------------------------------
# SNIPPET 8

print('Number of features = %d' % dataset_df.shape[1])
print('Number of participants = %d' % dataset_df.shape[0])

# Out
# Number of features = 172
# Number of participants = 740
# --------------------------------------------------------------------------
# SNIPPET 9
null_lin_bool = dataset_df.isnull().any(axis=1)
null_cols = dataset_df.columns[dataset_df.isnull().any(axis=0)]

n_null = dataset_df.isnull().sum().sum()
print('Number of missing data = %d' % n_null)
subj_null = dataset_df[null_lin_bool].index
print('IDs: %s' % (', ').join(subj_null.tolist()))
# >>> pd.DataFrame(dataset_df[null_cols].isnull().sum(), columns=['N missing'])

# Output
# Number of missing data = 43
# IDs: c006, p149, p150, p156, p157, p175, p195, p196, p197, p210, p211, p212, p227, p228, p229, p264, p265, p266, p267, p268, p269, p270, p271, p281, p282, p283, p289, p302, p303, p307, p311, p312, p319, p321, p356, p357, p358, p359, p360, p361, p362, p363, p364
#      N missing
# Age         43
# --------------------------------------------------------------------------
# SNIPPET 10

dataset_df = dataset_df.dropna()
print('Number of participants = %d' % dataset_df.shape[0])

# Out
# Number of participants = 697
# --------------------------------------------------------------------------
# SNIPPET 11

# >>> dataset_df['Diagnosis'].value_counts()

# Out
# hc    367
# sz    330
# Name: Diagnosis, dtype: int64
# --------------------------------------------------------------------------
# SNIPPET 12

sns.countplot(x='Diagnosis', hue='Gender', data=dataset_df, palette=['#839098', '#f7d842'])
plt.legend(['Male', 'Female'])
plt.show()

# --------------------------------------------------------------------------
# SNIPPET 13

# Create the contingency table
contingency_table = pd.crosstab(dataset_df['Gender'], dataset_df['Diagnosis'])
print(contingency_table)

# Perform the homogeneity test
chi2, p_gender, _, _ = stats.chi2_contingency(contingency_table, correction=False)
print('Gender')
print('Chi-square test: chi2 stats = %.3f p-value = %.3f' % (chi2, p_gender))

# Out
# Diagnosis   hc   sz
# Gender
# F          162  121
# M          205  209
# Gender
# Chi-square test: chi2 stats = 4.026 p-value = 0.045
# --------------------------------------------------------------------------
# SNIPPET 14

print('Removing participant to balance gender...')
while p_gender < 0.05:
    # Select one female controls at random and get their indexes
    scz_women = dataset_df[(dataset_df['Diagnosis'] == healthy_str) & (dataset_df['Gender'] == female_str)]
    indexes_to_remove = scz_women.sample(n=1, random_state=1).index
    print('Droping %s' % str(indexes_to_remove.values[0]))
    # remove them from the data
    dataset_df = dataset_df.drop(indexes_to_remove)
    contingency_table = pd.crosstab(dataset_df['Gender'], dataset_df['Diagnosis'])
    chi2, p_gender, _, _ = stats.chi2_contingency(contingency_table, correction=False)
    print('new p-value = %.3f' % p_gender)

print('Gender')
print('Chi-square test: chi2 stats = %.3f p-value = %.3f' % (chi2, p_gender))

# Check new sample size
contingency_table = pd.crosstab(dataset_df['Gender'], dataset_df['Diagnosis'])
print(contingency_table)

# Out
# Removing participant to balance gender...
# Droping c082
# new p-value = 0.049
# Droping c083
# new p-value = 0.054
# Gender
# Chi-square test: chi2 stats = 3.698 p-value = 0.054
# Diagnosis   hc   sz
# Gender
# F          160  121
# M          205  209
# --------------------------------------------------------------------------
# SNIPPET 15

age_hc = dataset_df[dataset_df['Diagnosis'] == healthy_str]['Age']
age_sz = dataset_df[dataset_df['Diagnosis'] == patient_str]['Age']

# Plot normal curve
sns.kdeplot(age_hc,
            color='#839098',
            label='HC',
            shade=True)
sns.kdeplot(age_sz,
            color='#f7d842',
            label='SZ',
            shade=True)
plt.show()

# Shapiro test for normality
_, p_age_hc_normality = stats.shapiro(age_hc)
_, p_age_sz_normality = stats.shapiro(age_sz)

print('HC: Normality test: p-value = %.3f' % p_age_hc_normality)
print('SZ: Normality test: p-value = %.3f' % p_age_sz_normality)

# Descriptives
mean_age_hc, sd_age_hc = age_hc.describe().loc[['mean', 'std']]
mean_age_sz, sd_sz = age_sz.describe().loc[['mean', 'std']]

print('Age')
print('HC: Mean(SD) = %.2f(%.2f)' % (mean_age_hc, sd_age_hc))
print('SZ: Mean(SD) = %.2f(%.2f)' % (mean_age_sz, sd_sz))

# Out
# HC: Normality test: p-value = 0.005
# SZ: Normality test: p-value = 0.018
# Age
# HC: Mean(SD) = 25.31(2.84)
# SZ: Mean(SD) = 24.98(3.12)
# --------------------------------------------------------------------------
# SNIPPET 16

t_stats, p_age = stats.ttest_ind(age_sz, age_hc)
print('Age')
print("Student's t test: t stats = %.3f, p-value = %.3f" % (t_stats, p_age))

# Out
# Age
# Student's t test: t stats = -1.464, p-value = 0.144
# --------------------------------------------------------------------------
# SNIPPET 17

features_names = dataset_df.columns[3:]
features_df = dataset_df[features_names]
targets_df = dataset_df['Diagnosis']

# --------------------------------------------------------------------------
# SNIPPET 18

features_df.to_csv(experiment_dir / 'prepared_features.csv')
targets_df.to_csv(experiment_dir / 'prepared_targets.csv')

# --------------------------------------------------------------------------
# SNIPPET 19

targets_df = targets_df.map({healthy_str: 0, patient_str: 1})
targets = targets_df.values.astype('int')

features = features_df.values.astype('float32')

# --------------------------------------------------------------------------
# SNIPPET 20

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

# --------------------------------------------------------------------------
# SNIPPET 21

predictions_df = pd.DataFrame(targets_df)
predictions_df['predictions'] = np.nan

bac_cv = np.zeros((n_folds, 1))
sens_cv = np.zeros((n_folds, 1))
spec_cv = np.zeros((n_folds, 1))
coef_cv = np.zeros((n_folds, len(features_names)))

models_dir = experiment_dir / 'models'
models_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# SNIPPET 22

for i_fold, (train_idx, test_idx) in enumerate(skf.split(features, targets)):
    features_train, features_test = features[train_idx], features[test_idx]
    targets_train, targets_test = targets[train_idx], targets[test_idx]

    print('CV iteration: %d' % (i_fold + 1))
    print('Training set size: %d' % len(targets_train))
    print('Test set size: %d' % len(targets_test))

    # Out
    # CV iteration: 1
    # Training set size: 625
    # Test set size: 70
    # --------------------------------------------------------------------------
    # SNIPPET 23

    scaler = StandardScaler()

    scaler.fit(features_train)

    features_train_norm = scaler.transform(features_train)
    features_test_norm = scaler.transform(features_test)

    # --------------------------------------------------------------------------
    # SNIPPET 24

    clf = LinearSVC(loss='hinge')

    # --------------------------------------------------------------------------
    # SNIPPET 25

    # Hyperparameter seach space
    param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

    # Gridsearch
    internal_cv = StratifiedKFold(n_splits=10)
    grid_cv = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=internal_cv,
                           scoring='balanced_accuracy',
                           verbose=1)

    # --------------------------------------------------------------------------
    # SNIPPET 26

    grid_result = grid_cv.fit(features_train_norm, targets_train)
    # Out
    # Fitting 10 folds for each of 8 candidates, totalling 80 fits
    # [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    # [Parallel(n_jobs=1)]: Done  80 out of  80 | elapsed:    8.3s finished
    # --------------------------------------------------------------------------
    # SNIPPET 27

    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))

    # Out
    # Best: 0.675791 using {'C': 0.125}
    # 0.673129 (0.076206) with: {'C': 0.015625}
    # 0.674068 (0.091944) with: {'C': 0.03125}
    # 0.668388 (0.089292) with: {'C': 0.0625}
    # 0.675791 (0.077299) with: {'C': 0.125}
    # 0.669378 (0.083826) with: {'C': 0.25}
    # 0.662557 (0.057900) with: {'C': 0.5}
    # 0.653957 (0.060696) with: {'C': 1}
    # 0.657501 (0.067748) with: {'C': 2}

    # --------------------------------------------------------------------------
    # SNIPPET 28

    best_clf = grid_cv.best_estimator_

    joblib.dump(best_clf, models_dir / ('classifier_%d.joblib' % i_fold))
    joblib.dump(scaler, models_dir / ('scaler_%d.joblib' % i_fold))

    # --------------------------------------------------------------------------
    # SNIPPET 29
    coef_cv[i_fold, :] = np.abs(best_clf.coef_)

    # --------------------------------------------------------------------------
    # SNIPPET 30
    target_test_predicted = best_clf.predict(features_test_norm)

    for row, value in zip(test_idx, target_test_predicted):
        predictions_df.iloc[row, predictions_df.columns.get_loc('predictions')] = value

    # --------------------------------------------------------------------------
    # SNIPPET 31

    print('Confusion matrix')
    cm = confusion_matrix(targets_test, target_test_predicted)
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    bac_test = balanced_accuracy_score(targets_test, target_test_predicted)
    sens_test = tp / (tp + fn)
    spec_test = tn / (tn + fp)

    print('Balanced accuracy: %.3f ' % bac_test)
    print('Sensitivity: %.3f ' % sens_test)
    print('Specificity: %.3f ' % spec_test)

    bac_cv[i_fold, :] = bac_test
    sens_cv[i_fold, :] = sens_test
    spec_cv[i_fold, :] = spec_test

# Out
# Confusion matrix
# [[31  6]
#  [10 23]]
# Balanced accuracy: 0.767
# Sensitivity: 0.697
# Specificity: 0.837
# --------------------------------------------------------------------------
# SNIPPET 32

print('CV results')
print('Bac: Mean(SD) = %.3f(%.3f)' % (bac_cv.mean(), bac_cv.std()))
print('Sens: Mean(SD) = %.3f(%.3f)' % (sens_cv.mean(), sens_cv.std()))
print('Spec: Mean(SD) = %.3f(%.3f)' % (spec_cv.mean(), spec_cv.std()))

# Out
# CV results
# Bac: Mean(SD) = 0.744(0.046)
# Sens: Mean(SD) = 0.718(0.078)
# Spec: Mean(SD) = 0.770(0.063)
# --------------------------------------------------------------------------
# SNIPPET 33

# Saving feature importance
mean_coef = np.mean(coef_cv, axis=0).reshape(1, -1)

coef_df = pd.DataFrame(data=mean_coef, columns=features_names.values)
coef_df.to_csv(experiment_dir / 'feature_importance.csv', index=False)

# Saving predictions
predictions_df.to_csv(experiment_dir / 'predictions.csv', index=True)

# Saving metrics
metrics = np.concatenate((bac_cv, sens_cv, spec_cv), axis=1)
metrics_df = pd.DataFrame(data=metrics, columns=['bac', 'sens', 'spec'])
metrics_df.index.name = 'CV iteration'
metrics_df.to_csv(experiment_dir / 'metrics.csv', index=True)

# -----------------------------------------------------------------------------
# SNIPPET 34

permutation_dir = experiment_dir / 'permutation'
permutation_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# SNIPPET 35

bac_from_model = bac_cv.mean()
sens_from_model = sens_cv.mean()
spec_from_model = spec_cv.mean()

# --------------------------------------------------------------------------
# SNIPPET 36

n_permutations = 5

bac_perm = np.zeros((n_permutations, 1))
sens_perm = np.zeros((n_permutations, 1))
spec_perm = np.zeros((n_permutations, 1))
coef_perm = np.zeros((n_permutations, len(features_names)))

# --------------------------------------------------------------------------
# SNIPPET 37
for i_perm in range(n_permutations):
    print('Permutation: %d' % (i_perm + 1))

    np.random.seed(i_perm)
    targets_permuted = np.random.permutation(targets)

    # Out
    # Permutation: 1
    # Permutation: 2
    # Permutation: 3
    # Permutation: 4
    # Permutation: 5
    # --------------------------------------------------------------------------
    # SNIPPET 38

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    bac_cv = np.zeros((n_folds, 1))
    sens_cv = np.zeros((n_folds, 1))
    spec_cv = np.zeros((n_folds, 1))
    coef_cv = np.zeros((n_folds, len(features_names)))

    for i_fold, (train_idx, test_idx) in enumerate(skf.split(features, targets_permuted)):
        features_train, features_test = features[train_idx], features[test_idx]
        targets_train, targets_test = targets_permuted[train_idx], targets_permuted[test_idx]

        scaler = StandardScaler()
        features_train_norm = scaler.fit_transform(features_train)
        features_test_norm = scaler.transform(features_test)

        clf = LinearSVC(loss='hinge')

        param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

        internal_cv = StratifiedKFold(n_splits=10)
        grid_cv = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               cv=internal_cv,
                               scoring='balanced_accuracy',
                               verbose=0)

        grid_result = grid_cv.fit(features_train_norm, targets_train)

        best_clf = grid_cv.best_estimator_

        coef_cv[i_fold, :] = np.abs(best_clf.coef_)

        target_test_predicted = best_clf.predict(features_test_norm)

        cm = confusion_matrix(targets_test, target_test_predicted)

        tn, fp, fn, tp = cm.ravel()

        bac_test = balanced_accuracy_score(targets_test, target_test_predicted)
        sens_test = tp / (tp + fn)
        spec_test = tn / (tn + fp)

        bac_cv[i_fold, :] = bac_test
        sens_cv[i_fold, :] = sens_test
        spec_cv[i_fold, :] = spec_test

    # --------------------------------------------------------------------------
    # SNIPPET 39

    np.save(permutation_dir / ('perm_test_bac_%03d.npy' % i_perm), bac_cv.mean())
    np.save(permutation_dir / ('perm_test_sens_%03d.npy' % i_perm), sens_cv.mean())
    np.save(permutation_dir / ('perm_test_spec_%03d.npy' % i_perm), spec_cv.mean())
    np.save(permutation_dir / ('perm_coef_%03d.npy' % i_perm), coef_cv.mean(axis=0))

    bac_perm[i_perm, :] = bac_cv.mean()
    sens_perm[i_perm, :] = sens_cv.mean()
    spec_perm[i_perm, :] = spec_cv.mean()
    coef_perm[i_perm, :] = coef_cv.mean(axis=0)

# --------------------------------------------------------------------------
# SNIPPET 40

# Get p_values from metrics
bac_p_value = (np.sum(bac_perm >= bac_from_model) + 1) / (n_permutations + 1)
sens_p_value = (np.sum(sens_perm >= sens_from_model) + 1) / (n_permutations + 1)
spec_p_value = (np.sum(spec_perm >= spec_from_model) + 1) / (n_permutations + 1)

print('BAC: p-value = %.3f' % bac_p_value)
print('SENS: p-value = %.3f' % sens_p_value)
print('SPEC: p-value = %.3f' % spec_p_value)

# Out

# --------------------------------------------------------------------------
# SNIPPET 41

# Get p_values from coef
coef_p_values = np.zeros((1, len(features_names)))
for i_feature in range(len(features_names)):
    coef_value_from_perm = coef_perm[:, i_feature]
    coef_value_from_model = mean_coef[0, i_feature]

    n_perm_better_model = np.sum(coef_value_from_perm >= coef_value_from_model)

    coef_p_values[0, i_feature] = (n_perm_better_model + 1) / (n_permutations + 1)

# --------------------------------------------------------------------------
# SNIPPET 42

# Saving
perm_metrics_df = pd.DataFrame(data={'metric': ['bac', 'sens', 'spec'],
                                     'value': [bac_from_model,
                                               sens_from_model,
                                               spec_from_model],
                                     'p_value': [bac_p_value,
                                                 sens_p_value,
                                                 spec_p_value]})

perm_metrics_df.to_csv(experiment_dir / 'metrics_permutation_pvalue.csv', index=False)

coef_df = pd.DataFrame(index=['coefficients', 'p value'],
                       data=np.concatenate((mean_coef, coef_p_values)),
                       columns=features_names)
coef_df.to_csv(experiment_dir / 'coef_permutation_pvalue.csv', index=True)
