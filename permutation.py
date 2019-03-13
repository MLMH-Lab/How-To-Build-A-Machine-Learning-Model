"""
Permutation script.

Reference for the p_value from coefficients.
Mourao-Miranda, Janaina, et al. "Classifying brain states and determining the discriminating activation patterns: support vector machine on functional MRI data." NeuroImage 28.4 (2005): 980-995.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')


experiment_name = 'example'
results_dir = Path('./results')
results_dir.mkdir(exist_ok=True)

experiment_dir = results_dir / experiment_name
results_dir.mkdir(exist_ok=True)

permutation_dir = experiment_dir / 'permutation'
permutation_dir.mkdir(exist_ok=True)

dataset_df = pd.read_csv("./Chapter_19_data.csv")
dataset_df = dataset_df.dropna()
features_name = dataset_df.columns[4:]

y = dataset_df['label'].values.astype('float32')
X = dataset_df[features_name].values.astype('float32')

# O X e o y daqui precisa ser igual aos utilizados no codigo normal, ou seja, precisa aplicar a mesma limpeza de dados,
# como, por exemplo, remocao de NaNs, tratamento para confoundings, feature selection .
# Por esse motivo, talvez vale a pena ter salvo anteriormente o dataset limpo e so carregar ele aqui.
seed = 1

n_permutations = 2

perm_test_bac_list = []
perm_test_sens_list = []
perm_test_spec_list = []
perm_abs_coef_list = []

for i_perm in range(n_permutations):

    # Define a random seed para a permutacao como sendo o i_perm (para haver uma seed diferente a cada iteracao)
    np.random.seed(i_perm)
    y_permuted = np.random.permutation(y)

    # A validacao cruzada daqui precisa ser igual a que foi realizada no codigo normal.
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_test_bac = np.zeros((n_folds,))
    cv_test_sens = np.zeros((n_folds,))
    cv_test_spec = np.zeros((n_folds,))
    cv_error_rate = np.zeros((n_folds,))
    cv_coefficients = []

    for i_fold, (train_index, test_index) in enumerate(skf.split(X, y_permuted)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_permuted[train_index], y_permuted[test_index]

        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        clf = LinearSVC(loss='hinge')

        param_grid = {'C': [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1]}

        internal_cv = StratifiedKFold(n_splits=10)
        grid_cv = GridSearchCV(estimator=clf, param_grid=param_grid, cv=internal_cv, scoring='balanced_accuracy', verbose=1)

        grid_result = grid_cv.fit(X_train_normalized, y_train)

        best_clf = grid_cv.best_estimator_

        y_predicted = best_clf.predict(X_test_normalized)

        cm = confusion_matrix(y_test, y_predicted)

        tn, fp, fn, tp = cm.ravel()
        test_bac = balanced_accuracy_score(y_test, y_predicted)
        test_sens = tp / (tp + fn)
        test_spec = tn / (tn + fp)

        cv_test_bac[i_fold] = test_bac
        cv_test_sens[i_fold] = test_sens
        cv_test_spec[i_fold] = test_spec

        coef_abs_value = np.abs(best_clf.coef_.squeeze())
        cv_coefficients.append(coef_abs_value)

    perm_test_bac = np.mean(cv_test_bac, axis=0)
    perm_test_sens = np.mean(cv_test_sens, axis=0)
    perm_test_spec = np.mean(cv_test_spec, axis=0)

    cv_coefficients = np.asarray(cv_coefficients, dtype='float32')
    perm_abs_coef = np.mean(cv_coefficients, axis=0)

    np.save(permutation_dir / ('perm_test_bac_%3d.npy'%i_perm), perm_test_bac)
    np.save(permutation_dir / ('perm_test_sens_%3d.npy'%i_perm), perm_test_sens)
    np.save(permutation_dir / ('perm_test_spec_%3d.npy'%i_perm), perm_test_spec)
    np.save(permutation_dir / ('perm_coef_%3d.npy' % i_perm), perm_abs_coef)

    perm_test_bac_list.append(perm_test_bac)
    perm_test_sens_list.append(perm_test_sens)
    perm_test_spec_list.append(perm_test_spec)
    perm_abs_coef_list.append(perm_abs_coef)

perm_test_bac_list = np.asarray(perm_test_bac_list, dtype='float32')
perm_test_sens_list = np.asarray(perm_test_sens_list, dtype='float32')
perm_test_spec_list = np.asarray(perm_test_spec_list, dtype='float32')
perm_abs_coef_list = np.asarray(perm_abs_coef_list, dtype='float32')

# Fazer o load ou conseguir de alguma maneira as metricas e os coeficientes do SVM nao permutado
bac = 0.90
sens = 0.97
spec = 0.99992
coeffs = np.ones((1, len(features_name)))



# Get p_values from metrics
pvalue_test_bac = (np.sum(perm_test_bac_list >= bac) + 1.0) / (n_permutations + 1)
pvalue_test_sens = (np.sum(perm_test_sens_list >= sens) + 1.0) / (n_permutations + 1)
pvalue_test_spec = (np.sum(perm_test_spec_list >= spec) + 1.0) / (n_permutations + 1)

# Get p_values from coef
coef_pvalues = []
for i_feature in range(len(features_name)):
    pvalue_temp = (np.sum(perm_abs_coef_list[:, i_feature] >= coeffs[0, i_feature]) + 1.0) / (n_permutations + 1)
    coef_pvalues.append(pvalue_temp)
coef_pvalues = np.asarray(coef_pvalues, dtype='float32')
coef_pvalues = coef_pvalues[np.newaxis,]


# Saving
metrics_df = pd.DataFrame(data={'metric': ['bac', 'sens', 'spec'],
                                'p value': [pvalue_test_bac,
                                            pvalue_test_sens,
                                            pvalue_test_spec]})

metrics_df.to_csv(experiment_dir/'metrics_pvalue.csv', index=False)


coef_df = pd.DataFrame( index=['coefficients', 'p value'], data=np.concatenate((coeffs,coef_pvalues)), columns=features_name)
coef_df.to_csv(experiment_dir/'coef_pvalue.csv', index=True)