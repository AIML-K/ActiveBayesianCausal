from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
import numpy as np
import math
from numpy import linalg as LA
import math


regression_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)


def thresh_SVD(X, threshold):
    """"Threshold SVD returns a smoothed matrix by removing singular directions with low singular values"""
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    small_sing_index = len(S)
    for i in range(0, len(S)):
        if S[i] ** 2 < threshold:
            small_sing_index = i
            break
    Uplus, Splus, Vhplus = U[:, 0:small_sing_index], S[0:small_sing_index], Vh[0:small_sing_index, :]
    Xplus = Uplus @ np.diag(Splus, k=0) @ Vhplus

    return Xplus


def normalize(p):
    """Normalize it to make it a probability distribution"""
    return p / p.sum()


def leverage_score(X, **qargs):
    """Compute leverage scores"""
    n, d = X.shape
    Ul, Sl, Vhl = np.linalg.svd(X, full_matrices=False)
    scores = [LA.norm(Ul[i]) ** 2 for i in range(0, n)]

    return scores


def lev_bernoulli_sampling(X, s, gamma):
    """"Bernoulli sampling based on leverage scores multiplied by the factor gamma"""
    n, d = X.shape
    lscores = np.array(leverage_score(X))
    lscores = np.array([min(1.0, lscores[i] * float(gamma)) for i in range(0, len(lscores))])

    sampled_rows = []

    for i in range(0, len(lscores)):
        if np.random.random() <= lscores[i]:
            sampled_rows.append(i)

    return sampled_rows, lscores


def sampled_linear_regression(X, sampled_rows, Y, s, pi):
    """Linear regression using sub-sampled matrix"""
    n, d = X.shape

    # S is a s x n matrix, D is a s x s matrix
    S = np.zeros((s, n))
    D = np.zeros((s, s))
    i = 0

    #scaling the matrix appropriately
    for row in sampled_rows:
        S[i][row] = 1
        D[i][i] = 1.0 / (math.sqrt(pi[row]))
        i += 1

    Xsampled = D @ S @ X
    Ysampled = D @ S @ Y

    # beta = np.linalg.lstsq(Xsampled, Ysampled, rcond=None)[0]

    return Xsampled, Ysampled


def ITE_estimator(nsamples, train_data_matrix, train_Y1, train_Y0, test_data_matrix, gamma=1.0, sampling=lev_bernoulli_sampling):
    srows_0, pi_0 = sampling(train_data_matrix, int(nsamples / 2.0), gamma)
    srows_1, pi_1 = sampling(train_data_matrix, int(nsamples / 2.0), gamma)
    srows_1_dup_rmv = []

    # Remove people sampled twice from the treatment group
    for i in range(0, len(srows_1)):
        if srows_1[i] in set(srows_0):
            continue
        else:
            srows_1_dup_rmv.append(srows_1[i])
    
    X0sampled, Y0sampled = sampled_linear_regression(train_data_matrix, srows_0, train_Y0, len(srows_0), pi_0)

    # Reweighting the new probabilities, as we removed the duplicates from treatment group
    pi_1_upd = [0.0 for i in range(0, len(pi_1))]
    for i in range(0, len(pi_1)):
        pi_1_upd[i] = pi_1[i] * (1-pi_1[i])

    # Solve linear regression for treatment group
    X1sampled, Y1sampled = sampled_linear_regression(train_data_matrix, srows_1_dup_rmv, train_Y1, len(srows_1_dup_rmv), pi_1_upd)  # changed

    gp0, gp1 = fit_gp(train_data_matrix[srows_0], train_data_matrix[srows_1_dup_rmv], train_Y0[srows_0], train_Y1[srows_1_dup_rmv])
    mu_hat, std_hat = predict_gp(gp0, gp1, test_data_matrix)
    mmd = compute_mmd_rbf(train_data_matrix[srows_0], train_data_matrix[srows_1_dup_rmv])

    return mu_hat, std_hat, mmd


@ignore_warnings(category=ConvergenceWarning)
def fit_gp(x0, x1, y0, y1):
    model0 = GaussianProcessRegressor(regression_kernel, normalize_y=True)
    model1 = GaussianProcessRegressor(regression_kernel, normalize_y=True)
    if len(y0) != 0:
        model0.fit(np.array(x0), np.array(y0))
    if len(y1) != 0:
        model1.fit(np.array(x1), np.array(y1))
    return model0, model1


def predict_gp(model0, model1, x):
    y0_hat, y0_std = model0.predict(x, return_std=True)
    y1_hat, y1_std = model1.predict(x, return_std=True)
    mu_hat = y1_hat - y0_hat
    std_hat = np.sqrt(y0_std ** 2 + y1_std ** 2)
    return mu_hat, std_hat


def compute_mmd_rbf(X, Y, gamma=1.0):
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def run_leverage(samples_list, train_data, test_data, true_ite):
    """Obtain estimators for various ITE estimators"""
    train_X, train_Y1, train_Y0 = np.array(train_data['X'].tolist()), np.array(train_data['Y1']), np.array(train_data['Y0'])
    test_X, test_Y1, test_Y0 = np.array(test_data['X'].tolist()), np.array(test_data['Y1']), np.array(test_data['Y0'])
    normalizer = Normalizer() # StandardScaler()
    train_X = normalizer.fit_transform(train_X)
    test_X = normalizer.transform(test_X)
    n, ndim = train_X.shape
    ate_error, ite_error, mu_list, std_list, mmd_list = [], [], [], [], []
    for nsamples in samples_list:
        gamma = (float(nsamples) / float(2.0 * ndim))
        threshold = gamma
        train_Xplus = thresh_SVD(train_X, threshold)
        test_Xplus = thresh_SVD(test_X, threshold)
        mu_hat, std_hat, mmd = ITE_estimator(nsamples, train_Xplus, train_Y1, train_Y0,
                                        test_Xplus, gamma, lev_bernoulli_sampling)
        mu_list.append(mu_hat)
        std_list.append(std_hat)
        mmd_list.append(mmd)
        ate_error.append((mu_hat.mean() - true_ite.mean())**2)
        ite_error.append(((mu_hat - true_ite) ** 2).mean()) 
    return ate_error, ite_error, mu_list, std_list, mmd_list

