from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from torch.linalg import solve_triangular, cholesky
from torch import cholesky_inverse
import torch
from scipy.optimize import minimize
import numpy as np

WARMUP = 1


class AbstractAgent:
    def __init__(self, train_X, test_X, regression_kernel, cls=False, svr=False):
        self.x0 = list()
        self.x1 = list()
        self.y0 = list()
        self.y1 = list()
        self.model0 = None
        self.model1 = None
        self.normalizer = StandardScaler()
        self.covariates = self.normalizer.fit_transform(train_X)
        if test_X is not None:
            self.test_covariates = self.normalizer.transform(test_X)
        self.num_data = len(self.covariates)
        self.unseen = list(range(self.num_data))
        self.cls = cls
        self.svr = svr
        self.regression_kernel = regression_kernel

    def pick_subject(self):
        raise NotImplementedError('Function is not implemented')

    def treatment(self):
        raise NotImplementedError('Function is not implemented')

    def bernoulli(self):
        p = np.random.uniform(0, 1)
        if p >= 0.5:
            return 1
        else:
            return 0

    def update(self, treatment, x, y, idx):
        if treatment:
            self.x1.append(self.normalizer.transform(x.reshape(1, -1)).tolist()[0])
            self.y1.append(y)
        else:
            self.x0.append(self.normalizer.transform(x.reshape(1, -1)).tolist()[0])
            self.y0.append(y)
        self.unseen.remove(idx)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self):
        if self.cls:
            self.model0 = GaussianProcessClassifier(kernel=self.regression_kernel, n_jobs=-1)
            self.model1 = GaussianProcessClassifier(kernel=self.regression_kernel, n_jobs=-1)
        if self.svr:
            self.model0 = SVR(kernel=self.regression_kernel)
            self.model1 = SVR(kernel=self.regression_kernel)
        else:
            self.model0 = GaussianProcessRegressor(kernel=self.regression_kernel, normalize_y=True)
            self.model1 = GaussianProcessRegressor(kernel=self.regression_kernel, normalize_y=True)
        self.model0.fit(np.array(self.x0), np.array(self.y0))
        self.model1.fit(np.array(self.x1), np.array(self.y1))
    
    def predict(self):
        if self.cls:
            y0_hat = self.model0.predict_proba(self.test_covariates)[:, 1]
            y1_hat = self.model1.predict_proba(self.test_covariates)[:, 1]
            y0_std = np.sqrt(y0_hat * (1 - y0_hat))
            y1_std = np.sqrt(y1_hat * (1 - y1_hat))
        if self.svr:
            y0_std, y1_std = 0, 0
            y0_hat = self.model0.predict(self.test_covariates)
            y1_hat = self.model1.predict(self.test_covariates)
        else:
            y0_hat, y0_std = self.model0.predict(self.test_covariates, return_std=True)
            y1_hat, y1_std = self.model1.predict(self.test_covariates, return_std=True)
        return y0_hat, y1_hat, y0_std, y1_std

    def cate_credible_interval(self):
        y0_hat, y1_hat, y0_std, y1_std = self.predict()
        mu_hat = y1_hat - y0_hat
        std_hat = np.sqrt(y0_std ** 2 + y1_std ** 2)
        return mu_hat, std_hat

    def ate_cate(self):
        if len(self.x0) >= WARMUP and len(self.x1) >= WARMUP:
            y0_hat, y1_hat, _, _ = self.predict()
            cate_hat = y1_hat - y0_hat
            ate_hat = cate_hat.mean()
        else:
            cate_hat = np.zeros(len(self.test_covariates))
            ate_hat = 0
        return ate_hat, cate_hat


class UncertaintyAwareAgent(AbstractAgent):
    def __init__(self, train_X, test_X, regression_kernel, uncertainty_kernel, sigma0=1, sigma1=1, cls=False, svr=False):
        super().__init__(train_X, test_X, regression_kernel, cls, svr)
        self.err0 = sigma0
        self.err1 = sigma1
        self.uncertainty_kernel = uncertainty_kernel
    
    def pick_subject(self):
        raise NotImplementedError('Function is not implemented')

    def treatment(self):
        if len(self.x0) <= WARMUP or len(self.x1) <= WARMUP:
            idx = np.random.choice(self.unseen)
            treatment = self.bernoulli()
        else:
            idx, treatment = self.pick_subject()
        return idx, treatment

    def compute_cholesky(self, seen, target):
        matrix = torch.tensor(self.uncertainty_kernel(seen, seen)).to('cuda')
        matrix = (matrix + ([self.err0, self.err1][target] ** 2) * torch.eye(matrix.shape[0]).to('cuda'))
        matrix = cholesky(matrix)
        return matrix

    def compute_tilde_star(self, seen, unseen):
        tilde = torch.tensor(self.uncertainty_kernel(seen, unseen)).to('cuda')
        star = torch.tensor(self.uncertainty_kernel(seen,  self.covariates)).to('cuda')
        return tilde, star

    def compute_predictive_covariance(self):
        x0_seen = self.x0
        x1_seen = self.x1
        x_unseen = self.covariates[self.unseen]
        with torch.no_grad():
            matrix0 = cholesky_inverse(self.compute_cholesky(x0_seen, 0))
            matrix1 = cholesky_inverse(self.compute_cholesky(x1_seen, 1))
            tilde0 = torch.tensor(self.uncertainty_kernel(x0_seen, x_unseen)).to('cuda')
            tilde1 = torch.tensor(self.uncertainty_kernel(x1_seen, x_unseen)).to('cuda')
            prior = torch.tensor(self.uncertainty_kernel(x_unseen, self.test_covariates)).to('cuda')
            tilde_test0 = torch.tensor(self.uncertainty_kernel(x0_seen, self.test_covariates)).to('cuda')
            tilde_test1 = torch.tensor(self.uncertainty_kernel(x1_seen, self.test_covariates)).to('cuda')
            cov0 = (prior - (tilde0.T.matmul(matrix0).matmul(tilde_test0))).mean(dim=1) ** 2
            cov1 = (prior - (tilde1.T.matmul(matrix1).matmul(tilde_test1))).mean(dim=1) ** 2
        cov = torch.vstack([cov0, cov1]).cpu().numpy()
        return cov

    def compute_predictive_variance(self):
        x0_seen = self.x0
        x1_seen = self.x1
        x_unseen = self.covariates[self.unseen]
        with torch.no_grad():
            matrix0 = cholesky_inverse(self.compute_cholesky(x0_seen, 0))
            matrix1 = cholesky_inverse(self.compute_cholesky(x1_seen, 1))
            tilde0 = torch.tensor(self.uncertainty_kernel(x0_seen, x_unseen)).to('cuda')
            tilde1 = torch.tensor(self.uncertainty_kernel(x1_seen, x_unseen)).to('cuda')
            var0 = 1 - (tilde0.T.matmul(matrix0) * tilde0.T).sum(dim=1)
            var1 = 1 - (tilde1.T.matmul(matrix1) * tilde1.T).sum(dim=1)
        var = torch.vstack([var0, var1]).cpu().numpy()
        return var

    def compute_variance_reduction(self):
        x0_seen = self.x0
        x1_seen = self.x1
        x_unseen = self.covariates[self.unseen]
        with torch.no_grad():
            matrix0 = self.compute_cholesky(x0_seen, 0)
            matrix1 = self.compute_cholesky(x1_seen, 1)
            tilde0, star0 = self.compute_tilde_star(x0_seen, x_unseen)
            tilde1, star1 = self.compute_tilde_star(x1_seen, x_unseen)
            true = torch.tensor(self.uncertainty_kernel(x_unseen, self.covariates)).to('cuda')

            left0 = solve_triangular(matrix0, tilde0, upper=False)
            right0 = solve_triangular(matrix0, star0, upper=False)
            left1 = solve_triangular(matrix1, tilde1, upper=False)
            right1 = solve_triangular(matrix1, star1, upper=False)

            num0 = ((left0.T.matmul(right0) - true) ** 2).sum(dim=1)
            denom0 = 1 - (left0 * left0).sum(dim=0)
            num1 = ((left1.T.matmul(right1) - true) ** 2).sum(dim=1)
            denom1 = 1 - (left1 * left1).sum(dim=0)

            var0 = num0 / denom0
            var1 = num1 / denom1
        var = torch.vstack([var0, var1]).cpu().numpy()
        return var


class NaiveAgent(AbstractAgent):
    def treatment(self):
        return None, self.bernoulli()


class MackayAgent(UncertaintyAwareAgent):
    def pick_subject(self):
        var = self.compute_predictive_variance()
        treatment, index = np.unravel_index(var.argmax(), var.shape)
        return self.unseen[index], treatment
    

class ACEAgent(UncertaintyAwareAgent):
    def pick_subject(self):
        var = self.compute_predictive_variance()
        cov, _ = self.compute_predictive_covariance()
        reduction = cov / var
        treatment, index = np.unravel_index(reduction.argmax(), var.shape)
        return self.unseen[index], treatment


class ABC3Agent(UncertaintyAwareAgent):
    def pick_subject(self):
        var = self.compute_variance_reduction()
        treatment, index = np.unravel_index(var.argmax(), var.shape)
        return self.unseen[index], treatment


class ABC3Opt(ABC3Agent):
    def __init__(self, train_X, test_X, regression_kernel, uncertainty_kernel, sigma0=1, sigma1=1, cls=False, svr=False, n_sampling=100):
        super().__init__(train_X, test_X, regression_kernel, uncertainty_kernel, sigma0, sigma1, cls, svr)
        self.n_sampling = n_sampling
        self.sampled = None

    def compute_tilde_star(self, seen, unseen):
        tilde = torch.tensor(self.uncertainty_kernel(seen, unseen)).to('cuda')
        star = torch.tensor(self.uncertainty_kernel(seen,  self.covariates[self.sampled])).to('cuda')
        return tilde, star

    def compute_variance_reduction(self, x_unseen):
        self.sampled = np.random.randint(low=0, high=self.num_data, size=self.n_sampling)
        if len(self.x0) <= self.n_sampling:
            x0_seen = self.x0
        else:
            x0_seen_idx = np.random.randint(0, high=len(self.x0), size=self.n_sampling)
            x0_seen = np.array(self.x0)[x0_seen_idx]
        if len(self.x1) <= self.n_sampling:
            x1_seen = self.x1
        else:
            x1_seen_idx = np.random.randint(0, high=len(self.x1), size=self.n_sampling)
            x1_seen = np.array(self.x1)[x1_seen_idx]
        with torch.no_grad():
            x_unseen = x_unseen.reshape(1, -1)
            matrix0 = self.compute_cholesky(x0_seen, 0)
            matrix1 = self.compute_cholesky(x1_seen, 1)
            tilde0, star0 = self.compute_tilde_star(x0_seen, x_unseen)
            tilde1, star1 = self.compute_tilde_star(x1_seen, x_unseen)
            true = torch.tensor(self.uncertainty_kernel(x_unseen, self.covariates[self.sampled])).to('cuda')

            left0 = solve_triangular(matrix0, tilde0, upper=False)
            right0 = solve_triangular(matrix0, star0, upper=False)
            left1 = solve_triangular(matrix1, tilde1, upper=False)
            right1 = solve_triangular(matrix1, star1, upper=False)

            num0 = ((left0.T.matmul(right0) - true) ** 2).sum(dim=1)
            denom0 = 1 - (left0 * left0).sum(dim=0)
            num1 = ((left1.T.matmul(right1) - true) ** 2).sum(dim=1)
            denom1 = 1 - (left1 * left1).sum(dim=0)

            var0 = num0 / denom0
            var1 = num1 / denom1
            treatment = int(var0.item() < var1.item())
            var = max(var0, var1)
        return var.item(), treatment

    def return_negative_variance(self, x_unseen):
        return - self.compute_variance_reduction(x_unseen)[0]

    def return_treatment(self, x_unseen):
        return self.compute_variance_reduction(x_unseen)[1]

    def pick_subject(self):
        x_unseen = self.covariates[self.unseen]
        x_mean = x_unseen.mean(axis=0)
        x_min = x_unseen.min(axis=0)
        x_max = x_unseen.max(axis=0)
        bound = np.concatenate([x_min.reshape(1,-1), x_max.reshape(1,-1)], axis=0).T
        opt = minimize(self.return_negative_variance, x0=x_mean, bounds=bound)
        similarity = self.uncertainty_kernel(self.covariates[self.unseen], opt.x.reshape(1,-1))
        index = self.unseen[similarity.argmax()]
        treatment = self.return_treatment(self.covariates[index])
        return index, treatment


class ABC3Sample(ABC3Opt):
    def compute_variance_reduction(self, x_unseen):
        self.sampled = np.random.randint(low=0, high=self.num_data, size=self.n_sampling)
        if len(self.x0) <= self.n_sampling:
            x0_seen = self.x0
        else:
            x0_seen_idx = np.random.randint(0, high=len(self.x0), size=self.n_sampling)
            x0_seen = np.array(self.x0)[x0_seen_idx]
        if len(self.x1) <= self.n_sampling:
            x1_seen = self.x1
        else:
            x1_seen_idx = np.random.randint(0, high=len(self.x1), size=self.n_sampling)
            x1_seen = np.array(self.x1)[x1_seen_idx]
        with torch.no_grad():
            matrix0 = self.compute_cholesky(x0_seen, 0)
            matrix1 = self.compute_cholesky(x1_seen, 1)
            tilde0, star0 = self.compute_tilde_star(x0_seen, x_unseen)
            tilde1, star1 = self.compute_tilde_star(x1_seen, x_unseen)
            true = torch.tensor(self.uncertainty_kernel(x_unseen, self.covariates[self.sampled])).to('cuda')

            left0 = solve_triangular(matrix0, tilde0, upper=False)
            right0 = solve_triangular(matrix0, star0, upper=False)
            left1 = solve_triangular(matrix1, tilde1, upper=False)
            right1 = solve_triangular(matrix1, star1, upper=False)

            num0 = ((left0.T.matmul(right0) - true) ** 2).sum(dim=1)
            denom0 = 1 - (left0 * left0).sum(dim=0)
            num1 = ((left1.T.matmul(right1) - true) ** 2).sum(dim=1)
            denom1 = 1 - (left1 * left1).sum(dim=0)

            var0 = num0 / denom0
            var1 = num1 / denom1
            var = torch.vstack([var0, var1]).cpu().numpy()
        return var

    
    def pick_subject(self):
        unseen_sampled = np.random.randint(low=0, high=len(self.unseen), size=self.n_sampling)
        unseen_sampled = np.array(self.unseen)[unseen_sampled]
        var = self.compute_variance_reduction(self.covariates[unseen_sampled])
        treatment, index = np.unravel_index(var.argmax(), var.shape)
        return unseen_sampled[index], treatment
