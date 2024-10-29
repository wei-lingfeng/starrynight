import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(X, y, yerr):
    """Weighted linear regression

    Parameters
    ----------
    X : array-like of shape (n, m_features)
        Training data.
    y : array-like of shape (n,)
        Target values.
    yerr : array-like of shape (n,)
        Individual weights for each sample.
    """
    weight = 1/yerr**2
    W = np.diag(weight)
    regr = LinearRegression().fit(X, y, sample_weight=weight)
    common = np.linalg.inv(X.T @ W @ X) @ X.T @ W
    beta_hat = common @ y
    var_beta_hat = common @ np.diag(yerr**2) @ common.T
    return regr, var_beta_hat

if __name__ == '__main__':
    np.random.seed(1024)
    # y = 2X + 3
    X = np.array([1, 2, 3, 4, 5])
    y = 2*X + 3 + np.random.normal(0, 2, len(X))
    yerr = np.array([1.5, 0.2, 1.2, 0.8, 1.])
    X = sm.add_constant(X)
    
    mod_wls = sm.WLS(y, X, weights=1/yerr**2)
    res_wls = mod_wls.fit()
    print(res_wls.summary())
    
    # regr = LinearRegression().fit(X[:, 1].reshape(-1, 1), y, sample_weight=1/yerr**2)
    regr, var_beta_hat = linear_regression(X, y, yerr)
    
    Xs = sm.add_constant(np.array([X[:, 1].min(), X[:, 1].max()]))
    fig, ax = plt.subplots()
    ax.errorbar(X[:, 1], y, yerr=yerr, fmt='.')
    ax.plot(Xs[:, 1], regr.predict(Xs), lw=2, label='Linear Regression')
    ax.plot(Xs[:, 1], res_wls.predict(Xs), lw=2, ls='--', label='WLS')
    plt.legend()
    plt.show()