import numpy as np
import warnings
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


# ============================================================
#  Newton's Method (Bounded) with Backtracking Line Search
# ============================================================

def newtonsMethodBounded(objfunc, x_0, x_min, x_max, options):
    x_i = x_0
    lv_newton = 1

    max_iter = options.get('max_iter', 15)
    max_ls_iter = options.get('max_ls_iter', 15)
    tau_0 = options.get('tau_0', 0.1)
    tau_shrink = options.get('tau', 0.5)
    min_delta = options.get('min_delta', 1e-3)

    f_x, df_dx = objfunc(x_i)

    while lv_newton <= max_iter:
        lv_ls = 1
        ls_tau = tau_0

        x_test = x_i
        f_x_test = f_x
        df_dx_test = df_dx

        line_search_converged = False

        while lv_ls <= max_ls_iter:
            if abs(df_dx) < 1e-10:
                step = 0
            else:
                step = f_x / df_dx

            x_test = x_i - ls_tau * step
            f_x_test, df_dx_test = objfunc(x_test)

            if f_x_test < f_x and x_min <= x_test <= x_max:
                x_i = x_test
                line_search_converged = True
                break
            else:
                ls_tau = tau_shrink * ls_tau
                lv_ls += 1

        if not line_search_converged:
            warnings.warn('Line search failed to converge.')

        if abs(f_x) < 1e-10:
            rel_change = 0
        else:
            rel_change = (f_x - f_x_test) / f_x

        if rel_change <= min_delta:
            break
        else:
            f_x = f_x_test
            df_dx = df_dx_test
            lv_newton += 1

    if lv_newton > max_iter:
        warnings.warn("Newton's method failed to converge.")

    return x_i


# ============================================================
#  Adaptive Loss & getAlphaStar
# ============================================================

def _adaptiveLoss(epsilon, alpha):
    if abs(alpha - 2) < 1e-8:
        rho = (epsilon**2) / 2
    elif abs(alpha) < 1e-8:
        rho = np.log((epsilon**2) / 2 + 1)
    elif alpha == float('-inf'):
        rho = 1 - np.exp(-(epsilon**2) / 2)
    else:
        rho = (abs(alpha - 2) / alpha) * \
              (((epsilon**2) / abs(alpha - 2) + 1)**(alpha/2) - 1)
    return rho


def _drho_dalpha(epsilon, alpha):
    if abs(alpha) < 1e-10:
        return (_drho_dalpha(epsilon, 1e-9) + _drho_dalpha(epsilon, -1e-9)) / 2
    elif abs(alpha - 2) < 1e-10:
        return (_drho_dalpha(epsilon, 2 - 1e-9) + _drho_dalpha(epsilon, 2 + 1e-9)) / 2

    b = (epsilon**2) / abs(alpha - 2) + 1

    term_1 = (abs(alpha - 2) / alpha) * (b**(alpha/2)) * \
             (0.5 * np.log(b) - (alpha * (epsilon**2) * (alpha - 2)) / (2 * b * abs(alpha - 2)**3))
    term_2 = -(abs(alpha - 2) / (alpha**2)) * (b**(alpha/2) - 1)
    term_3 = ((alpha - 2) * (b**(alpha/2) - 1)) / (alpha * abs(alpha - 2))

    return term_1 + term_2 + term_3


def _objFuncAlpha(epsilon, alpha, tau_min, tau_max):
    N = epsilon.size

    integrand = lambda e: np.exp(-_adaptiveLoss(np.array([e]), alpha)[0])
    z_tilde, _ = quad(integrand, tau_min, tau_max)

    loss_vals = _adaptiveLoss(epsilon, alpha)
    Lambda = N * np.log(z_tilde) + np.sum(loss_vals)

    integrand_grad = lambda e: -np.exp(-_adaptiveLoss(np.array([e]), alpha)[0]) * \
                               _drho_dalpha(np.array([e]), alpha)[0]
    int_n, _ = quad(integrand_grad, tau_min, tau_max)

    drho_vals = _drho_dalpha(epsilon, alpha)
    dLambda_dalpha = (N / z_tilde) * int_n + np.sum(drho_vals)

    return Lambda, dLambda_dalpha


def getAlphaStar(epsilon, lower_bound, upper_bound):
    options = {
        'max_iter': 15,
        'min_delta': 1e-3,
        'tau_0': 0.1,
        'tau': 0.5,
        'max_ls_iter': 15,
    }
    objfunc = lambda alpha: _objFuncAlpha(epsilon, alpha, lower_bound, upper_bound)
    alpha_star = newtonsMethodBounded(objfunc, -2, -10, 2, options)
    return alpha_star


# ============================================================
#  GNC-ADAPT Weight Function
# ============================================================

def gnc_adapt(residuals, params):
    mu = params['mu']
    alpha_star = params['alphaStar']

    alpha = (alpha_star * mu + 2) / (mu + 1)

    w = np.ones(residuals.shape)

    if alpha <= 2:
        for k in range(len(residuals)):
            res_sq = residuals[k]**2
            if abs(alpha - 2) < 1e-8:
                w[k] = 1
            elif abs(alpha) < 1e-8:
                w[k] = 1.0 / (0.5 * res_sq + 1)
            elif alpha == float('-inf'):
                w[k] = np.exp(-0.5 * res_sq)
            else:
                w[k] = (res_sq / abs(alpha - 2) + 1)**(alpha/2 - 1)
    else:
        raise ValueError("Invalid mu/alpha value encountered in GNC-ADAPT")

    return w


def gnc_adapt_vd(residuals, params):
    """GNC-ADAPT with Variance Reduction — weight computation delegates to gnc_adapt."""
    return gnc_adapt(residuals, params)


# ============================================================
#  GNC Parameter Initialisation
# ============================================================

def gncParams(rlf, residuals, tau, alpha=None):
    func_name = rlf.__name__.upper()

    params = {}

    if func_name in ('GNC_ADAPT', 'GNC_ADAPT_MOM', 'GNC_ADAPT_VD'):
        params['mu'] = max(1.0 / (2 * np.max(residuals) - 1), 1e-3)
        params['update'] = lambda x, c=None: x * 1.4
        params['converged'] = lambda x: x > 1e3

        if alpha is None:
            params['alphaStar'] = getAlphaStar(residuals, -tau, tau)
        else:
            params['alphaStar'] = alpha
        params['tau'] = tau
    else:
        raise ValueError(f"GNC parameters not implemented for function: {func_name}")

    return params


# ============================================================
#  Core Estimators
# ============================================================

def robustMeanEstimateGNC(data, lossFunc, mu_init, opt_params, params):
    """Robust mean estimation using GNC (inner solver for each group)."""
    mu = mu_init.copy()
    iteration = 0
    costHistory = []
    gnc_params = {}
    weights = None

    while iteration < opt_params['maxIter']:
        residuals = np.linalg.norm(data - mu, axis=0)**2

        if iteration == 0:
            gnc_params = gncParams(lossFunc, residuals, opt_params['tau'], None)

        weights = lossFunc(residuals, gnc_params)

        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            w_sum = 1e-10
        mu_new = np.sum(data * weights, axis=1, keepdims=True) / w_sum

        currentCost = np.sum(weights * residuals)
        costHistory.append(currentCost)

        if not gnc_params['converged'](gnc_params['mu']):
            gnc_params['mu'] = gnc_params['update'](gnc_params['mu'], None)

        if iteration > 0:
            if np.linalg.norm(mu_new - mu) < opt_params['costThreshold']:
                status = 'Converged'
                break
            if abs(costHistory[-2] - currentCost) < opt_params['costThreshold']:
                status = 'CostConverged'
                break

        mu = mu_new
        iteration += 1

        if opt_params.get('verbose') and iteration % 10 == 0:
            print(f"  Iter {iteration}: cost = {currentCost:.4e}, mu = {gnc_params['mu']:.2e}")

    if iteration >= opt_params['maxIter']:
        status = 'MaxIter'

    return {
        'mu_est': mu,
        'iter': iteration,
        'cost': costHistory[-1] if costHistory else 0,
        'costHistory': costHistory,
        'weights': weights,
        'status': status,
    }


# ============================================================
#  Quantile Aggregation
# ============================================================

def quantile_loss(u, tau):
    """ρ_τ(u) = u(τ - 1{u<0})"""
    u = np.asarray(u)
    return np.where(u >= 0, tau * u, (1 - tau) * (-u))


def quantile_aggregation(group_estimates, K):
    """
    Aggregate group estimates using K quantiles.
    Minimizes: sum_{k=1}^K ρ_{τ_k}(y_k − μ)  where τ_k = k/(K+1)
    """
    dim, num_groups = group_estimates.shape
    taus = np.array([(k + 1) / (K + 1) for k in range(K)])

    mu_est = np.zeros((dim, 1))

    for d in range(dim):
        y = group_estimates[d, :]

        def objective(mu_val):
            residuals = y - mu_val
            return np.sum([quantile_loss(residuals, tau) for tau in taus])

        y_min, y_max = np.min(y), np.max(y)
        result = minimize_scalar(objective, bounds=(y_min, y_max), method='bounded')
        mu_est[d, 0] = result.x

    return mu_est


# ============================================================
#  GNC-ADAPT-VD  (top-level API)
# ============================================================

def robustMeanEstimateVD(data, mu_init, opt_params, params):
    """
    Robust mean estimation using GNC-ADAPT with Variance Reduction
    (Quantile Aggregation).

    Parameters
    ----------
    data : ndarray, shape (dim, N)
        Input measurements.
    mu_init : ndarray, shape (dim, 1)
        Initial mean estimate.
    opt_params : dict
        Must contain at least:
          - maxIter      : int   — max IRLS iterations per group
          - tau          : float — inlier noise bound
          - costThreshold: float — convergence tolerance
        Optional:
          - K_quantiles  : int   — number of groups / quantile levels (default 20)
          - verbose      : bool
    params : dict
        Must contain:
          - dim : int — data dimensionality

    Returns
    -------
    dict with keys: mu_est, iter, cost, status, group_estimates, K_quantiles
    """
    N = data.shape[1]
    K = opt_params.get('K_quantiles', 20)
    group_size = N // K

    if opt_params.get('verbose'):
        print(f"  Using {K} groups with quantile aggregation, ~{group_size} samples per group")

    group_estimates = np.zeros((params['dim'], K))
    group_costs = []
    group_iters = []

    for k in range(K):
        start_idx = k * group_size
        end_idx = N if k == K - 1 else (k + 1) * group_size

        group_data = data[:, start_idx:end_idx]
        group_init = np.mean(group_data, axis=1, keepdims=True)

        if opt_params.get('verbose') and k % max(1, K // 5) == 0:
            print(f"  Processing group {k+1}/{K}...")

        res = robustMeanEstimateGNC(group_data, gnc_adapt_vd, group_init, opt_params, params)

        group_estimates[:, k] = res['mu_est'].flatten()
        group_costs.append(res['cost'])
        group_iters.append(res['iter'])

    mu_est = quantile_aggregation(group_estimates, K)

    return {
        'mu_est': mu_est,
        'iter': sum(group_iters),
        'cost': np.mean(group_costs),
        'status': 'VD',
        'group_estimates': group_estimates,
        'K_quantiles': K,
    }


# ============================================================
#  Demo / Self-test
# ============================================================

if __name__ == "__main__":
    np.random.seed(100)

    params = {
        'N': 1000,
        'outlierRatio': 0.5,
        'noiseSigma': 1.0,
        'outlierSigma': 10.0,
        'dim': 1,
        'mu_true': np.array([[2.0]]),
    }

    print('Generating data...')
    print(f"  Dimension: {params['dim']}")
    print(f"  Total measurements: {params['N']}")
    print(f"  Outlier ratio: {params['outlierRatio']*100:.1f}%")

    numInliers = int(round(params['N'] * (1 - params['outlierRatio'])))
    inliers = params['mu_true'] + params['noiseSigma'] * np.random.randn(params['dim'], numInliers)

    numOutliers = params['N'] - numInliers
    outliers = params['mu_true'] + params['outlierSigma'] * np.random.randn(params['dim'], numOutliers)

    data = np.hstack((inliers, outliers))
    perm = np.random.permutation(params['N'])
    data = data[:, perm]

    opt_params = {
        'maxIter': 50,
        'tau': 3 * params['noiseSigma'],
        'costThreshold': 1e-6,
        'verbose': True,
        'K_quantiles': 20,
    }

    mu_init = np.mean(data, axis=1, keepdims=True)
    print(f"Initial guess: {mu_init.flatten()}")
    print(f"True mean: {params['mu_true'].flatten()}")
    print(f"Initial Error: {np.linalg.norm(mu_init - params['mu_true']):.4f}\n")

    print("=== Running GNC-ADAPT-VD ===")
    result = robustMeanEstimateVD(data, mu_init, opt_params, params)

    err = np.linalg.norm(result['mu_est'] - params['mu_true'])
    print(f"\n{'='*40}")
    print(f"Status     : {result['status']}")
    print(f"Iterations : {result['iter']}")
    print(f"Cost       : {result['cost']:.4e}")
    print(f"Estimate   : {result['mu_est'].flatten()}")
    print(f"True mean  : {params['mu_true'].flatten()}")
    print(f"Error      : {err:.4f}")
