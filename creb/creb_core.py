import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import logging
from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose: bool = False, log_level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('creb')
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.propagate = False  
    
    return logger

def get_logger() -> logging.Logger:
    """Get the creb logger instance."""
    return logging.getLogger('creb')

# ============================================================================
# STATISTICAL OPERATIONS
# ============================================================================

def fit_covariates(X: np.ndarray, Z: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Fit linear regression coefficients."""
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    XtX = X.T @ X
    XtZ = X.T @ Z
    B = np.linalg.solve(XtX, XtZ)
    
    logger.info(f"Fitted regression coefficients: shape {B.shape}")
    logger.debug(f"Coefficient matrix condition number: {np.linalg.cond(XtX):.2e}")
    return B


def build_design_matrix(covars: pd.DataFrame, 
                       covariate_cols: List[str],
                       verbose: bool = False) -> Tuple[np.ndarray, List[str], Optional[List[str]]]:
    """Build design matrix with intercept + covariates"""

    ## the intercept should handle the baseline signal value for each feature
    ## doing all linear fit here. non-linear covar (covar that have known non-linear relationship with features)
    ## are not supported here. this non-linear relationship could be fitted with spline. 

    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    X_list = []
    covar_names = []
    
    # start with intercept
    X_list.append(np.ones(len(covars)))
    covar_names.append("Intercept")
    
    # non-site covariates
    for col in covariate_cols:
        try:
            values = pd.to_numeric(covars[col], errors='raise').values
            X_list.append(values)
            covar_names.append(col)
        except (ValueError, TypeError):
            raise ValueError(f"Covariate '{col}' cannot be converted to numeric")  
    X = np.column_stack(X_list) if len(X_list) > 1 else X_list[0].reshape(-1, 1)
    
    logger.info(f"Built design matrix: shape {X.shape}, variables: {covar_names}")
    return X, covar_names


def aggregate_site_bins(residuals: np.ndarray, sites: np.ndarray,
                        verbose: bool = False) -> Dict[str, np.ndarray]:
    """Aggregate residuals into site-level bins."""
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)

    unique_sites = np.unique(sites)
    n_bins = len(unique_sites)
    d_features = residuals.shape[1]
    
    logger.info(f"Creating {n_bins} site bins for sites: {unique_sites}")
    
    # initialize output arrays
    n_bin = np.zeros(n_bins, dtype=np.int64)
    site_bin = np.array(unique_sites, dtype=object)
    mu_bin = np.zeros((n_bins, d_features), dtype=np.float32)
    sigma2_bin = np.zeros((n_bins, d_features), dtype=np.float32)
    
    # calculate feature mean and var for each site
    for i, site in enumerate(unique_sites):
        mask = (sites == site)
        R_bin = residuals[mask, :]
        n = R_bin.shape[0]
        
        if n > 0:
            n_bin[i] = n
            # Mean residual per feature (site effect)
            mu = R_bin.mean(axis=0)
            mu_bin[i, :] = mu.astype(np.float32)

            # SST per feature (within-site variance)
            sst = np.sum((R_bin - mu[None, :]) ** 2, axis=0)
            sigma2 = sst / np.maximum(n - 1, 1)
            sigma2_bin[i, :] = sigma2.astype(np.float32)
            
            logger.info(f"Site {site}: {n} subjects, mu_bin range [{mu.min():.4f}, {mu.max():.4f}]")
    
    return n_bin, site_bin, mu_bin, sigma2_bin

# ============================================================================
# EMPIRICAL BAYES FUNCTIONS  
# ============================================================================

def estimate_priors_joint(site_stats: Dict[str, Dict[str, np.ndarray]],
                                    feature_names: List[str],
                                    verbose: bool = False) -> Dict[str, float]:
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)

    d = len(feature_names)
    logger.info(f"Fitting reference priors for {d} features all sites")
    
    # all site statistics
    all_rbar = []
    all_sig2 = []
    all_sig2_over_n = []

    for st in site_stats.values():
        n = int(st["n"])
        rbar = st["mean"]
        sig2_hat = st["sigma2"]
        sig2_over_n = sig2_hat / max(n, 1.0)

        all_rbar.append(rbar)
        all_sig2.append(sig2_hat)
        all_sig2_over_n.append(sig2_over_n)

    rbar_flat = np.concatenate(all_rbar, axis=0)
    sig2_flat = np.concatenate(all_sig2, axis=0)
    sig2_over_n_flat = np.concatenate(all_sig2_over_n, axis=0)
    
    tiny = 1e-8

    # variance prior (inverse-gamma matched to mean/var)
    m = float(sig2_flat.mean())
    v = float(sig2_flat.var(ddof=1)) if sig2_flat.size > 1 else (m * 0.1) ** 2
    if v < tiny:
        a0 = 3.0
        b0 = m * (a0 - 1.0)
    else:
        a0 = (m ** 2) / v + 2.0
        if a0 <= 2.0:
            a0 = 3.0
        b0 = m * (a0 - 1.0)

    # mean prior strength (remove measurement noise from Var(rbar))
    var_rbar = float(rbar_flat.var(ddof=1)) if rbar_flat.size > 1 else 0.0
    mean_sig2_over_n = float(sig2_over_n_flat.mean()) if sig2_over_n_flat.size > 0 else 0.0
    var_gamma_est = max(var_rbar - mean_sig2_over_n, tiny)

    E_sigma2 = b0 / (a0 - 1.0)
    kappa0 = E_sigma2 / var_gamma_est if var_gamma_est > tiny else 1e6

    priors = {"kappa0": float(kappa0), "a0": float(a0), "b0": float(b0)}
    logger.info(f"Priors: kappa0={kappa0:.2e}, a0={a0:.2f}, b0={b0:.6f}")
    return priors

def posterior_updates_joint(rbar: np.ndarray,
                            sst: np.ndarray,
                            n: int,
                            priors: Dict[str, Dict[str, float]],
                            verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    d = rbar.shape[0]
    a0 = priors["a0"]
    b0 = priors["b0"] 
    k0 = priors["kappa0"]
    
    
    kn = k0 + float(n)
    mu_n = (float(n) * rbar) / kn
    an = a0 + 0.5 * float(n)
    bn = b0 + 0.5 * sst + 0.5 * (k0 * float(n) / kn) * (rbar ** 2)

    gamma_hat = mu_n
    an_m1 = np.maximum(an - 1.0, 1e-6)
    delta_hat = np.sqrt(bn / an_m1)

    logger.debug(f"Posterior shrinkage: gamma range [{gamma_hat.min():.4f}, {gamma_hat.max():.4f}], "
                f"delta range [{delta_hat.min():.4f}, {delta_hat.max():.4f}]")
    return gamma_hat, delta_hat


def estimate_priors_iterative(bins_site: np.ndarray,
                          bins_n: np.ndarray,
                          bins_mu: np.ndarray, 
                          bins_sigma2: np.ndarray, 
                          verbose: bool = False) -> Dict[str, float]:
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)

    site_gammas = bins_mu  
    site_ns = bins_n
    site_sigmas2 = bins_sigma2

    # mu_0 and tau2_0 are directly estimated from all features and all site
    # not only are we using all features info, we are also using all site info
    # the priors are estimated for all bundle generation site, not each site has its set of priors
    all_gammas = site_gammas.flatten()
    mu_0 = np.mean(all_gammas) 
    tau2_0 = np.var(all_gammas, ddof=1)  

    all_sigmas = site_sigmas2.flatten() 

    # alpha and beta are estiamted from moments of inverse gamma distribution
    def global_aprior(delta_values):
        m = np.mean(delta_values)
        s2 = np.var(delta_values, ddof=1)
        return (2 * s2 + m**2) / s2

    def global_bprior(delta_values):
        m = np.mean(delta_values)
        s2 = np.var(delta_values, ddof=1)
        return (m * s2 + m**3) / s2

    a_0 = global_aprior(all_sigmas) 
    b_0 = global_bprior(all_sigmas) 

    if verbose:
        logger.info(f"Estimated priors: mu_0={mu_0:.4f}, tau2_0={tau2_0:.4f}, a_0={a_0:.4f}, b_0={b_0:.4f}")
    
    return {'mu_0': mu_0, 'tau2_0': tau2_0, 'a_0': a_0, 'b_0': b_0}


def posterior_updates_iterative(site_gamma_raw: np.ndarray,
                     site_sst_raw: np.ndarray,
                     n_site: int,
                     priors: Dict[str, float],
                     R_site: np.ndarray,
                     max_iter: int = 100,
                     tol: float = 1e-6,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    mu_0 = priors["mu_0"]
    tau2_0 = priors["tau2_0"]
    alpha_0 = priors["a_0"]
    beta_0 = priors["b_0"]
    
    ## gamma old and sigma2 old are raw estimation from the data of the site that we are trying to update
    
    gamma_old = site_gamma_raw.copy()
    sigma2_old = site_sst_raw / max(n_site - 1, 1) 
    
    n_features = len(site_gamma_raw)
    if verbose:
        logger.info(f"Starting iterative updates for {n_features} features")
        logger.info(f"Global priors: mu_0={mu_0:.4f}, tau2_0={tau2_0:.4f}, alpha_0={alpha_0:.4f}, beta_0={beta_0:.4f}")
    
    # In the iterative update, we seperately update for mean and variance. we first update mean assuming we know variance
    # we then update the variance using the newly estiamted mean. this iteration continue until we reach convergence standard 
    # or if the max iteration criteria is reached. 
    for iteration in range(max_iter):
        precision_prior = 1 / tau2_0
        precision_data = n_site / sigma2_old  

        # update posterior for mean assume the variance is fixed
        tau2_post = 1 / (precision_prior + precision_data) # posterior variance 
        mu_post = tau2_post * (precision_prior * mu_0 + precision_data * site_gamma_raw)
        
        # use mean of posterior distribution as the estimated value of new gamma
        gamma_new = mu_post 

        # update alpha and beta assuming have gamma fixed 
        alpha_post = alpha_0 + n_site / 2.0 
        sst_adjusted = np.sum((R_site - gamma_new[None, :])**2, axis=0)
        beta_post = beta_0 + 0.5 * sst_adjusted 

        # estimate new sigma base on new alpha and beta
        if alpha_post > 1:
            sigma2_new = beta_post / (alpha_post - 1) 
        else:
            sigma2_new = beta_post 
        
        ## convergence check
        gamma_change = (np.abs(gamma_new - gamma_old) / gamma_old).max()
        sigma2_change = (np.abs(sigma2_new - sigma2_old) / sigma2_old).max()
        
        max_change = max(gamma_change, sigma2_change)
        
        if verbose and iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: max_change = {max_change:.2e}")
        
        if max_change < tol:
            if verbose:
                logger.info(f"Converged after {iteration + 1} iterations")
            break
            
        gamma_old = gamma_new.copy()
        sigma2_old = sigma2_new.copy()

    else:
        if verbose:
            logger.warning(f"Did not converge after {max_iter} iterations (max_change = {max_change:.2e})")
    return gamma_new, np.sqrt(sigma2_new)


# ============================================================================
# DATA FORMAT VALIDATION
# ============================================================================

def validate_training_data(covars: pd.DataFrame, data: pd.DataFrame, verbose: bool = False) -> None:
    """Validate input data for bundle creation."""
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    logger.info("Validating training data...")
    
    if len(covars) != len(data):
        error_msg = f"Covariates ({len(covars)} rows) and data ({len(data)} rows) must have same length"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if 'SITE' not in covars.columns:
        error_msg = "Covariates dataframe must contain 'SITE' column"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if covars.isnull().any().any():
        error_msg = "Covariates contain missing values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if data.isnull().any().any():
        error_msg = "Feature data contains missing values"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Training data validation passed")


def validate_harmonization_data(covars: pd.DataFrame, 
                               data: pd.DataFrame, 
                               bundle: Dict[str, Any],
                               verbose: bool = False) -> None:
    """Validate that data is compatible with bundle."""
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    logger.info("Validating training data...")
    # Check dimensions
    if len(covars) != len(data):
        error_msg = f"Covariates ({len(covars)} rows) and data ({len(data)} rows) must have same length"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check required columns
    if 'SITE' not in covars.columns:
        error_msg = "Covariates dataframe must contain 'SITE' column"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check covariates match bundle
    missing_covars = [col for col in bundle["covariate_cols"] if col not in covars.columns]
    if missing_covars:
        error_msg = f"Missing required covariates {missing_covars} for using current bundle"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check features match bundle
    missing_features = [col for col in bundle["feature_cols"] if col not in data.columns]
    if missing_features:
        error_msg = f"Missing required feature col {missing_features} for using current bundle"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Training data validation passed")


# ============================================================================
# PLOTTING AND VISUALIZATION
# ============================================================================

def plot_distribution_comparison(raw_values: np.ndarray,
                                updated_values: np.ndarray,
                                distribution_name: str = "Distribution",
                                site_name: str = "Site",
                                output_dir: Optional[Path] = None,
                                figsize: tuple = (10, 6)) -> None:

    logger = get_logger()

    plt.figure(figsize=figsize)
    sns.kdeplot(raw_values, label='Raw (Observed)', alpha=0.8, linewidth=3)
    sns.kdeplot(updated_values, label='Posterior Updated', alpha=0.8, linewidth=3)

    plt.xlabel(f'{distribution_name} Values', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'{distribution_name} Distribution Comparison: {site_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)


    stats_text = f'''Raw: mu={raw_values.mean():.4f}, sigma={raw_values.std():.4f}
    Posterior: mu={updated_values.mean():.4f}, sigma={updated_values.std():.4f}'''
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{distribution_name.lower()}_comparison_{site_name}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved {distribution_name} comparison plot: {output_path}")

    plt.close()



# ============================================================================
# LOAD AND SHOW BUNDLE INFO
# ============================================================================


def loadBundle(bundle_path: str, verbose: bool = False) -> Dict[str, Any]:
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    logger.info(f"Loading bundle from: {bundle_path}")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"Bundle loaded: {len(bundle['feature_cols'])} features, {len(bundle['bins_site'])} sites")
    return bundle

def getBundleInfo(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_features": len(bundle["feature_cols"]),
        "n_sites": len(bundle["bins_site"]),
        "total_subjects": int(bundle["bins_n"].sum()),
        "covariates": bundle["covariate_cols"],
        "sites": list(bundle["bins_site"]),
        "subjects_per_site": dict(zip(bundle["bins_site"], bundle["bins_n"]))
    }
