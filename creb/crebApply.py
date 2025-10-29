# crebApply.py 
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path

from .creb_core import (
    setup_logging, get_logger,
    build_design_matrix,
    estimate_priors_joint, posterior_updates_joint,
    estimate_priors_iterative, posterior_updates_iterative,
    validate_harmonization_data,
    plot_distribution_comparison
)

def get_priors_method_joint(bundle: Dict[str, Any], verbose: bool = False) -> Dict[str, float]:
    """Get priors for joint updates method (simplified single group)."""
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    
    # convert bundle to a dictonary format with key as site name
    # and value as another dictionary holding n subject of this site
    # mean of each feature of this site, and sigma2 of each feature of this site
    site_stats = {}
    for i, site in enumerate(bundle["bins_site"]):
        site_stats[str(site)] = {
            "n": int(bundle["bins_n"][i]),
            "mean": bundle["bins_mu"][i, :].astype(np.float64),
            "sigma2": bundle["bins_sigma2"][i, :].astype(np.float64)
        }
    
    priors = estimate_priors_joint(site_stats, bundle["feature_cols"], verbose=verbose)
    return priors


def get_priors_method_iterative(bundle: Dict[str, Any], verbose: bool = False) -> Dict[str, float]:
    logger = get_logger()
    if verbose:
        setup_logging(verbose=True)
    priors = estimate_priors_iterative(bundle["bins_site"], bundle["bins_n"], bundle["bins_mu"], bundle["bins_sigma2"], verbose=verbose)
    return priors

def crebApply(covars: pd.DataFrame,
                data: pd.DataFrame,
                bundle_path: str,
                method: str = "joint",  
                output_path: Optional[str] = None,
                verbose: bool = False,
                makeplot: bool = False,
                log_level: Optional[str] = None) -> pd.DataFrame:
    """
    Harmonize data using summary statistic from bundle.
    
    Args:
        covars: DataFrame with covariates (must contain 'SITE' column)
        data: DataFrame with feature data (same number of rows as covars)
        bundle_path: Path to harmonization bundle
        method: "separate" or "joint" posterior updates (default: "joint")
        output_path: Optional path to save harmonized data
        
    Returns:
        Harmonized DataFrame
    """
    
    logger = setup_logging(verbose=verbose, log_level=log_level)
    logger.info(f"=== Harmonizing Data with {method} method ===")
    # Load bundle
    logger.info(f"Loading bundle from: {bundle_path}")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"Loaded bundle: {len(bundle['feature_cols'])} features, {len(bundle['bins_site'])} training sites")
    
    # Validate data
    validate_harmonization_data(covars, data, bundle, verbose=verbose)
    
    sites = covars["SITE"].unique()
    logger.info(f"Data to harmonize: {len(covars)} subjects, {len(sites)} sites ({list(sites)})")

    # validation
    if method == "joint":
        priors = get_priors_method_joint(bundle, verbose=verbose)  
    elif method == "iterative":
        priors = get_priors_method_iterative(bundle, verbose=verbose)  
    else:
        error_msg = "Method must be 'joint' or 'iterative"
        logger.error(error_msg)
        raise ValueError(error_msg)  
    
    # # Extract bundle parameters
    feature_cols = bundle["feature_cols"]
    var_pooled = bundle["var_pooled"]
    B = bundle["B"].astype(np.float64)

    
    # # Normalize features
    logger.info("Normalizing features...")
    Y = data[feature_cols].to_numpy(dtype=np.float64)
    # Build design matrix matching training
    logger.info("Building design matrix...")
    X, _= build_design_matrix(
        covars, 
        bundle["covariate_cols"], 
        verbose=verbose
    )

    # # Compute residuals (subtract only covariate effects)
    R = Y - (X @ B)
    R = R / np.sqrt(var_pooled)
    logger.info(f"Residuals computed: range [{R.min():.4f}, {R.max():.4f}]")
    
    # # Harmonize by site
    Y_harmonized = Y.copy()
    
    for site in sites:
        site_mask = (covars["SITE"] == site)
        n_site = site_mask.sum()
        R_site = R[site_mask, :]
        
        logger.info(f"Processing site {site}: {n_site} subjects")
        gamma_raw = R_site.mean(axis=0)
        sst_raw = np.sum((R_site - gamma_raw[None, :])**2, axis=0)
        delta_raw = np.sqrt(sst_raw / (n_site - 1))
        if method == "joint":  
            gamma_star, delta_star = posterior_updates_joint(
                gamma_raw, sst_raw, 
                n_site, priors,
                verbose=verbose
            )
            if makeplot:  # plot if verbose mode is on
                figures_dir = None
                if output_path:
                    figures_dir = Path(output_path).parent / "figures"

                plot_distribution_comparison(gamma_raw, gamma_star, "Gamma", str(site), figures_dir)
                plot_distribution_comparison(delta_raw, delta_star, "Delta", str(site), figures_dir)

        
            # Apply correction
            R_harmonized = (R_site - gamma_star[None, :]) / delta_star[None, :]
            logger.debug(f"Site {site} correction: gamma range [{gamma_star.min():.4f}, {gamma_star.max():.4f}], "
                f"dleta range [{delta_star.min():.4f}, {delta_star.max():.4f}]")

        elif method == "iterative":
            gamma_star, delta_star = posterior_updates_iterative(
                    gamma_raw,
                    sst_raw,
                    n_site,
                    priors,
                    R_site,
                    max_iter=1000,
                    tol=1e-6,
                    verbose=False)

            if makeplot:  
                figures_dir = None
                if output_path:
                    figures_dir = Path(output_path).parent / "figures"
                plot_distribution_comparison(gamma_raw, gamma_star, "Gamma", str(site), figures_dir)
                plot_distribution_comparison(delta_raw, delta_star, "Delta", str(site), figures_dir)

            R_harmonized = (R_site - gamma_star[None, :]) / delta_star[None, :]
        else:
            raise ValueError("choose between 'joint', 'iterative' updates")
        # Reconstruct harmonized Z-scores
        Y_site_harmonized = (X[site_mask, :] @ B) + (R_harmonized * np.sqrt(var_pooled))
        Y_harmonized[site_mask, :] = Y_site_harmonized
    
    # Convert to float32 to save some space on disk
    Y_harmonized = Y_harmonized.astype(np.float32)
    # # # Create output DataFrame
    df_out = covars.copy()
    data_harmonized = pd.DataFrame(Y_harmonized, columns=bundle['feature_cols'], index=data.index)
    df_out = pd.concat([df_out, data_harmonized], axis=1)
    
    if output_path:
        
        logger.info(f"Saving harmonized data to: {output_path}")
        df_out.to_parquet(output_path)
    
    return df_out


