# crebLearn.py - Simple functional interface for bundle creation
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path

from .creb_core import (
    fit_covariates,
    build_design_matrix, aggregate_site_bins, setup_logging, get_logger,
    validate_training_data
)

## ensure SITE is in covars dataframe, ensure nothing is missing in dataframe
def crebLearn(covars: pd.DataFrame,
                data: pd.DataFrame,
                output_path: Optional[str] = None,
                verbose: bool = False,
                log_level: Optional[str] = None) -> Dict[str, Any]:
    """
    Create harmonization bundle from training data.
    
    Args:
        covars: DataFrame with covariates (must contain 'SITE' column)
        data: DataFrame with feature data (same number of rows as covars)
        output_path: Optional path to save the bundle
        
    Returns:
        Dictionary containing the harmonization bundle
    """

    logger = setup_logging(verbose=verbose, log_level=log_level)
    
    validate_training_data(covars, data, verbose=verbose)
    covariate_cols = [col for col in covars.columns if col != 'SITE']
    feature_cols = list(data.columns)
    
    logger.info(f"Training data summary:")
    logger.info(f"  - Subjects: {len(covars)}")
    logger.info(f"  - Sites: {covars['SITE'].nunique()} ({list(covars['SITE'].unique())})")
    logger.info(f"  - Covariates: {covariate_cols}")
    logger.info(f"  - Features: {len(feature_cols)}")
    

    Y = data.to_numpy(dtype=np.float64)
    X, covar_names= build_design_matrix(covars, covariate_cols, verbose=verbose)

    B = fit_covariates(X, Y, verbose=verbose)

    # X @ B contain both the batch baseline for each feature, and the biological signal
    # the intercept is for fitting the batch baseline, and the covar column in design matrix are fitted 
    # for the biological signal
    R = Y - (X @ B) 
    squared_residuals = R**2
    var_pooled = np.mean(squared_residuals, axis=0, keepdims=True)
    var_pooled = np.maximum(var_pooled, 1e-8)
    
    R = R/np.sqrt(var_pooled)
    
    logger.info(f"Residuals computed: range [{R.min():.4f}, {R.max():.4f}]")
    logger.info(f"Subtracted effects: {covar_names}")

    logger.info("Step 4: Aggregating into site bins...")
    n_bin, site_bin, mu_bin, sigma2_bin = aggregate_site_bins(R, covars['SITE'].values, verbose=verbose)

    logger.info("Step 5: Creating bundle...")
    bundle = {
        # Configuration
        "covariate_cols": covariate_cols,
        "feature_cols": feature_cols,
        
        # Regression model
        "design_names": covar_names,
        "B": B.astype(np.float32),

        # Var of residual 
        "var_pooled": var_pooled,

        # Site bins
        "bins_n": n_bin.astype(np.int64),
        "bins_mu": mu_bin.astype(np.float32),
        "bins_sigma2": sigma2_bin.astype(np.float32),
        "bins_site": site_bin,
        
        # Metadata
        "n_subjects_training": len(covars),
        "n_sites_training": len(np.unique(covars['SITE'].values)),
    }
    if output_path:
        logger.info("Saving bundle...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)
        
        logger.info(f"Bundle saved to: {output_path}")
    
    # Summary
    logger.info("=== Bundle Summary ===")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Sites: {len(n_bin)}")
    logger.info(f"Total subjects: {n_bin.sum()}")
    logger.info(f"Subjects per site: {dict(zip(site_bin, n_bin))}")
    logger.info(f"Covariates: {covariate_cols}")
    
    return bundle



