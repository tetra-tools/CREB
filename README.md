# CREB: Consistent Reference External Batch Harmonization

CREB is a harmonization tool for harmonizing data with empirical Bayes methods similar to ComBat. Unlike ComBat, CREB is capable of harmonizing train and test data independently which prevents data leakage. CREB has two main functions CREBLearn and CREBApply, which first learns 'site' priors and then applies this to new unseen data, respectively. This model can be easily deployed in machine learning models to harmonize data prior to conducting prediction in test sets. This package provides functionality for correcting site effects in data while preserving biologically relevant covariate effects.

## Citations
Cite the CREB manuscript: Kharade A.*, Pan Y.*, Andreescu C., Karim H., CREB: Consistent Reference External Batch Harmonization. Under review. 

**Contacts:**
* Helmet Karim
* Yiyan Pan
* Ameya Kharade
  
## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Working with Parquet Files](#working-with-parquet-files)
5. [API Reference](#api-reference)
6. [How It Works](#how-it-works)
7. [Usage Tips](#usage-tips)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)


## Overview
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This package provides a simple, memory-efficient way to correct for site effects in data using empirical Bayes methods. The core algorithm first builds a harmonization bundle from training data, then applies the learned corrections to new datasets. The package allow for harmonization of completely unseen sites that were not present in the original training data.


## Installation


### Option 1: Install from PyPI

```bash
pip install creb
```

### Option 2: Install from Source with uv

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management:

```bash
# Clone the repository
git clone https://github.com/tetra-tools/CREB.git
cd creb

# Create virtual environment and install dependencies
uv venv
uv sync
```


## Quick Start

The package requires two inputs:

1. **Data matrix**: A pandas DataFrame containing features to harmonize (e.g., brain volumes, connectivity measures)
2. **Covariates matrix**: A pandas DataFrame containing covariates with a required "SITE" column

### Data Matrix Example

Your data should be a numeric matrix where rows are subjects and columns are features:

```python
       feature_1  feature_2  feature_3  ...  feature_103741
0       3138.0    3164.2      206.4    ...     1847.3
1       1708.4    2351.2      364.0    ...     1942.1
...      ...       ...        ...      ...        ...
```

### Covariates Matrix Example

The covariates DataFrame must contain a "SITE" column and any other numeric covariates:

```python
     SITE   AGE  SEX_M
0  SITE_A  76.5      1
1  SITE_B  80.1      1
2  SITE_A  82.9      0
...   ...   ...    ...
```

**Important notes:**
- Both matrices must have the same number of rows (subjects)
- All covariates must be numeric (handle categorical variables with pandas.get_dummies beforehand)
- No missing values are allowed - perform complete case analysis first
- The order of subjects must be identical in both matrices

### Basic Usage

```python
import pandas as pd
import creb.creb as cr

# Load your data
data = pd.read_csv('brain_features.csv')
covars = pd.read_csv('subject_covariates.csv')

# Create harmonization bundle from training data
bundle = cr.crebLearn(
    covars=covars,
    data=data,
    output_path="harmonization_bundle.pkl",
    verbose=True
)

# Load bundle info
print(cr.getBundleInfo(cr.loadBundle("harmonization_bundle.pkl")))

# Harmonize new data using the bundle
harmonized_data = cr.crebApply(
    covars=covars,
    data=data,
    bundle_path="harmonization_bundle.pkl",
    method="joint",  # or "iterative"
    verbose=True,
)
```

### Using Pre-uploaded Synthetic Bundle

For quick testing and prototyping, you can use our pre-uploaded bundle that was trained on 9 diverse neuroimaging sites:

```python
import pandas as pd
import creb.creb as cr

# Load your data
data = pd.read_csv('your_brain_features.csv')
covars = pd.read_csv('your_subject_covariates.csv')

harmonized_data = cr.crebApply(
    covars=covars,
    data=data,
    bundle_path="pretrained_bundle_9_sites.pkl",
    method="joint"
)
# Note: The pre-uploaded bundle will be regularly updated and expanded
```

## Working with Parquet Files

For large datasets, the package supports Parquet format:

```python
import pandas as pd
import creb.creb as cr

# Load data from Parquet
data = pd.read_parquet('brain_features.parquet')
covars = pd.read_parquet('subject_covariates.parquet')

# Create bundle
bundle = cr.crebLearn(covars=covars, data=data, output_path="bundle.pkl")

# Harmonize and save as Parquet
harmonized = cr.crebApply(
    covars=covars,
    data=data,
    bundle_path="bundle.pkl",
    output_path="harmonized_data.parquet"
)
```

---

## API Reference

### crebLearn

Creates a harmonization bundle from training data.

```python
def crebLearn(covars: pd.DataFrame,
              data: pd.DataFrame,
              include_site_dummies: bool = False,
              output_path: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `covars`: DataFrame with covariates (must contain 'SITE' column)
- `data`: DataFrame with feature data (same number of rows as covars)
- `output_path`: Optional path to save the bundle as pickle

**Returns:**
- Dictionary containing the harmonization bundle with learned parameters

### crebApply

Harmonizes new data using a pre-trained bundle.

```python
def crebApply(covars: pd.DataFrame,
                data: pd.DataFrame,
                bundle_path: str,
                method: str = "joint",  
                output_path: Optional[str] = None,
                verbose: bool = False,
                makeplot: bool = False,
                log_level: Optional[str] = None) -> pd.DataFrame:
```

**Parameters:**
- `covars`: DataFrame with covariates (must contain 'SITE' column)
- `data`: DataFrame with feature data (same number of rows as covars)
- `bundle_path`: Path to harmonization bundle
- `method`: "joint" (default) or "iterative" posterior updates
- `output_path`: Optional path to save harmonized data
- `verbose`: Optional flag for verbose output printing
- `makeplot`: Optional flag to plot distribution of site effect mean and variance before and after posterior update
- `log_level`: Optional input for logging level


**Returns:**
- DataFrame with harmonized data

### loadBundle

Loads a harmonization bundle from file.

```python
def loadBundle(bundle_path: str, verbose: bool = False) -> Dict[str, Any]
```

### getBundleInfo

Gets summary information about a bundle.

```python
def getBundleInfo(bundle: Dict[str, Any]) -> Dict[str, Any]
```


## How It Works

### 1. Bundle Creation (crebLearn)

1. **Covariate regression**: Fit linear model `Y = X * B + R` where X includes intercept + covariates. Normalize R with pooled variance
3. **Site effect aggregation**: Compute site-level summary statistics (means, SST) from residuals R
4. **Prior estimation**: Learn Empirical Bayes priors from site statistics
5. **Bundle creation**: Save all parameters needed for harmonization

### 2. Harmonization (crebApply)

1. **Residual computation**: Compute residuals `R = Y - X * B` using learned coefficients from bundle. Normalize with pooled variance from train bundle. 
3. **Site statistics**: Compute per-site means and SST from residuals
4. **Posterior updates**: Apply empirical Bayes update with learned priors
5. **Reconstruction**: Multiply by pooled variance, add biological covariate effects back 

### Correction Methods
- **"joint"**: Uses group-wise priors for simultaneous location/scale correction
- **"iterative"**: iteratively assume we know mean or variance of the Residual distribution to make update, return when reach convergence. 


## Usage Tips

### Typical Workflow:

To use CREB, user create harmonization bundle using their training data. They can then upload/share only the the trained bundle (not the training data). Anyone with access to the bundle can apply harmonization to new datasets at runtime.

```python
import creb.creb as cr
harmonized_data = cr.crebApply(
    covars=new_data_covars,
    data=new_data_features,
    bundle_path='harmonization_bundle.pkl'
)
```

This approach enables:
- **Privacy compliance**: Training data stays secure while harmonization capabilities are deployed
- **Reproducibility**: Consistent harmonization parameters across deployments



### Handling Missing Values

The package requires complete data. Handle missing values before harmonization:

```python
# Remove subjects with missing covariates
mask = covars.notna().all(axis=1) & data.notna().all(axis=1)
covars_clean = covars[mask].copy()
data_clean = data[mask].copy()
```

### Custom Feature Selection

```python
# Select specific feature types
feature_cols = [col for col in data.columns if col.startswith('connectivity_')]
data_selected = data[feature_cols]
```

### Multiple External Datasets

```python
# Harmonize multiple external datasets with same bundle
external_datasets = ['camcan', 'aging', 'hcp']
for name in external_datasets:
    ext_data = pd.read_parquet(f'external_{name}.parquet')
    ext_covars = pd.read_parquet(f'covars_{name}.parquet')

    harmonized = cr.crebApply(
        covars=ext_covars,
        data=ext_data,
        bundle_path="bundle.pkl",
        output_path=f'harm_{name}.parquet'
    )
```



## Troubleshooting

### Common Issues

**"SITE column not found"**
- Ensure your covariates DataFrame contains a column named exactly "SITE"

**"Missing required covariates"**
- Make sure all covariates present in training are also in new data
- Check for exact column name matches (case-sensitive)




## Contributing

Contributions are welcome! Please submit pull requests to the GitHub repository.

