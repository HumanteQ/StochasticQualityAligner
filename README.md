### Overview

This project implements a stochastic method to align quality estimates across different datasets and metrics, as detailed in the paper "Measure Twice and Cut Once: Aligning Quality Estimates in Automated Depression Detection." Its goal is to create a unified scale for comparing quality estimates by generating data samples with predetermined correlations and mapping various quality metrics back to these correlations.

The process of conversion consists of two stages:

1. Sampling and saving tuples of quality estimates along with correlation and transformation parameters used for their estimation.
2. Using the collected tuples to convert a specific metric value to a correlation estimate.
These stages are encapsulated in the functions 'sample' and 'convert' in the `main.py` file.

### `main.py: sample`

**Purpose**: This function is designed to build a database of tuples that includes correlation coefficients, generation parameters, and estimated metrics. It systematically generates pairs of values simulating latent variables and predictions with predefined correlations, applies various transformations to these pairs to reflect real-world scenarios, and calculates a wide range of quality metrics under controlled conditions.
**Parameters**:
 - `sample_size`: The size of the random samples used to estimate the metrics.
 - `iterations`: The number of parameter-metric tuples to be estimated.
 - `transformation_name`: The name of the transformation.
 - `transformation_func`: The transformation function  applied to the original Gaussian sample (callable).
 - `param_sampling`: The method for sampling transformation parameters, which can be either a callable or a dictionary mapping parameter names to lists.
 - `metric_dict`: A dictionary mapping the names of metrics to be calculated to callables.
 - `flush_every`: The frequency at which data is flushed to the database.
 - `metrics_folder`: The folder where parameter-metric tuples are stored.

**Command-line invocation**
Sampling can be invoked from the command line using `python main.py sample --transformation <transformation_name>`. Parameters such as `sample_size`, `iterations`, `flush_every`, and `metrics_folder` can be set through keywords of the same names. All other parameters are loaded from `data_transformations.py` based on the transformation name.

**Expanding the transformation collection**
To add new transformation functions, define them in the `data_transformations.py` file using the function name as the transformation identifier. For each transformation, you should also define metrics as a dictionary mapping metric names to callable functions, assigning this dictionary to `<transformation_name>.metrics`. For parameter sampling, `<transformation_name>.params` can take two forms:
 -  As a dictionary, it should map each parameter name to a list of all possible values from which samples will be drawn.
 -  As a function, it should return a dictionary that maps each parameter name to a specific sampled value for each invocation.

  Additionally, specify metrics where lower values indicate better performance by listing them in `<transformation_name>.less_is_better_metrics`.

### `main.py: convert`
**Purpose**: The `convert` function is designed to convert quality metrics to correlation estimates based on specified values for the metric and the transformation parameters.

**Parameters**:
 - `metrics_folder`: The folder where parameter-metric tuples are stored.
 - `sample_size`: The size of the random samples used for estimating the metrics.
 - `transformation_name`: The name of the transformation.
 - `metric_name`: The name of the quality metric to convert to the correlation.
 - `metric_value`: The value of the input quality metric.
 - `fixed_parameters`: Optional; specific transformation parameters to fix during conversion.
 - `allow_inexact`: Optional; allows for inexact matching of parameters.




