from argparse import ArgumentParser
from data_transformations import transformations as dt
from collections.abc import Iterable
from dbase import CSVMetricsDatabase
import numpy as np

from constants import HUNDREDTHS


def sample_from_correlation(pearson_r,
                           sample_size):
    xy = np.random.multivariate_normal(mean=[0,0],
                             cov=[[1,pearson_r],
                                  [pearson_r,1]],
                                 size=sample_size)
    return xy[:,0].reshape(-1), xy[:,1].reshape(-1)


def sample_params(param_sampling):
    if callable(param_sampling):
        result = param_sampling()
   
    else:
        result = {}
        for name, possible_values in param_sampling.items():
            if isinstance(possible_values, Iterable):
                result[name] = np.random.choice(possible_values)
            elif callable(possible_values):
                result[name] = possible_values(**result)
    return result

    
def calculate_metrics(target, prediction, metric_dict):
    result = {}
    for metric_name, func in metric_dict.items():
        result[metric_name] = func(target, prediction)
    return result

def sample(sample_size, 
           iterations,
           transformation_name,
           transformation_func, 
           param_sampling,
           metric_dict,
           flush_every,
           metrics_folder
          ):
    db = CSVMetricsDatabase(metrics_folder, 
                           sample_size,
                           transformation_name)
    for i in range(1, 1 + iterations):
        pearson_r = np.random.choice(HUNDREDTHS)
        params = sample_params(param_sampling)
        target, prediction = sample_from_correlation(pearson_r, sample_size)
        target, prediction = transformation_func(target, prediction, **params)
        metrics = calculate_metrics(target, prediction, metric_dict)
        db.save_result(pearson_r, params, metrics)
        if flush_every is not None and i % flush_every == 0:
            db.flush()
    db.close()
    
    
def convert(metrics_folder, sample_size, transformation_name, 
           metric_name, metric_value, fixed_parameters=None, 
           allow_inexact = True):
    if fixed_parameters is None:
        fixed_parameters = {}
    db = CSVMetricsDatabase(metrics_folder, 
                           sample_size,
                           transformation_name)
    metric_df = db.get_metric_table()
    if (len(metric_df) < 36000) and (transformation_name == 'binarize_target_prediction'):
        print (len(metric_df))
    transformation = dt[transformation_name]
    for param_name, param_value in fixed_parameters.items():
        if allow_inexact:
            min_dist = (metric_df[param_name] - param_value).abs().min()
        else:
            min_dist = 0
        metric_df = metric_df[(metric_df[param_name] - param_value).abs() <= min_dist]
    if (hasattr(transformation, 'less_is_better_metrics') and 
        metric_name in transformation.less_is_better_metrics
       ):
        metric_df = metric_df[metric_df[metric_name] <= metric_value]
    else:
        metric_df = metric_df[metric_df[metric_name] >= metric_value]
    if len(metric_df) == 0:
        return 1
    else:
        return metric_df.pearson_r.min()
        
    
    

def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument('--sample_size', type=int, default=10_000)
    sample_parser.add_argument('--iterations', type=int, default=1)
    sample_parser.add_argument('--flush_every', type=int)
    sample_parser.add_argument('--transformation', required=True)
    sample_parser.add_argument('--metrics_folder', default = './metrics')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.subparser_name == 'sample':
        transformation_name = args.transformation
        transformation = dt[transformation_name]
        sample(sample_size = args.sample_size,
               iterations = args.iterations,
               transformation_name = transformation_name,
               transformation_func = transformation,
               param_sampling = transformation.params,
               metric_dict = transformation.metrics,
               flush_every = args.flush_every,
               metrics_folder = args.metrics_folder
              )
    

