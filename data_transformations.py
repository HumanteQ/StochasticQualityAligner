import inspect
import numpy as np
import pandas as pd

from scipy.stats import norm, pearsonr
from sklearn.metrics import (accuracy_score, f1_score,
                             r2_score, precision_score,
                             roc_auc_score, balanced_accuracy_score,
                             recall_score, roc_auc_score,
                             mean_absolute_error,
                             mean_squared_error)

from constants import HUNDREDTHS, POSITIVE_TWENTIETHS


def binarize_target(target, prediction, target_positive_proportion):
    target_threshold = norm.ppf(1 - target_positive_proportion)
    return target > target_threshold, prediction
    
binarize_target.params = {
    'target_positive_proportion': POSITIVE_TWENTIETHS
}
binarize_target.metrics = {
    'roc_auc': roc_auc_score
}

def identity_transform(target, prediction):
    return target, prediction

identity_transform.params = {
}

identity_transform.metrics = {
    'r2':lambda x,y: pearsonr(x, y).statistic ** 2
}


def scale_prediction(target, prediction, prediction_log2_scale):
    prediction_scale = np.exp2(prediction_log2_scale)
    return target, prediction * prediction_scale

scale_prediction.params = {
    'prediction_log2_scale': list(range(-10,11))
}
scale_prediction.metrics = {
   'normalized_rmse' : lambda x,y: mean_squared_error(x,y,squared=False),
    'normalized_mse' : lambda x,y: mean_squared_error(x,y,squared=True),
    'normalized_mae' : mean_absolute_error
}
scale_prediction.less_is_better_metrics = ['normalized_rmse',
                                                    'normalized_mse',
                                                    'normalized_mae']


def binarize_target_prediction(target, prediction, target_positive_proportion,
                              prediction_positive_proportion):
    target_threshold = norm.ppf(1 - target_positive_proportion)
    prediction_threshold = norm.ppf(1 - prediction_positive_proportion)
    return target > target_threshold, prediction > prediction_threshold
    
binarize_target_prediction.params = {
    'target_positive_proportion': POSITIVE_TWENTIETHS,
    'prediction_positive_proportion': POSITIVE_TWENTIETHS
}
binarize_target_prediction.metrics = {
    'accuracy': accuracy_score,
    'f1':f1_score,
    'precision':precision_score,
    'recall':recall_score,
    'balanced_accuracy':balanced_accuracy_score,
    'f1_macro':lambda x,y: f1_score(x,y,average='macro'),
    'precision_macro':lambda x,y: precision_score(x,y,average='macro'),
    'recall_macro':lambda x,y: recall_score(x,y,average='macro'),
    'f1_micro':lambda x,y: f1_score(x,y,average='micro'),
    'precision_micro':lambda x,y: precision_score(x,y,average='micro'),
    'recall_micro':lambda x,y: recall_score(x,y,average='micro'),
    'f1_weighted':lambda x,y: f1_score(x,y,average='weighted'),
    'precision_weighted':lambda x,y: precision_score(x,y,average='weighted'),
    'recall_weighted':lambda x,y: recall_score(x,y,average='weighted'),
}
binarize_target_prediction.less_is_better_metrics = []


def sample_params_downsampled_datasets():
    eeg_mdd_pos_neg_data_ratio = 34 / 30
    iber_lef_pos_neg_data_ratio = 68 / 86
    smhd_xai_pos_neg_data_ratio = 7818 / 7816
    smhd_hybrid_pos_neg_data_ration = 1316 / 1316
    pos_neg_data_ratio = np.random.choice([
        eeg_mdd_pos_neg_data_ratio,
        iber_lef_pos_neg_data_ratio,
        smhd_xai_pos_neg_data_ratio,
        smhd_hybrid_pos_neg_data_ration
        
    ])
    
    
    target_positive_proportion = np.random.choice([i/100 for i in range(3,16)])
    prediction_positive_proportion = np.random.choice(POSITIVE_TWENTIETHS)
    pos_neg_population_ratio = target_positive_proportion / (1 - target_positive_proportion)
    
    positive_subsample = 1
    negative_subsample = round(pos_neg_population_ratio / pos_neg_data_ratio, 3)
    return {
        'target_positive_proportion': target_positive_proportion,
        'prediction_positive_proportion': prediction_positive_proportion,
        'positive_subsample': positive_subsample,
        'negative_subsample': negative_subsample
    }

def binarize_target_prediction_downsample_target(
    target, 
    prediction,     
    target_positive_proportion,
    prediction_positive_proportion,
    positive_subsample,
    negative_subsample,
                                        ):
    sample_df = pd.DataFrame({'x':target,
                             'y':prediction})
    target_threshold = norm.ppf(1 - target_positive_proportion)
    sample_df['x_binary'] = sample_df['x'] > target_threshold
    positive_df = sample_df[sample_df.x_binary == True]
    negative_df = sample_df[sample_df.x_binary == False]
    if positive_subsample < 1:
        positive_df = positive_df.sample(frac=positive_subsample)
    if negative_subsample < 1:
        negative_df = negative_df.sample(frac=negative_subsample)
    subsample_df = pd.concat([positive_df, negative_df],
                     axis=0)
    prediction_threshold = subsample_df['y'].quantile(1 - prediction_positive_proportion)
    subsample_df['y_binary'] = subsample_df['y'] > prediction_threshold
    return subsample_df['x_binary'].values, subsample_df['y_binary'].values
    
binarize_target_prediction_downsample_target.params = sample_params_downsampled_datasets

binarize_target_prediction_downsample_target.metrics = {
    'accuracy': accuracy_score,
    'f1':f1_score,
    'precision':precision_score,
    'recall':recall_score,
    'balanced_accuracy':balanced_accuracy_score,
    'f1_macro':lambda x,y: f1_score(x,y,average='macro'),
    'precision_macro':lambda x,y: precision_score(x,y,average='macro'),
    'recall_macro':lambda x,y: recall_score(x,y,average='macro'),
    'f1_micro':lambda x,y: f1_score(x,y,average='micro'),
    'precision_micro':lambda x,y: precision_score(x,y,average='micro'),
    'recall_micro':lambda x,y: recall_score(x,y,average='micro'),
    'f1_weighted':lambda x,y: f1_score(x,y,average='weighted'),
    'precision_weighted':lambda x,y: precision_score(x,y,average='weighted'),
    'recall_weighted':lambda x,y: recall_score(x,y,average='weighted'),
}
binarize_target_prediction_downsample_target.less_is_better_metrics = []


def sample_params_ranlp2023():
    target_positive_proportion, target_negative_proportion = 0.18, 0.27
    pairs = [(0.04, 0.33),(0.05, 0.39),(0.05, 0.32)]
    pair_idx = np.random.choice(len(pairs))
    prediction_positive_proportion, prediction_negative_proportion = pairs[pair_idx]
    return {
        'target_positive_proportion':target_positive_proportion,
        'target_negative_proportion':target_negative_proportion,
        'prediction_positive_proportion':prediction_positive_proportion,
        'prediction_negative_proportion':prediction_negative_proportion,
    }
    

def discretise_target_prediction_3class(target, prediction, 
                            target_positive_proportion,
                           target_negative_proportion,
                           prediction_positive_proportion,
                           prediction_negative_proportion):
    assert target_negative_proportion + target_positive_proportion < 1
    assert prediction_negative_proportion + prediction_positive_proportion < 1
    target_positive_threshold = norm.ppf(1 - target_positive_proportion)
    target_negative_threshold = norm.ppf(target_negative_proportion)
    prediction_positive_threshold = norm.ppf(1 - prediction_positive_proportion)
    prediction_negative_threshold = norm.ppf(prediction_negative_proportion)
    target_discrete = (1 * (target > target_positive_threshold)
                            - 1 * (target < target_negative_threshold)
                            )
    prediction_discrete = (1 * (prediction > prediction_positive_threshold)
                            - 1 * (prediction < prediction_negative_threshold)
                            )
    return target_discrete, prediction_discrete


discretise_target_prediction_3class.params = sample_params_ranlp2023

discretise_target_prediction_3class.metrics = {
    'accuracy': accuracy_score,
    'balanced_accuracy':balanced_accuracy_score,
    'f1_macro':lambda x,y: f1_score(x,y,average='macro'),
    'precision_macro':lambda x,y: precision_score(x,y,average='macro'),
    'recall_macro':lambda x,y: recall_score(x,y,average='macro'),
    'f1_micro':lambda x,y: f1_score(x,y,average='micro'),
    'precision_micro':lambda x,y: precision_score(x,y,average='micro'),
    'recall_micro':lambda x,y: recall_score(x,y,average='micro'),
    'f1_weighted':lambda x,y: f1_score(x,y,average='weighted'),
    'precision_weighted':lambda x,y: precision_score(x,y,average='weighted'),
    'recall_weighted':lambda x,y: recall_score(x,y,average='weighted'),
}

def list_local_functions():
    current_module = inspect.getmodule(inspect.currentframe())
    functions = [obj for name, obj in 
                 inspect.getmembers(current_module, inspect.isfunction) 
                 if obj.__module__ == current_module.__name__ 
                 and name != list_local_functions.__name__]
        
    return functions



transformations = {}
for func in list_local_functions():
    transformations[func.__name__] = func
