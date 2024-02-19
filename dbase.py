import os
from pathlib import Path
import pandas as pd
import json

class CSVMetricsDatabase:
    def __init__(self, base_folder, sample_size, transformation):
        folder = os.path.join(base_folder, f'sample_{sample_size}')
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(folder, f'{transformation}.csv')
        self.meta_path = os.path.join(folder, f'{transformation}.json')
        if os.path.isfile(self.path):
            self.metric_table_df = pd.read_csv(self.path)
        else:
            self.metric_table_df = None
        if os.path.isfile(self.meta_path):
            with open(self.meta_path) as f:
                metadata = json.load(f)
            self.groupby_cols = metadata['groupby_cols']
            self.aggregated_cols = metadata['aggregated_cols']

        else:
            self.groupby_cols = None
            self.aggregated_cols = None
        self.unsaved_results = []
    
    def flush(self):
        if len(self.unsaved_results) > 0:
            new_df = pd.DataFrame(self.unsaved_results, 
                                  columns=self.groupby_cols + self.aggregated_cols)
            self.unsaved_results = []
            if self.metric_table_df is None:
                self.metric_table_df = new_df

            else:
                self.metric_table_df = pd.concat([self.metric_table_df,
                                             new_df], axis=0)
            self.metric_table_df = (self.metric_table_df
                                    .groupby(self.groupby_cols)
                                    [self.aggregated_cols]
                                    .sum()
                                    .reset_index()
                                   )
        if self.metric_table_df is not None:
            self.metric_table_df.to_csv(self.path, index=False)
        if self.aggregated_cols is not None and self.groupby_cols is not None:
            metadata = {
                'aggregated_cols': self.aggregated_cols,
                'groupby_cols':self.groupby_cols
            }
            with open(self.meta_path,'w') as f:
                json.dump( metadata,f)
    
    def save_result(self, pearson_r, parameters, metrics):
        d = {'pearson_r':pearson_r,
            'estimate__count__': 1.0}
        for param, value in parameters.items():
            assert param not in d, f'Duplicate key: {param}'
            d[param] = value
        for metric, value in metrics.items():
            assert metric not in d, f'Duplicate key: {param}'
            d[metric] = value
        
        if self.groupby_cols is None or self.aggregated_cols is None:
            self.groupby_cols = ['pearson_r'] + list( sorted(parameters.keys()))
            self.aggregated_cols = (list(sorted(metrics.keys())) + 
                                    ['estimate__count__'])
        cols = (self.groupby_cols + self.aggregated_cols)
        
                    
        new_results = [d[col] for col in cols]
        self.unsaved_results.append(new_results)
        
    def close(self):
        self.flush()
        
    def get_metric_table(self):
        self.flush()
        result_df =  self.metric_table_df.copy()
        denominator = result_df.estimate__count__.values.reshape(-1,1)
        result_df[self.aggregated_cols] = result_df[self.aggregated_cols] / denominator
        return result_df.drop(columns=['estimate__count__'])