# WISCON
Wisdom of the Contexts: Active Ensembl Learning for Contextual Anomaly Detection

This is an implementation of WisCon (Wisdom of the Contexts) algorithm. WisCon is contextual anomaly detection algorithm which automatically creates contexts from the given feature set and constructs an ensemble of multiple contexts, with varying importance
scores. 

# USAGE
The following code is an example of how to perform anomaly detection using WISCON. An example config file (params.conf) is provided in the project file. 

```python
df_feat = pd.read_csv(file_name, index_col=None)
#The last column of the dataset is ground-truth labels.  
features = df_feat.columns[0:-1]

wiscon = WisCon(params='params.conf', X=df_feat.iloc[:, :-1].values, y=df_feat.iloc[:, -1].values,
                     features=features)
wiscon.initialize_ensemble()
wiscon.run_active_learning_schema()
wiscon.anomaly_score_aggregation()
wiscon.calculate_performance()
```
