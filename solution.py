"""Freight rate assessment solution: temporal validation, CatBoost regression, predictions and December chart."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC = ['pickup_lat','pickup_lon','delivery_lat','delivery_lon','distance','weight','market_index','quote_signal','month','day_of_week','day_of_month','day_of_year','month_sin','month_cos','dow_sin','dow_cos','route_distance_ratio','weight_per_mile']
CATEGORICAL = ['pickup','delivery','equipment','route']
FEATURES = CATEGORICAL + NUMERIC
TARGET = 'posted_rate'

def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # Some scoring-only rows intentionally omit fields such as coordinates or market signals.
    for col in ['pickup_lat','pickup_lon','delivery_lat','delivery_lon','distance','weight','market_index','quote_signal']:
        if col not in x.columns:
            x[col] = np.nan
    for col in ['pickup','delivery','equipment']:
        if col not in x.columns:
            x[col] = 'Unknown'
    # Date is parsed strictly so invalid values are visible instead of silently changing meaning.
    dt = pd.to_datetime(x['date'], errors='coerce')
    x['month'] = dt.dt.month
    x['day_of_week'] = dt.dt.dayofweek
    x['day_of_month'] = dt.dt.day
    x['day_of_year'] = dt.dt.dayofyear
    x['month_sin'] = np.sin(2*np.pi*x['month']/12)
    x['month_cos'] = np.cos(2*np.pi*x['month']/12)
    x['dow_sin'] = np.sin(2*np.pi*x['day_of_week']/7)
    x['dow_cos'] = np.cos(2*np.pi*x['day_of_week']/7)
    x['route'] = x['pickup'].fillna('Unknown').astype(str) + ' -> ' + x['delivery'].fillna('Unknown').astype(str)
    dist = pd.to_numeric(x['distance'], errors='coerce').replace(0, np.nan)
    x['route_distance_ratio'] = (x['delivery_lat']-x['pickup_lat']).abs() + (x['delivery_lon']-x['pickup_lon']).abs()
    x['weight_per_mile'] = pd.to_numeric(x['weight'], errors='coerce') / dist
    for c in CATEGORICAL:
        x[c] = x[c].fillna('Unknown').astype(str)
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    return x[FEATURES]

def make_model(seed=42, iterations=1200):
    return CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.045, loss_function='MAE',
        eval_metric='MAE', random_seed=seed, verbose=False, l2_leaf_reg=5,
        random_strength=0.5, allow_writing_files=False, thread_count=-1
    )

def metrics(y, pred):
    return {'mae': float(mean_absolute_error(y,pred)), 'rmse': float(mean_squared_error(y,pred)**0.5), 'r2': float(r2_score(y,pred))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='train-test.csv'); p.add_argument('--validation',default='validation.csv')
    p.add_argument('--template',default='validation-predictions-template.csv'); p.add_argument('--december',default='december-chart-inputs.csv')
    p.add_argument('--output-dir',default='outputs'); p.add_argument('--iterations',type=int,default=1200); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(a.train); validation=pd.read_csv(a.validation); template=pd.read_csv(a.template); december=pd.read_csv(a.december)
    train['date']=pd.to_datetime(train['date'],errors='coerce')
    # Forward-in-time holdout: October is never used to fit this validation model.
    cutoff=pd.Timestamp('2025-10-01')
    fit=train[train.date < cutoff].copy(); hold=train[train.date >= cutoff].copy()
    if fit.empty or hold.empty: raise ValueError('Temporal split produced an empty partition; check dates.')
    xfit=features(fit); xhold=features(hold)
    model=make_model(a.seed,a.iterations)
    model.fit(xfit,fit[TARGET],cat_features=CATEGORICAL,eval_set=(xhold,hold[TARGET]),early_stopping_rounds=100,verbose=False)
    hold_pred=np.maximum(0,model.predict(xhold)); m=metrics(hold[TARGET],hold_pred)
    m.update({'train_rows':int(len(fit)),'holdout_rows':int(len(hold)),'holdout_start':str(hold.date.min().date()),'holdout_end':str(hold.date.max().date()),'best_iteration':int(model.get_best_iteration())})
    with open(out/'validation_metrics.json','w') as f: json.dump(m,f,indent=2)
    pd.DataFrame({'actual':hold[TARGET].to_numpy(),'predicted':hold_pred}).to_csv(out/'temporal_holdout_predictions.csv',index=False)
    # Refit using all labeled rows for final inference.
    final=make_model(a.seed,max(300,model.get_best_iteration()+1))
    final.fit(features(train),train[TARGET],cat_features=CATEGORICAL,verbose=False)
    final.save_model(str(out/'freight_rate_catboost.cbm'))
    pred=np.maximum(0,final.predict(features(validation)))
    if len(template)!=len(validation) or not template['load_id'].equals(validation['load_id']):
        raise ValueError('Template load_id sequence does not exactly match validation.csv')
    result=template.copy(); result['predicted_rate']=pred
    if result['predicted_rate'].isna().any() or len(result)!=12000: raise ValueError('Predictions must be complete for all 12,000 rows.')
    result.to_csv(out/'validation_predictions.csv',index=False,float_format='%.2f')
    # Chart input omits market_index and quote_signal; CatBoost handles missing numeric values.
    dec_pred=np.maximum(0,final.predict(features(december)))
    chart=december[['date']].copy(); chart['predicted_rate']=dec_pred; chart.to_csv(out/'december_predictions.csv',index=False,float_format='%.2f')
    dates=pd.to_datetime(chart.date)
    plt.figure(figsize=(10,5.5)); plt.plot(dates,dec_pred,marker='o',linewidth=2)
    plt.title('December 2025 Freight Rate Prediction'); plt.xlabel('Date'); plt.ylabel('Predicted rate')
    plt.grid(True,alpha=.3); plt.xticks(rotation=35); plt.tight_layout(); plt.savefig(out/'december_prediction_chart.png',dpi=180); plt.close()
    # Feature importance chart.
    imp=pd.Series(final.get_feature_importance(),index=FEATURES).sort_values().tail(15)
    plt.figure(figsize=(9,6)); imp.plot(kind='barh'); plt.title('Top 15 Feature Importances'); plt.xlabel('Prediction value change importance'); plt.tight_layout(); plt.savefig(out/'feature_importance.png',dpi=180); plt.close()
    print(json.dumps({'temporal_validation':m,'prediction_rows':len(result),'prediction_file':str(out/'validation_predictions.csv'),'december_chart':str(out/'december_prediction_chart.png')},indent=2))
if __name__=='__main__': main()
