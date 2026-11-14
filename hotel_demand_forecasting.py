"""
Hotel demand forecasting: daily stayed arrivals per hotel (Hotel Booking Demand dataset).

Usage:   python hotel_demand_forecasting.py [path_to_hotel_bookings_demand.csv]

What it does
  1. Builds a daily series of stayed arrivals (is_canceled = 0) for each hotel.
  2. Runs four rolling-origin validation folds (about 90 days each) for seven models and an ensemble.
  3. Fits the final models on all data and forecasts the next 90 days (1 Sep to 29 Nov 2017).
  4. Writes Hotel_Demand_Validation_Metrics.csv and Hotel_Demand_Forecast_Sep_Nov_2017.csv.

Requires: pandas, numpy, scikit-learn, statsmodels, prophet, holidays
"""
import pandas as pd, numpy as np, warnings, pickle, logging, sys, time
warnings.filterwarnings("ignore"); logging.getLogger("prophet").setLevel(logging.ERROR); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays

DATA=sys.argv[1] if len(sys.argv)>1 else 'hotel_bookings_demand.csv'
df=pd.read_csv(DATA)
df['arrival_date']=pd.to_datetime(df.arrival_date_year.astype(str)+'-'+df.arrival_date_month+'-'+df.arrival_date_day_of_month.astype(str),format='%Y-%B-%d')
HOTELS=['City Hotel','Resort Hotel']; IDX=pd.date_range(df.arrival_date.min(),df.arrival_date.max())
stay=df[df.is_canceled==0]
Y={h:stay[stay.hotel==h].groupby('arrival_date').size().reindex(IDX,fill_value=0).astype(float) for h in HOTELS}
zero_filled={h:int((stay[stay.hotel==h].groupby('arrival_date').size().reindex(IDX).isna()).sum()) for h in HOTELS}
PT=holidays.country_holidays('PT',years=range(2015,2019))
START=IDX[0]

# ---------- calendar features
def fourier(dates,K,period=365.25):
    t=np.asarray((dates-START).days,dtype=float); cols={}
    for k in range(1,K+1): cols[f's{k}']=np.sin(2*np.pi*k*t/period); cols[f'c{k}']=np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(cols,index=dates)
def dow_dummies(dates):
    return pd.DataFrame({f'dow{d}':(dates.dayofweek==d).astype(float) for d in range(1,7)},index=dates)
def gbm_feats(dates):
    X=fourier(dates,2); X['dow']=dates.dayofweek; X['month']=dates.month; X['woy']=dates.isocalendar().week.values.astype(int)
    X['trend']=(dates-START).days; X['holiday']=[int(d in PT) for d in dates]; return X

# ---------- models: (train Series, horizon dates) -> array
def m_snaive7(tr,hd): last=tr.values[-7:]; return np.array([last[i%7] for i in range(len(hd))])
def m_snaive364(tr,hd):
    out=[]
    for d in hd:
        p=d-pd.Timedelta(days=364); out.append(tr.get(p,np.nan))
    out=np.array(out,dtype=float); fb=m_snaive7(tr,hd); return np.where(np.isnan(out),fb,out)
def m_wk4(tr,hd):
    last=tr.values[-28:]; means=[last[i::7].mean() for i in range(7)]; return np.array([means[i%7] for i in range(len(hd))])
def m_ets(tr,hd):
    f=ExponentialSmoothing(tr.values,trend='add',damped_trend=True,seasonal='add',seasonal_periods=7,initialization_method='estimated').fit()
    return f.forecast(len(hd))
def m_sarimax(tr,hd):
    Xtr=pd.concat([fourier(tr.index,3),dow_dummies(tr.index)],axis=1); Xte=pd.concat([fourier(hd,3),dow_dummies(hd)],axis=1)
    f=SARIMAX(tr.values,exog=Xtr.values,order=(1,0,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=150)
    return f.forecast(len(hd),exog=Xte.values)
def m_prophet(tr,hd):
    m=Prophet(weekly_seasonality=True,yearly_seasonality=True,daily_seasonality=False,seasonality_mode='additive'); m.add_country_holidays('PT')
    m.fit(pd.DataFrame({'ds':tr.index,'y':tr.values})); return m.predict(pd.DataFrame({'ds':hd}))['yhat'].values
def m_gbm(tr,hd):
    g=HistGradientBoostingRegressor(max_depth=4,learning_rate=0.05,max_iter=300,min_samples_leaf=15,random_state=0)
    g.fit(gbm_feats(tr.index),tr.values); return g.predict(gbm_feats(hd))
MODELS={'Seasonal naive (7d)':m_snaive7,'Seasonal naive (364d)':m_snaive364,'Weekday mean (4 wks)':m_wk4,'ETS (Holt-Winters)':m_ets,'SARIMAX + Fourier':m_sarimax,'Prophet':m_prophet,'Gradient boosting':m_gbm}

FOLDS=[('Fold 1','2016-08-31','2016-09-01','2016-11-30'),('Fold 2','2016-11-30','2016-12-01','2017-02-28'),('Fold 3','2017-02-28','2017-03-01','2017-05-31'),('Fold 4 (holdout)','2017-05-31','2017-06-01','2017-08-31')]
rows=[]; t0=time.time()
for fname,tr_end,te_start,te_end in FOLDS:
    for h in HOTELS:
        tr=Y[h][:tr_end]; hd=pd.date_range(te_start,te_end); te=Y[h][hd]
        for mn,fn in MODELS.items():
            try: p=np.clip(np.asarray(fn(tr,hd),dtype=float),0,None)
            except Exception as e: print('FAIL',fname,h,mn,e); p=m_snaive7(tr,hd)
            rows.append(pd.DataFrame({'date':hd,'hotel':h,'fold':fname,'model':mn,'y':te.values,'yhat':p}))
    print(fname,'done',round(time.time()-t0),'s',flush=True)
cv=pd.concat(rows,ignore_index=True)

def metrics(g):
    e=g.y-g.yhat; return pd.Series({'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),'WAPE':e.abs().sum()/g.y.sum()*100,'Bias':e.mean(),'n':len(g)})
# portfolio rows
port=cv.groupby(['date','fold','model'],as_index=False)[['y','yhat']].sum(); port['hotel']='Portfolio'
cv_all=pd.concat([cv,port[cv.columns]],ignore_index=True)
sel_folds=['Fold 1','Fold 2','Fold 3']
sel=cv[cv.fold.isin(sel_folds)&cv.hotel.isin(HOTELS)].groupby(['model']).apply(metrics)['WAPE'].sort_values()
print('Selection WAPE (folds 1-3, hotel level):\n',sel.round(2))
ml=[m for m in sel.index if not m.startswith('Seasonal naive') and not m.startswith('Weekday')]
top3=ml[:3]; print('Ensemble members:',top3)
# ensemble rows
ens=cv[cv.model.isin(top3)].groupby(['date','hotel','fold'],as_index=False).agg(y=('y','first'),yhat=('yhat','mean')); ens['model']='Ensemble (top 3)'
cv=pd.concat([cv,ens[cv.columns]],ignore_index=True)
port=cv.groupby(['date','fold','model'],as_index=False)[['y','yhat']].sum(); port['hotel']='Portfolio'
cv_all=pd.concat([cv,port[cv.columns]],ignore_index=True)

# intervals from relative residuals (ensemble)
def rel_q(df_,qs):
    r=(df_.y-df_.yhat)/np.maximum(df_.yhat,1); return {q:np.quantile(r,q) for q in qs}
QS=[0.025,0.10,0.90,0.975]
E=cv[cv.model=='Ensemble (top 3)']
q_sel={h:rel_q(E[(E.hotel==h)&E.fold.isin(sel_folds)],QS) for h in HOTELS}
q_all={h:rel_q(E[E.hotel==h],QS) for h in HOTELS}
hold=E[E.fold=='Fold 4 (holdout)'].copy()
for h in HOTELS:
    m=hold.hotel==h
    for lab,(lo,hi) in {'80':(0.10,0.90),'95':(0.025,0.975)}.items():
        hold.loc[m,f'lo{lab}']=hold.loc[m,'yhat']*(1+q_sel[h][lo]); hold.loc[m,f'hi{lab}']=hold.loc[m,'yhat']*(1+q_sel[h][hi])
cover={h:{lab:float(((hold[hold.hotel==h].y>=hold[hold.hotel==h][f'lo{lab}'])&(hold[hold.hotel==h].y<=hold[hold.hotel==h][f'hi{lab}'])).mean()*100) for lab in ('80','95')} for h in HOTELS}
print('Holdout interval coverage:',cover)

# ---------- final fit on all data, forecast 90 days
FH=pd.date_range('2017-09-01',periods=90)
fin=[]; comps={}
for h in HOTELS:
    tr=Y[h]
    preds={}
    for mn in top3+['Seasonal naive (364d)']:
        preds[mn]=np.clip(np.asarray(MODELS[mn](tr,FH),dtype=float),0,None)
    ens_p=np.mean([preds[m] for m in top3],axis=0)
    d=pd.DataFrame({'date':FH,'hotel':h,'yhat':ens_p}); 
    for lab,(lo,hi) in {'80':(0.10,0.90),'95':(0.025,0.975)}.items():
        d[f'lo{lab}']=np.maximum(0,d.yhat*(1+q_all[h][lo])); d[f'hi{lab}']=d.yhat*(1+q_all[h][hi])
    for mn,p in preds.items(): d[mn]=p
    fin.append(d)
    m=Prophet(weekly_seasonality=True,yearly_seasonality=True,seasonality_mode='additive'); m.add_country_holidays('PT'); m.fit(pd.DataFrame({'ds':tr.index,'y':tr.values}))
    allf=m.predict(pd.DataFrame({'ds':pd.date_range(START,FH[-1])}))
    comps[h]={'weekly':allf.groupby(allf.ds.dt.dayofweek)['weekly'].mean(),'yearly':allf.assign(doy=allf.ds.dt.dayofyear).groupby('doy')['yearly'].mean(),'trend':allf.set_index('ds')['trend']}
final=pd.concat(fin,ignore_index=True)

def metrics_tbl(g):
    e=g.y-g.yhat; return pd.Series({'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),'WAPE_pct':e.abs().sum()/g.y.sum()*100,'Bias':e.mean()})
S=cv_all.groupby(['fold','hotel','model']).apply(metrics_tbl).reset_index().round(2)
S.to_csv('Hotel_Demand_Validation_Metrics.csv',index=False)
# portfolio intervals from portfolio ensemble residuals (all folds)
EP=cv_all[(cv_all.model=='Ensemble (top 3)')&(cv_all.hotel=='Portfolio')]
rp=(EP.y-EP.yhat)/np.maximum(EP.yhat,1); qp={q:np.quantile(rp,q) for q in QS}
pf=final.groupby('date',as_index=False)[['yhat']].sum(); pf['hotel']='Portfolio'
for lab,(lo,hi) in {'80':(0.10,0.90),'95':(0.025,0.975)}.items():
    pf[f'lo{lab}']=pf.yhat*(1+qp[lo]); pf[f'hi{lab}']=pf.yhat*(1+qp[hi])
out=pd.concat([final[['date','hotel','yhat','lo80','hi80','lo95','hi95']],pf[['date','hotel','yhat','lo80','hi80','lo95','hi95']]],ignore_index=True)
out=out.rename(columns={'yhat':'forecast'}).round(1); out.to_csv('Hotel_Demand_Forecast_Sep_Nov_2017.csv',index=False)
h4=S[(S.fold=='Fold 4 (holdout)')&(S.hotel=='Portfolio')].sort_values('WAPE_pct'); print(h4.to_string(index=False))
print('Saved Hotel_Demand_Validation_Metrics.csv and Hotel_Demand_Forecast_Sep_Nov_2017.csv | total time',round(time.time()-t0),'s')
