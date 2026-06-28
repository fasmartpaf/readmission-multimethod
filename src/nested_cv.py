import numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

X = np.load('X.npy'); y = np.load('y.npy')
spw = (len(y)-y.sum())/y.sum()  # scale_pos_weight for imbalance
grid = {'num_leaves':[31,63], 'learning_rate':[0.05], 'n_estimators':[150]}
base = LGBMClassifier(objective='binary', scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=42)

outer = StratifiedKFold(5, shuffle=True, random_state=42)
inner = StratifiedKFold(3, shuffle=True, random_state=1)
t=time.time(); outer_auc=[]; best_params=[]
for k,(tr,te) in enumerate(outer.split(X,y),1):
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=1)
    gs.fit(X[tr], y[tr])
    p = gs.predict_proba(X[te])[:,1]
    a = roc_auc_score(y[te], p); outer_auc.append(a); best_params.append(gs.best_params_)
    print(f'outer fold {k}: AUC={a:.4f} best={gs.best_params_} ({time.time()-t:.0f}s)', flush=True)

m=np.mean(outer_auc); h=2.776*np.std(outer_auc,ddof=1)/np.sqrt(5)
res={'method':'LightGBM (real library) with nested 5x3-fold CV',
     'nested_cv_auc_mean':round(float(m),4),'nested_cv_auc_ci95':round(float(h),4),
     'outer_fold_aucs':[round(float(a),4) for a in outer_auc],'best_params_per_fold':best_params,
     'scale_pos_weight':round(float(spw),3),'grid':grid}
print('NESTED CV AUC: %.4f ± %.4f'%(m,h))

# final tuned model on the standard 80/20 split for reporting
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
gs=GridSearchCV(base,grid,scoring="roc_auc",cv=inner,n_jobs=1).fit(Xtr,ytr)
p=gs.predict_proba(Xte)[:,1]
res['final_best_params']=gs.best_params_
res['holdout_auc']=round(roc_auc_score(yte,p),4)
res['holdout_pr_auc']=round(average_precision_score(yte,p),4)
print('Final tuned LightGBM holdout AUC',res['holdout_auc'],'PR-AUC',res['holdout_pr_auc'],'params',gs.best_params_)
json.dump(res, open('nestedcv_results.json','w'), indent=1)
print('SAVED nestedcv_results.json')
