import numpy as np, json, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, roc_curve)
from sklearn.calibration import calibration_curve

X = np.load('X.npy'); y = np.load('y.npy')
idx = np.arange(len(y))
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, idx, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
np.save('test_idx.npy', ite)

def best_threshold(yv, p):  # Youden's J
    fpr, tpr, th = roc_curve(yv, p); j = np.argmax(tpr - fpr); return th[j]

models = {}
t=time.time()
lr = LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1).fit(Xtr_s, ytr)
rf = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight='balanced', n_jobs=-1, random_state=42).fit(Xtr, ytr)
# class imbalance for HistGB via sample_weight
w = np.where(ytr==1, (len(ytr)-ytr.sum())/ytr.sum(), 1.0)
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, random_state=42).fit(Xtr, ytr, sample_weight=w)
probas = {'Logistic Regression': lr.predict_proba(Xte_s)[:,1],
          'Random Forest': rf.predict_proba(Xte)[:,1],
          'HistGradientBoosting': hgb.predict_proba(Xte)[:,1]}
print('fit done', round(time.time()-t,1),'s')

res={'n_test':int(len(yte)),'pos_rate':round(float(y.mean()),4),'models':{}}
for name,p in probas.items():
    thr = best_threshold(yte, p); pred = (p>=thr).astype(int)
    res['models'][name]={
        'accuracy':round(accuracy_score(yte,pred),4),'precision':round(precision_score(yte,pred,zero_division=0),4),
        'recall':round(recall_score(yte,pred,zero_division=0),4),'f1':round(f1_score(yte,pred,zero_division=0),4),
        'roc_auc':round(roc_auc_score(yte,p),4),'pr_auc':round(average_precision_score(yte,p),4),
        'mcc':round(matthews_corrcoef(yte,pred),4),'brier':round(brier_score_loss(yte,p),4),
        'threshold':round(float(thr),3),'cm':confusion_matrix(yte,pred).tolist()}
    print(name,'AUC',res['models'][name]['roc_auc'],'PR',res['models'][name]['pr_auc'],'MCC',res['models'][name]['mcc'],'Brier',res['models'][name]['brier'])

# bootstrap AUC CIs + paired diffs
rng=np.random.default_rng(7); B=1000; n=len(yte)
boot={}; names=list(probas)
aucs={nm:[] for nm in names}; diffs={'HistGradientBoosting-Logistic Regression':[], 'HistGradientBoosting-Random Forest':[]}
for _ in range(B):
    bi=rng.integers(0,n,n)
    if yte[bi].sum()==0 or yte[bi].sum()==len(bi): continue
    a={nm:roc_auc_score(yte[bi],probas[nm][bi]) for nm in names}
    for nm in names: aucs[nm].append(a[nm])
    diffs['HistGradientBoosting-Logistic Regression'].append(a['HistGradientBoosting']-a['Logistic Regression'])
    diffs['HistGradientBoosting-Random Forest'].append(a['HistGradientBoosting']-a['Random Forest'])
for nm in names:
    lo,hi=np.percentile(aucs[nm],[2.5,97.5]); res['models'][nm]['auc_ci']=[round(lo,4),round(hi,4)]
res['auc_diff']={}
for k,v in diffs.items():
    lo,hi=np.percentile(v,[2.5,97.5]); p_gt0=float(np.mean(np.array(v)<=0))
    res['auc_diff'][k]={'mean':round(float(np.mean(v)),4),'ci':[round(lo,4),round(hi,4)],'p_two_sided':round(2*min(p_gt0,1-p_gt0),4)}
print('AUC diff HGB-LR', res['auc_diff']['HistGradientBoosting-Logistic Regression'])

# calibration curves
cal={}
for nm,p in probas.items():
    frac,mean_pred=calibration_curve(yte,p,n_bins=10,strategy='quantile')
    cal[nm]={'mean_pred':list(np.round(mean_pred,4)),'frac_pos':list(np.round(frac,4))}
json.dump({'res':res,'cal':cal}, open('advanced_results.json','w'), indent=1)
np.save('best_probas.npy', probas['HistGradientBoosting'])
print('SAVED advanced_results.json')
