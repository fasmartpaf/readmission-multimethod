import numpy as np, json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

X = np.load('X.npy'); y = np.load('y.npy')
cv = StratifiedKFold(5, shuffle=True, random_state=42)
T = 2.776  # t_{0.975, df=4}

def make(name):
    if name=='Logistic Regression': return LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1), True
    if name=='Random Forest': return RandomForestClassifier(n_estimators=200, max_depth=16, class_weight='balanced', n_jobs=-1, random_state=42), False
    return 'hgb', False

out={}
for name in ['Logistic Regression','Random Forest','HistGradientBoosting']:
    aucs,prs,briers=[],[],[]
    for tr,va in cv.split(X,y):
        Xtr,Xva,ytr,yva=X[tr],X[va],y[tr],y[va]
        if name=='HistGradientBoosting':
            w=np.where(ytr==1,(len(ytr)-ytr.sum())/ytr.sum(),1.0)
            m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_depth=6,random_state=42).fit(Xtr,ytr,sample_weight=w)
            p=m.predict_proba(Xva)[:,1]
        else:
            m,scale=make(name)
            if scale:
                s=StandardScaler().fit(Xtr); Xtr2,Xva2=s.transform(Xtr),s.transform(Xva)
            else:
                Xtr2,Xva2=Xtr,Xva
            m.fit(Xtr2,ytr); p=m.predict_proba(Xva2)[:,1]
        aucs.append(roc_auc_score(yva,p)); prs.append(average_precision_score(yva,p)); briers.append(brier_score_loss(yva,p))
    def ci(v):
        v=np.array(v); m=v.mean(); h=T*v.std(ddof=1)/np.sqrt(len(v)); return [round(m,4),round(h,4)]
    out[name]={'roc_auc':ci(aucs),'pr_auc':ci(prs),'brier':ci(briers)}
    print(f"{name}: AUC {out[name]['roc_auc'][0]}±{out[name]['roc_auc'][1]} | PR {out[name]['pr_auc'][0]}±{out[name]['pr_auc'][1]} | Brier {out[name]['brier'][0]}±{out[name]['brier'][1]}")

json.dump(out, open('cv_results.json','w'), indent=1)
print('SAVED cv_results.json')
