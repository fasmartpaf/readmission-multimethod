import numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

X = np.load('X.npy'); y = np.load('y.npy')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def hgb(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, random_state=42)

# uncalibrated (no class weighting, so probabilities are meaningful)
base = hgb().fit(Xtr, ytr); p_unc = base.predict_proba(Xte)[:,1]
platt = CalibratedClassifierCV(hgb(), method='sigmoid', cv=5).fit(Xtr, ytr); p_platt = platt.predict_proba(Xte)[:,1]
iso = CalibratedClassifierCV(hgb(), method='isotonic', cv=5).fit(Xtr, ytr); p_iso = iso.predict_proba(Xte)[:,1]

res={'brier':{}, 'auc':{}, 'reliability':{}}
for nm,p in [('Uncalibrated',p_unc),('Platt',p_platt),('Isotonic',p_iso)]:
    res['brier'][nm]=round(brier_score_loss(yte,p),4)
    res['auc'][nm]=round(roc_auc_score(yte,p),4)
    frac,mean_pred=calibration_curve(yte,p,n_bins=10,strategy='quantile')
    res['reliability'][nm]={'mean_pred':list(np.round(mean_pred,4)),'frac_pos':list(np.round(frac,4))}
print('Brier:', res['brier']); print('AUC:', res['auc'])

# Decision Curve Analysis using isotonic-calibrated probabilities
p = p_iso; n=len(yte); prev=yte.mean()
pts=np.round(np.arange(0.02,0.51,0.01),3)
nb_model=[]; nb_all=[]
for pt in pts:
    pred=(p>=pt).astype(int)
    tp=((pred==1)&(yte==1)).sum(); fp=((pred==1)&(yte==0)).sum()
    nb_model.append(tp/n - (fp/n)*(pt/(1-pt)))
    nb_all.append(prev - (1-prev)*(pt/(1-pt)))
res['dca']={'pt':list(pts),'net_benefit_model':list(np.round(nb_model,5)),
            'net_benefit_all':list(np.round(nb_all,5)),'prevalence':round(float(prev),4)}
# range where model beats both treat-all and treat-none(0)
better=[pt for pt,nm,na in zip(pts,nb_model,nb_all) if nm>max(na,0)+1e-6]
res['dca']['model_best_range']=[float(min(better)),float(max(better))] if better else None
print('DCA model-best threshold range:', res['dca']['model_best_range'])
json.dump(res, open('caldca_results.json','w'), indent=1)

# figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAVY,BLUE,TEAL,GREY,LIGHT='#0B2545','#1B6CA8','#0D9488','#64748B','#E2E8F0'
# calibration before/after
fig,ax=plt.subplots(figsize=(6,5.4),dpi=200)
ax.plot([0,1],[0,1],ls='--',color='#94A3B8',lw=1,label='Perfectly calibrated')
cols={'Uncalibrated':GREY,'Platt':BLUE,'Isotonic':TEAL}
for nm in ['Uncalibrated','Platt','Isotonic']:
    r=res['reliability'][nm]; ax.plot(r['mean_pred'],r['frac_pos'],marker='o',ms=5,lw=2,color=cols[nm],label=f"{nm} (Brier {res['brier'][nm]})")
ax.set_title('Calibration Before and After Recalibration',fontsize=12.5,weight='bold',color=NAVY,pad=10)
ax.set_xlabel('Mean predicted probability',fontsize=11,weight='bold'); ax.set_ylabel('Observed frequency',fontsize=11,weight='bold')
ax.grid(True,color=LIGHT); [ax.spines[s].set_visible(False) for s in ['top','right']]
ax.legend(frameon=False,fontsize=9,loc='upper left')
plt.tight_layout(); plt.savefig('charts2/f13_calibration_ba.png',bbox_inches='tight'); plt.close()
# DCA
fig,ax=plt.subplots(figsize=(7,4.6),dpi=200)
ax.plot(pts,nb_model,color=TEAL,lw=2.5,label='Gradient boosting (isotonic)',zorder=3)
ax.plot(pts,nb_all,color=GREY,lw=1.8,ls='--',label='Treat all',zorder=2)
ax.axhline(0,color=NAVY,lw=1.5,ls=':',label='Treat none',zorder=2)
ax.set_title('Decision Curve Analysis',fontsize=13,weight='bold',color=NAVY,pad=10)
ax.set_xlabel('Threshold probability',fontsize=11,weight='bold'); ax.set_ylabel('Net benefit',fontsize=11,weight='bold')
ax.set_ylim(-0.02, max(nb_model)+0.01); ax.grid(True,color=LIGHT); [ax.spines[s].set_visible(False) for s in ['top','right']]
ax.legend(frameon=False,fontsize=9.5); plt.tight_layout(); plt.savefig('charts2/f14_dca.png',bbox_inches='tight'); plt.close()
print('saved f13_calibration_ba.png, f14_dca.png')
