import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score

d = pd.read_csv('pkg/results/clean.csv', low_memory=False).reset_index(drop=True)
ite = np.load('test_idx.npy'); p = np.load('best_probas.npy'); y = np.load('y.npy')
adv = json.load(open('advanced_results.json')); thr = adv['res']['models']['HistGradientBoosting']['threshold']
te = d.iloc[ite].copy(); te['y'] = y[ite]; te['p'] = p; te['pred'] = (p >= thr).astype(int)

def metrics(sub):
    if sub['y'].nunique() < 2 or len(sub) < 200: return None
    tp = ((sub.pred==1)&(sub.y==1)).sum(); fn=((sub.pred==0)&(sub.y==1)).sum()
    fp = ((sub.pred==1)&(sub.y==0)).sum(); tn=((sub.pred==0)&(sub.y==0)).sum()
    return {'n':int(len(sub)),'prevalence':round(sub['y'].mean(),3),
            'roc_auc':round(roc_auc_score(sub['y'],sub['p']),3),
            'recall':round(tp/(tp+fn),3) if (tp+fn) else None,
            'fpr':round(fp/(fp+tn),3) if (fp+tn) else None,
            'selection_rate':round(sub['pred'].mean(),3)}

res={'threshold':round(float(thr),3),'overall':metrics(te),'by_age':{},'by_gender':{},'by_race':{}}
# age bands
te['ageband']=pd.cut(te['age_ord'],bins=[0,45,65,85,100],labels=['<45','45-64','65-84','85+'])
for g,sub in te.groupby('ageband',observed=True):
    m=metrics(sub);
    if m: res['by_age'][str(g)]=m
for g,sub in te.groupby('gender'):
    m=metrics(sub)
    if m: res['by_gender'][str(g)]=m
for g,sub in te.groupby('race'):
    m=metrics(sub)
    if m: res['by_race'][str(g)]=m
json.dump(res, open('fairness_results.json','w'), indent=1)
print('threshold',res['threshold'])
print('overall',res['overall'])
print('by_race:'); [print(' ',k,v) for k,v in res['by_race'].items()]
print('by_gender:'); [print(' ',k,v) for k,v in res['by_gender'].items()]
print('by_age:'); [print(' ',k,v) for k,v in res['by_age'].items()]

# figure: recall + AUC by race
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAVY,BLUE,TEAL,LIGHT="#0B2545","#1B6CA8","#0D9488","#E2E8F0"
races=list(res['by_race']); auc=[res['by_race'][r]['roc_auc'] for r in races]; rec=[res['by_race'][r]['recall'] for r in races]
x=np.arange(len(races)); w=0.38
fig,ax=plt.subplots(figsize=(7.5,4.3),dpi=200)
ax.bar(x-w/2,auc,w,label='ROC-AUC',color=TEAL,zorder=3); ax.bar(x+w/2,rec,w,label='Recall @ threshold',color=BLUE,zorder=3)
ax.set_xticks(x); ax.set_xticklabels([r.replace('AfricanAmerican','African Am.') for r in races],fontsize=9,rotation=15)
ax.set_title('Subgroup Performance by Race (HistGradientBoosting)',fontsize=12.5,weight='bold',color=NAVY,pad=10)
ax.set_ylabel('Score',fontsize=11,weight='bold'); ax.set_ylim(0,0.8); ax.yaxis.grid(True,color=LIGHT,zorder=0); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False,fontsize=9,ncol=2,loc='upper center')
plt.tight_layout(); plt.savefig('charts2/f11_fairness.png',bbox_inches='tight'); plt.close()
print('saved charts2/f11_fairness.png')
