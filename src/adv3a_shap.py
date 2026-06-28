import numpy as np, json, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

X = np.load('X.npy'); y = np.load('y.npy'); feat = json.load(open('pkg/results/feat_names.json'))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, max_depth=16, class_weight='balanced', n_jobs=-1, random_state=42).fit(Xtr, ytr)

rng = np.random.default_rng(0); samp = rng.choice(len(Xte), 500, replace=False)
t = time.time()
expl = shap.TreeExplainer(rf)
sv = expl.shap_values(Xte[samp])
# binary RF -> take positive class
if isinstance(sv, list): sv = sv[1]
elif sv.ndim == 3: sv = sv[:, :, 1]
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:12]
shap_top = [[feat[i], round(float(mean_abs[i]), 5)] for i in order]
print('SHAP done', round(time.time()-t,1),'s')
for n,v in shap_top: print(' ', n, v)
json.dump({'shap_top': shap_top}, open('shap_results.json','w'), indent=1)

# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAVY,TEAL,LIGHT="#0B2545","#0D9488","#E2E8F0"
fig,ax=plt.subplots(figsize=(7,4.6),dpi=200)
labels=[n.replace('_',' ')[:26] for n,_ in shap_top][::-1]; vals=[v for _,v in shap_top][::-1]
ax.barh(labels,vals,color=TEAL,zorder=3)
ax.set_title('SHAP Feature Importance (mean |SHAP|, Random Forest)',fontsize=12.5,weight='bold',color=NAVY,pad=10)
ax.set_xlabel('mean |SHAP value|',fontsize=11,weight='bold'); ax.xaxis.grid(True,color=LIGHT,zorder=0); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('charts2/f10_shap.png',bbox_inches='tight'); plt.close()
print('saved charts2/f10_shap.png')
