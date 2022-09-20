import os 
import yaml 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from datetime import date

sns.set_theme()
sns.set_context('paper')

SAVE_PATH = os.path.join('logs','small-results','ensemble pcam comparison',str(date.today()))
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
path_to_results = os.path.join('logs','ensemble_results','results_pcam.pkl')
df = pd.read_pickle(path_to_results)

#take out inits who are not to be shown 
df = df[~df['Initialisation'].isin(['Noise\nGHN_1','VQVAE'])]
df['Initialisation'] = df['Initialisation'].replace({'Noise\nGHN_0':'Noise GHN',"VQVAE\n+Pixel":"VQVAE*"})
df = df.rename({'acc':'Accuracy','nll':'Negative log-likelihood','brier':'Brier score','ece':'Expected calibration error'},axis='columns')



#'STOP'
sns.reset_defaults()
sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(20,9)})
sns.set(font_scale = 2.5)

for metric in ['Accuracy','Brier score','Expected calibration error','Negative log-likelihood']:
    plt.figure()
    b1 = sns.boxplot(data=df,y=metric,x='Initialisation',order=['Xavier','He','LVAE','CVAE','VQVAE*','TVAE','GHN','Noise GHN'])
    b1.set(ylabel=metric)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH,f'{metric}.pdf'))
plt.show()
