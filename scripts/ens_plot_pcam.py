import os 
import yaml 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from datetime import date

sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(8.5,5.5)})

dpi = 96
figure_width = 1
figure_height =2000
figsize=(figure_width/dpi,figure_width/dpi)

SAVE_PATH = os.path.join('logs','small-results',str(date.today()),'ensemble pcam comparison')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
path_to_results = os.path.join('logs','ensemble_results','results_pcam.pkl')
df = pd.read_pickle(path_to_results)

#take out inits who are not to be shown 
df = df[~df['Initialisation'].isin(['Noise\nGHN_1','VQVAE'])]
df['Initialisation'] = df['Initialisation'].replace({'Noise\nGHN_0':'Noise\nGHN'})
df = df.rename({'acc':'Accuracy','nll':'NLL','brier':'Brier Score','ece':'Expected Calibration Error'},axis='columns')


#'STOP'
sns.reset_defaults()
sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale = 1.9)

for metric in ['Accuracy','Brier Score','Expected Calibration Error','NLL']:
    plt.figure(dpi=96)
    b1 = sns.boxplot(data=df,y=metric,x='Initialisation',linewidth=0.6,fliersize=1.5)
    b1.set(ylabel=None)
    loc = 'lower right'
    plt.title(f'{metric}')
    plt.savefig(os.path.join(SAVE_PATH,f'{metric}.pdf'))
    plt.show()
