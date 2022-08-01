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

SAVE_PATH = os.path.join('logs','small-results',str(date.today()),'ensemble comparison')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
path_to_results = os.path.join('logs','ensemble_results','resultsrtz.pkl')
df = pd.read_pickle(path_to_results)

#take out inits who are not to be shown 
df = df[~df['Initialisation'].isin(['Noise\nGHN_1','VQVAE'])]
df['Initialisation'] = df['Initialisation'].replace({'Noise\nGHN_0':'Noise\nGHN'})
df = df.rename({'acc':'Accuracy','nll':'NLL','brier':'Brier Score','ece':'Expected Calibration Error'},axis='columns')

METRICS = ['Accuracy','NLL','Brier Score','Expected Calibration Error']
for metric in METRICS:
    for n_members in [5]:
        rel_df = df[ df['n_members']==n_members]
        #plt.figure(figsize=figsize,dpi=dpi)
        plt.figure(dpi=96)
        sns.boxplot(data=rel_df,x='Corruption level',y=metric,hue='Initialisation',linewidth=0.6,fliersize=1.5)
        loc = 'upper right' if metric in ['Accuracy'] else 'lower right'
        plt.legend(loc=loc)
        plt.title(f'{metric} of ensembles with {n_members} members on corrupted CIFAR-10 data')
        plt.savefig(os.path.join(SAVE_PATH,f'{metric}{n_members}members.pdf'))
        plt.show()


#'STOP'
sns.reset_defaults()
sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale = 1.9)

metric = 'Accuracy'
n_members = 5 
rel_df = df[ df['n_members']==n_members]
corr_1_df = rel_df[ rel_df['Corruption level']==1 ]

plt.figure(dpi=96)
b1 = sns.boxplot(data=corr_1_df,y=metric,x='Initialisation',linewidth=0.6,fliersize=1.5)
b1.set(ylabel=None)
loc = 'lower right'
plt.title(f'{metric}')
plt.savefig(os.path.join(SAVE_PATH,f'{metric}single1.pdf'))
plt.show()

metric = 'Brier Score'
n_members = 5 
rel_df = df[ df['n_members']==n_members]
corr_1_df = rel_df[ rel_df['Corruption level']==1 ]

plt.figure(dpi=96)
b2 = sns.boxplot(data=corr_1_df,y=metric,x='Initialisation',linewidth=0.6,fliersize=1.5)
b2.set(ylabel=None)
loc = 'upper right'
plt.title(f'{metric}')
plt.savefig(os.path.join(SAVE_PATH,f'{metric}single1.pdf'))
plt.show()

