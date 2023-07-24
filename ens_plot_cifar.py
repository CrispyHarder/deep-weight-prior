import os 
import yaml 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from datetime import date

sns.set_theme()
sns.set_context('paper')
sns.set(rc={'figure.figsize':(20,11)}) #sns.set(rc={'figure.figsize':(12,11)})
sns.set(font_scale = 2.5)

SAVE_PATH = os.path.join('logs','small-results','ensemble cifar comparison',str(date.today()))
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    
path_to_results = os.path.join('logs','ensemble_results','resultsrtz.pkl')
df = pd.read_pickle(path_to_results)

#take out inits who are not to be shown 
df = df[df['n_members']==5] #only with 5 members
df['Initialisation'] = df['Initialisation'].replace({'Noise\nGHN_0':'Noise GHN',"VQVAE\n+Pixel":"VQVAE*","CVAE":"VAE"}) #rename some of the columns/initialisations
df = df.rename({'acc':'Accuracy','nll':'Negative log-likelihood','brier':'Brier score','ece':'Expected calibration error'},axis='columns') #rename some columns

corr_df = df[df['Corruption level']>0.5] #only results on corrpted data 
corr_df[~corr_df['Initialisation'].isin(['Noise\nGHN_1','VQVAE','TVAE','LVAE','Noise GHN'])] #dont take these initialisations  corr_df = corr_df[~corr_df['Initialisation'].isin(['Noise\nGHN_1','VQVAE','TVAE','LVAE','Xavier','He','VAE','VQVAE*'])]#  

METRICS = ['Accuracy','Expected calibration error']# ['Accuracy','Negative log-likelihood','Brier score','Expected calibration error']


# pal = sns.color_palette('deep')
# print(pal.as_hex())

for metric in METRICS:
    plt.figure()
    #
    ax = sns.boxplot(data=corr_df,x='Corruption level',y=metric,hue='Initialisation',linewidth=0.6,fliersize=1.5, hue_order=['Xavier','He','VAE','VQVAE*','GHN']) 
    #ax = sns.boxplot(data=corr_df,x='Corruption level',y=metric,hue='Initialisation',hue_order=['GHN','Noise GHN'],palette=['#8172b3','#937860']) 
    [ax.axvline(x+.5,color='white') for x in ax.get_xticks()]
    ax.set(title='CIFAR-C')
    loc = 'upper right' if metric in ['Accuracy'] else 'lower right'
    plt.legend(loc=loc)
    plt.savefig(os.path.join(SAVE_PATH,f'{metric} ens cifar c.pdf'))
plt.show()


# # #Here comes the part for uncorrupted data
# unc_df = df[df['Corruption level']==0]
# unc_df = unc_df[~unc_df['Initialisation'].isin(['Noise\nGHN_1','VQVAE'])]

# for metric in METRICS:
#     plt.figure()
#     b1 = sns.boxplot(data=unc_df,y=metric,x='Initialisation',order=['Xavier','He','LVAE','CVAE','VQVAE*','TVAE','GHN','Noise GHN'])
#     #b1.set(ylabel=None)
#     loc = 'lower right'
#     plt.savefig(os.path.join(SAVE_PATH,f'{metric} ens cifar.pdf'))
# plt.show()
