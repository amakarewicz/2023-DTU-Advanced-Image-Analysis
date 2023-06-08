#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
# for lr in ['1e-03', '1e-04', '1e-05']:
for t in range(1,4):
    res = pd.read_csv(f'./results/results_{t}_trans_32_ch_1e-05_lr.csv')
    plt.plot(res['train_loss'], label=f'{t}_train')
    plt.plot(res['valid_loss'], label=f'{t}_valid')

plt.legend()
plt.show()
# %%
results = pd.DataFrame()
LR = ['1e-3', '1e-4', '1e-5']
for i, lr in enumerate(['1e-03', '1e-04', '1e-05']):
    for t, n in enumerate(['no augmentation', 'augmentation 1', 'augmentation 2']):
        for ch in [16,32,64]:
            res = pd.read_csv(f'./results/results_{t+1}_trans_{ch}_ch_{lr}_lr.csv')
            last = res.tail(1)
            last['learning rate'] = LR[i]
            last['num channels'] = str(ch)+' channels'
            last['augmentation'] = n
            results = pd.concat([results, last])

#%%
# sns.stripplot(x="learning_rate", y="valid_loss", hue="augmentation",
            #    data=results, palette="Set2", dodge=False, jitter=0.0)    
# plt.title("Accuracy depending on learning rate", size=14);
# fig, ax = plt.subplots(figsize=(9,6))
sns.set(style='darkgrid', font_scale=1.2, rc={'figure.figsize':(11.7,8.27)})

g = sns.catplot(data=results.rename(columns={'augmentation':'', 'valid_acc':'validation accuracy'}),
            x="learning rate",
            y="validation accuracy",
            hue="",
            col="num channels",
            aspect=1.4, height=6,
            jitter=False,
            s=8)
# catplot.legend_.set_title(None)
# # plt.legend(loc='lower center')
sns.move_legend(g, 'lower center', bbox_to_anchor=(0.4, -0.3))
# g.set_titles(['16 channels', '32_channels'])
axes = g.axes.flatten()
axes[0].set_title("16 channels")
axes[1].set_title("32 channels")
axes[2].set_title("64 channels")
plt.show()
# %%
res_200 = pd.read_csv('./results/results_200_epochs3_trans_32_ch_1e-04_lr.csv')
plt.plot(res_200['train_loss'])
plt.plot(res_200['valid_loss'])
# %%
