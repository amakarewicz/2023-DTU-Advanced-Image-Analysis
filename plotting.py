#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
for lr in ['1e-03', '1e-04', '1e-05']:
    res = pd.read_csv(f'./results/results_64_ch_{lr}_lr_64_bs.csv')
    plt.plot(res['train_loss'], label=f'{lr}_train')
    plt.plot(res['valid_loss'], label=f'{lr}_valid')

plt.legend()
plt.show()
# %%
