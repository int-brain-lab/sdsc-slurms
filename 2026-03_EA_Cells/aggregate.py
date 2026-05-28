# %%
from pathlib import Path
import tqdm

import pandas as pd
import numpy as np
CELLS_PATH = Path('/mnt/home/owinter/ceph/ea/cells')
AGG_PATH = Path('/mnt/home/owinter/ceph/ea/cells_aggregates')

df_clusters = []
stpc = []
stlfp = []
for fil in tqdm.tqdm(CELLS_PATH.rglob('clusters.pqt')):
    _df_clusters = pd.read_parquet(fil)
    _df_clusters['pid'] = fil.parent.parts[-1]
    df_clusters.append(_df_clusters)
    stpc.append(np.load(fil.parent.joinpath('stpc.npy')))
    stlfp.append(np.load(fil.parent.joinpath('stlfp.npy')))

# %%
df_clusters = pd.concat(df_clusters)
stpc = np.concatenate(stpc)
stlfp = np.concatenate(stlfp)
mask_good = df_clusters['bitwise_fail'] == 0
df_clusters_good = df_clusters.loc[mask_good, :]
print('all_clusters', df_clusters.shape)
print('good_clusters', df_clusters_good.shape)
print('good_stpc', stpc.shape)
print('good_stlfp', stlfp.shape)

# df_clusters (925251, 37)
# df_clusters_good (108606, 37)
# stpc (108606, 1000)
# stlfp (108606, 250)

# %%
df_clusters.to_parquet(AGG_PATH.joinpath('all_clusters.pqt'))
df_clusters_good.to_parquet(AGG_PATH.joinpath('good_clusters.pqt'))
np.save(AGG_PATH.joinpath('good_stpc.npy'), stpc)
np.save(AGG_PATH.joinpath('good_stlfp.npy'), stlfp)

