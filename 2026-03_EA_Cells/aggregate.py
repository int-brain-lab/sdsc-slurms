# %%
from pathlib import Path
import tqdm

import pandas as pd
import numpy as np

OUTPUT_PATH = Path(f'/mnt/home/owinter/ceph/ea/cells_aggregates')

df_clusters = []
stpc = []
stlfp = []
for fil in tqdm.tqdm(OUTPUT_PATH.rglob('clusters.pqt')):
    _df_clusters = pd.read_parquet(fil)
    _df_clusters['pid'] = fil.parent.parts[-1]
    df_clusters.append(_df_clusters)
    stpc.append(np.load(fil.parent.joinpath('stpc.npy')))
    stlfp.append(np.load(fil.parent.joinpath('stlfp.npy')))

# %%
df_clusters = pd.concat(df_clusters)
stpc = np.concatenate(stpc)
stlfp = np.concatenate(stlfp)
df_clusters_good = df_clusters.loc[df_clusters['bitwise_fail'] == 0, :]
print('df_clusters_good', df_clusters_good.shape)
print('stpc', stpc.shape)
print('stlfp', stlfp.shape)

# %%
df_clusters.to_parquet(OUTPUT_PATH.joinpath('all_clusters.pqt'))
df_clusters_good.to_parquet(OUTPUT_PATH.joinpath('good_clusters.pqt'))
np.save(OUTPUT_PATH.joinpath('good_stpc.npy'), stpc)
np.save(OUTPUT_PATH.joinpath('good_stlfp.npy'), stlfp)
