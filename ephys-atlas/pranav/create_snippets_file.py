import pandas as pd
import numpy as np
from pathlib import Path

# Set the duration for the snippets (Sets a constant duration for each snippet)
DURATION = 5
OUTPUT_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics")


def main():
    df = pd.read_csv(OUTPUT_DIR / 'psychedlics_pids.csv')

    list_df = []
    for index, row in df.iterrows():
        df_temp = pd.DataFrame(columns=['pid', 'eid', 'probe_name', 'snippet_index', 't_start', 'duration'])
        # Set the start times for the snippets.
        start_times = np.array([300,310,320,330,340,2960,2970,2980,2990,3000])
        df_temp['snippet_index'] = np.arange(len(start_times))
        df_temp['t_start'] = start_times
        df_temp['pid'] = row['pid']
        df_temp['eid'] = row['eid']
        df_temp['probe_name'] = row['probe_name']
        df_temp['duration'] = DURATION
        list_df.append(df_temp)

    df = pd.concat(list_df)
    df.to_csv(OUTPUT_DIR / 'psychedlics_snippets_df.csv', index=False)


if __name__ == "__main__":
    main()