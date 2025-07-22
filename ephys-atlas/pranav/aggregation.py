from pathlib import Path
from ephysatlas.aggregation import aggregate_all_probes, produce_output_dataframes
from ephysatlas.data import read_features_from_disk

# Aggregations will be stored here.
OUTDIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics/aggregations/")

# Output of the feature compuation run. This should contain folder for each PID.
INP_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics/output")


if __name__ == "__main__":
    # Get probe level dirs.
    # This logic can be changed based on which probe_ids you want to include.
    # Here I am checking whether the channels.pqt file is present inside the probe_directory, to assess a valid run.
    probe_level_dirs = [dir for dir in INP_DIR.iterdir() if (dir/"channels.pqt").is_file()]
    # print(len(probe_level_dirs), probe_level_dirs[0])
    
    #Get snippets_df which contains information on feature version, start_times , durations etc, and direcory locations.
    full_snippets_df = aggregate_all_probes(probe_level_dirs)

    #Subset based on the criteria.
    before_inj_snippets = full_snippets_df[full_snippets_df['t_start'] <= 350]
    after_inj_snippets = full_snippets_df[full_snippets_df['t_start'] >= 2950]

    dict1 = {'agg_full': full_snippets_df , 'agg_before': before_inj_snippets, 'agg_after': after_inj_snippets}

    # Run the aggregation steps
    for fold_name,snippets_df in dict1.items():
        print(f"Doing calculations for {fold_name}, {snippets_df.shape}")
        #Creates the output dataframes for the channels, raw ephys and features.
        df_channels, df_raw_ephys, df_features_denoise = produce_output_dataframes(snippets_df, input_dir=INP_DIR, output_dir=OUTDIR / fold_name)

        #Produce the merged version of the features, and check if the output is correct.
        df_merged = read_features_from_disk(OUTDIR / fold_name)

        #Outputs the merged dataset as well.
        df_merged.to_parquet(OUTDIR / fold_name / "df_all_cols_merged.pqt")
