import argparse
from pathlib import Path

import numpy as np

from pykilosort.ibl import ibl_pykilosort_params, run_spike_sorting_ibl
from viz.reports import qc_plots_metrics


def slurm_pykilosort(cbin_file, output_dir):

    cbin_file = Path(cbin_file)
    scratch_dir = Path.home().joinpath("scratch", 'pykilosort')  # temporary path on which intermediate raw data will be written, we highly recommend a SSD drive
    ks_output_dir = Path(output_dir)  # path containing the kilosort output unprocessed
    alf_path = ks_output_dir.joinpath('alf')  # this is the output standardized as per IBL standards (SI units, ALF convention)

    if not ks_output_dir.exists():
        # loads parameters and run
        params = ibl_pykilosort_params(cbin_file)
        params['Th'] = [6, 3]
        if args.run_label == 'RUN_00_ks2':
            params['normalisation'] = 'whitening'
            params['preprocessing_function'] = 'ks2'
        if args.run_label == 'RUN_01_original':
            params['normalisation'] = 'original'
        if args.run_label == 'RUN_02_whitening':
            params['normalisation'] = 'whitening'
        elif args.run_label == 'RUN_03_zscore':
            params['normalisation'] = 'zscore'
        elif args.run_label == 'RUN_04_global_zscore':
            params['normalisation'] = 'global_zscore'

        run_spike_sorting_ibl(cbin_file, delete=True, scratch_dir=scratch_dir,
                              ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='INFO', params=params)

    if not alf_path.joinpath('qc').exists():
        print(f"computing metrics for {ks_output_dir}")
        qc_plots_metrics(
            out_path=alf_path.joinpath('qc'),
            bin_file=cbin_file,
            pykilosort_path=alf_path,
            raster_plot=True,
            raw_plots=True,
            summary_stats=True,
            # raster_start=0.0,
            # raster_len=1200.0,
            raw_start=600.0,
            raw_len=0.04,
        )


if __name__ == "__main__":
    # python slurm_pykilosort.py ---cbin_file /mnt/s0/spikesorting/benchmark/CSH_ZAD_026/2020-09-04/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin
    parser = argparse.ArgumentParser(description='Run batch spike sorter job')
    parser.add_argument('--cbin_file', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    slurm_pykilosort(args.cbin_file, args.output_dir)
