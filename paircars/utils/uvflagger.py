import numpy as np
import traceback
import os
import resource
import math
import psutil
from casatools import table, msmetadata, ms as casamstool
from joblib import Parallel, delayed as jobdelayed
from .flagging import do_flag_backup
from .ms_metadata import get_pol_names

#####################################################################################
# This code is adapted from SIMPL pipeline for LOFAR: Dey et al., 2025, A&A, 704, A75
# Main author: Soham Dey (NCRA-TIFR)
# Code is modified a little bit for adaption in P-AIRCARS
######################################################################################


def uvbin_flagger(
    uvwave, data, threshold=5.0, min_bin_samples=5, num_bins=50, binning_type="log"
):
    """
    Flag data based on binning in uv-wavelengths space, specifically designed for solar data.

    Parameters
    ----------
    uvwave: numpy.array
        UV distances in wavelengths (1D numpy array)
    data: numpy.array
        Visibility amplitudes (1D numpy array for this optimized version)
    threshold: float, optional
        Flagging threshold multiplier for MAD (default: 5.0)
    min_bin_samples: int, optional
        Minimum number of samples required in a bin (default: 5)
    num_bins: int, optional
        Number of logarithmic bins to use (default: 50)

    Returns
    -------
    numpy.array
        Boolean array with same shape as data indicating flagged points
    """
    if uvwave.shape[0] == 0 or data.shape[0] == 0:
        return np.zeros(data.shape, dtype=np.bool_)
    if uvwave.shape[0] != data.shape[0]:
        return np.zeros(data.shape, dtype=np.bool_)
    nrows = data.shape[0]
    flags = np.zeros(nrows, dtype=np.bool_)
    # Handle case with too few valid UV points or data points
    valid_mask = ~np.isnan(data) & ~np.isnan(uvwave)
    n_valid = np.sum(valid_mask)
    if n_valid < min_bin_samples * 2:
        return flags
    valid_uvwave = uvwave[valid_mask]
    valid_data = data[valid_mask]
    # Create non-uniform bins
    positive_uv = valid_uvwave[valid_uvwave > 0]
    if len(positive_uv) == 0:
        return flags
    min_uv = np.min(positive_uv)
    max_uv = np.max(valid_uvwave)
    # Check for invalid range before logspace
    if min_uv <= 0 or max_uv <= min_uv:
        return flags
    # Adaptive binning: more bins where data density is higher
    current_num_bins = num_bins
    if max_uv / min_uv > 1000:
        current_num_bins = min(100, num_bins * 2)
    # Ensure num_bins is at least 2 for logspace
    if current_num_bins < 2:
        current_num_bins = 2
    if binning_type == "log":
        bins = np.logspace(np.log10(min_uv), np.log10(max_uv), current_num_bins)
    else:
        bins = np.linspace(min_uv, max_uv, current_num_bins)
    # Bin the data and compute statistics
    # Indices correspond to bins[i-1] <= x < bins[i]
    # We want bins based on bin edges, so adjust indices.
    bin_indices = np.searchsorted(bins, valid_uvwave, side="right")
    # For each bin, compute median and MAD
    for bin_idx in range(
        current_num_bins
    ):  # Iterate through bin indices 0 to num_bins-1
        # Find data points belonging to this bin
        # bin_mask corresponds to points where bin_indices == bin_idx
        bin_mask = bin_indices == bin_idx
        bin_count = np.sum(bin_mask)
        if bin_count < min_bin_samples:  # Skip bins with too few points
            continue
        bin_data = valid_data[bin_mask]
        # Robust statistics
        # Use trimmed statistics if enough points
        if bin_count > 20:
            # Sort data and trim extreme values
            sorted_data = np.sort(bin_data)
            trim_size = max(1, int(bin_count * 0.05))  # Trim 5% from each end
            # Ensure trim_size doesn't exceed half the array size
            if 2 * trim_size >= bin_count:
                trim_size = max(
                    0, (bin_count - 1) // 2
                )  # Adjust trim size to avoid empty array
            if trim_size > 0:
                trimmed_data = sorted_data[trim_size:-trim_size]
                if trimmed_data.size == 0:  # Check if trimming resulted in empty array
                    bin_median = np.nanmedian(bin_data)  # Fallback to full median
                else:
                    bin_median = np.nanmedian(trimmed_data)
            else:
                bin_median = np.nanmedian(bin_data)
        else:
            bin_median = np.nanmedian(bin_data)
        # Use robust MAD calculation
        abs_diff = np.abs(bin_data - bin_median)
        bin_mad = np.nanmedian(abs_diff)
        # Avoid division by zero or very small MAD
        mad_threshold = 1e-9  # Adjusted threshold for comparison
        if bin_mad < mad_threshold:
            data_range = np.max(bin_data) - np.min(bin_data)
            if data_range < 1e-6:  # Adjusted threshold
                # Very uniform data, skip flagging
                continue
            else:
                # Use a fraction of the data range as MAD
                bin_mad = data_range * 0.1
                if (
                    bin_mad < mad_threshold
                ):  # Ensure MAD is not too small even after adjustment
                    bin_mad = mad_threshold
        # Adaptive threshold based on bin size and data spread
        adaptive_threshold = threshold
        if bin_count < 10:
            # Be more conservative with small bins
            adaptive_threshold *= 1.5
        # Flag outliers in this bin
        # Use the calculated adaptive_threshold and bin_mad
        # Ensure bin_mad is positive before division
        if bin_mad > 0:
            deviation = abs_diff / bin_mad
            outliers_mask = deviation > adaptive_threshold
        else:
            # Handle case where bin_mad is effectively zero after checks
            # Only flag if deviation from median is large in absolute terms?
            # Or skip flagging in this case. Let's skip for now.
            outliers_mask = np.zeros(bin_data.shape, dtype=np.bool_)
        # Only flag if not too many points would be flagged (avoid over-flagging)
        outlier_fraction = np.sum(outliers_mask) / bin_count if bin_count > 0 else 0.0
        final_outliers_mask = outliers_mask  # Start with the initial outlier mask
        if outlier_fraction > 0.4:  # If more than 40% would be flagged
            # This might be a real feature, not RFI - be more conservative
            if bin_mad > 0:  # Check bin_mad again
                extreme_outliers_mask = deviation > (adaptive_threshold * 2)
                if (
                    np.sum(extreme_outliers_mask) / bin_count < 0.2
                ):  # If less than 20% are extreme outliers
                    final_outliers_mask = (
                        extreme_outliers_mask  # Use the more conservative mask
                    )
                else:
                    # Skip flagging this bin - likely a real feature
                    final_outliers_mask = np.zeros(
                        bin_data.shape, dtype=np.bool_
                    )  # Clear flags for this bin
            else:
                # Skip flagging if MAD is zero and outlier fraction is high
                final_outliers_mask = np.zeros(bin_data.shape, dtype=np.bool_)
        # Map back to original indices within the valid data subset
        # Find indices within valid_data that correspond to this bin
        bin_indices_in_valid_data = np.where(bin_mask)[0]
        # Apply the final outlier mask to these indices
        outlier_indices_in_bin = bin_indices_in_valid_data[final_outliers_mask]
        # Map these indices back to the original data array size (nrows)
        # Find the original indices corresponding to the valid data
        original_indices_of_valid_data = np.where(valid_mask)[0]
        # Get the original indices of the outliers
        original_outlier_indices = original_indices_of_valid_data[
            outlier_indices_in_bin
        ]
        # Set flags in the main flags array
        flags[original_outlier_indices] = True
    return flags


def _process_timestamp(
    time_indices,
    uv_distances_full,
    data_full,
    flags,
    nchan,
    npol,
    threshold=5.0,
    num_bins=50,
    binning_type="log",
):
    """Processes a single timestamp's data to generate flags."""
    timestamp_flags = np.zeros((len(time_indices), nchan, npol), dtype=bool)
    timestamp_uv = uv_distances_full[time_indices]
    for chan in range(nchan):
        for pol in range(npol):
            # Get data for this timestamp, channel, and polarization
            # Extract data only for the relevant indices
            timestamp_data = data_full[time_indices, chan, pol]
            timestamp_flag = flags[time_indices, chan, pol]
            timestamp_data[timestamp_flag] = np.nan
            # Skip if too many NaNs or too few points
            valid_data_mask = ~np.isnan(timestamp_data)
            n_valid = np.sum(valid_data_mask)
            if n_valid < 10:  # Use a reasonable minimum number of points
                continue
            # Apply UV-based flagging only on this timestamp's valid data
            # Pass only the valid subset to uvbin_flagger
            current_flags_real = uvbin_flagger(
                timestamp_uv[valid_data_mask],
                np.real(timestamp_data[valid_data_mask]),
                threshold=threshold,
                min_bin_samples=5,  # Keep min_bin_samples consistent
                num_bins=num_bins,
                binning_type=binning_type,
            )
            current_flags_imag = uvbin_flagger(
                timestamp_uv[valid_data_mask],
                np.imag(timestamp_data[valid_data_mask]),
                threshold=threshold,
                min_bin_samples=5,  # Keep min_bin_samples consistent
                num_bins=num_bins,
                binning_type=binning_type,
            )
            current_flags = current_flags_real | current_flags_imag
            # Update flags for this timestamp, channel, and polarization
            if current_flags is not None and len(current_flags) > 0:
                # Map flags back to the original size of the timestamp slice
                valid_indices_in_timestamp = np.where(valid_data_mask)[0]
                timestamp_flags[
                    valid_indices_in_timestamp[current_flags], chan, pol
                ] = True
            del timestamp_data
    return time_indices, timestamp_flags


def flagger(
    msname,
    datacolumn,
    threshold=5.0,
    n_threads=4,
    absmem=-1,
    num_bins=50,
    binning_type="log",
    flagbackup=True,
    verbose=False,
):
    """
    Flagger optimized for solar observations

    Parameters
    ----------
    msname: str
        Measurement set.
    datacolumn: str
        Name of the data column (e.g. 'DATA', 'CORRECTED_DATA', 'RESIDUAL').
    threshold: float, optional
        Multiplier for the MAD-based flagging threshold.
    n_threads: int, optional
        Number of processes for parallel processing.
    num_bins: int, optional
        Number of UV bins for uvbin_flagger (default: 50)
    binning_type : str, optional
        Binning type (linear or log)
    flagbackup : bool, optional
        Take flag backup or not

    Returns
    --------
    int
        Success message
    int
        Total flag points
    int
        Total new flag points
    """
    if flagbackup:
        do_flag_backup(msname, flagtype="uvflag")
    GB = 1024.0**3
    datacolumn = datacolumn.lower()
    tb = table()
    tb.open(msname)
    colnames = tb.colnames()
    tb.close()
    if datacolumn=="corrected" or datacolumn=="corrected_data" and "CORRECTED_DATA" in colnames:
        datacolumn="corrected_data"
    elif datacolumn=="residual" or datacolumn=="residual_data":
        if "MODEL_DATA" in colnames:
            if "CORRECTED_DATA" in colnames:
                datacolumn="residual_data"
            else:
                datacolumn="obs_residual_data"
        else:
            if "CORRECTED_DATA" in colnames:
                datacolumn="corrected_data"
            else:
                datacolumn="data"
    else:
        datacolumn="data"
        
    msmd = msmetadata()
    msmd.open(msname)
    npol = msmd.ncorrforpol()[0]
    nchan = msmd.nchan(0)
    nrows = int(msmd.nrows())
    msmd.close()
    pol_list = get_pol_names(msname)
    mstool = casamstool()
    mstool.open(msname)
    uvw = mstool.getdata("UVW")["uvw"]
    uv = np.sqrt(uvw[0,...]**2+uvw[1,...]**2)
    del uvw
    timecol = mstool.getdata("TIME")["time"]
    mstool.close()
    os.system(f"rm -rf {msname}/flag.dat")
    mm_flag = np.memmap(f"{msname}/flag.dat",dtype=np.bool_,mode="w+",shape=(npol,nchan,nrows))
    mm_flag.flush()
    os.system(f"rm -rf {msname}/uv.dat")
    mm_uv = np.memmap(f"{msname}/uv.dat",dtype=uv.dtype,mode="w+",shape=uv.shape)
    mm_uv[:] = uv
    mm_uv.flush()
    del uv
    os.system(f"rm -rf {msname}/time.dat")
    mm_time = np.memmap(f"{msname}/time.dat",dtype=timecol.dtype,mode="w+",shape=timecol.shape)
    mm_time[:] = timecol
    mm_time.flush()
    unique_times = np.unique(timecol)
    del timecol
    
    try:
        mstool.open(msname)
        pol_chunk = npol
        chan_chunk = nchan
        if absmem>0:    
            process = psutil.Process(os.getpid())
            used_mem=process.memory_info().rss/GB
            absmem-=used_mem
            all_row_datasize = (nrows*16)/GB
            all_row_flagsize = nrows/GB
            all_row_uvsize = (nrows*8)/GB
            total_row_size = 4*(all_row_datasize + all_row_flagsize + all_row_uvsize) 
            if absmem<total_row_size:
                print("Not enough memory allocated to process.")
                return 1, None, None
            nchunk = math.ceil(absmem/total_row_size)
            pol_chunk = min(npol, nchunk)
            remaining = math.ceil(nchunk / pol_chunk)
            chan_chunk = min(nchan, remaining)
        if verbose:
            print(f"Total chunk: {nchunk}, Polarization chunk: {pol_chunk}, channel chunk: {chan_chunk}.")
        
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        njobs = min(int(0.5*soft),nchunk)
        old_flag = 0
        
        for p in range(0,npol,pol_chunk):
            start_pol_index = p
            end_pol_index = min(npol,start_pol_index+pol_chunk)
            for c in range(0,nchan,chan_chunk):
                start_chan = c
                end_chan = min(nchan,start_chan+chan_chunk)
                if verbose:
                    print(f"Reading data -- Polarization index range: {start_pol_index}~{end_pol_index}, "
                    f"Channel range: {start_chan}~{end_chan}")
                mstool.selectinit(datadescid=0,reset=True)
                mstool.selectpolarization(pol_list[start_pol_index:end_pol_index])
                mstool.selectchannel(start=start_chan,nchan=end_chan-start_chan)
                data = mstool.getdata(datacolumn)[datacolumn]
                flag = mstool.getdata("FLAG")["flag"]
                data = data.T
                flag = flag.T
                
                old_flag+=np.sum(flag)  # Initial number of flags
                data_actual_shape = data.shape
                if len(data_actual_shape) == 3:
                    n_rows, n_chan, n_pol = data_actual_shape
                else:
                    raise ValueError(
                        f"Unexpected data dimensions in column '{datacolumn}'. "
                        f"Expected 3 (rows, chans, pols), got {len(data_actual_shape)} with shape {data_actual_shape}."
                    )
                
                # Calculate UV distances in wavelengths
                uv_distances = mm_uv.T
                # Get indices that would sort the time column
                sorted_indices = np.argsort(mm_time, kind="stable")  # Stable sort preserves original order for ties
                # Get the sorted times and the locations where the time changes
                sorted_times = mm_time[sorted_indices]
                unique_times_sorted, split_points = np.unique(sorted_times, return_index=True)
                # Split the sorted_indices array at these change points
                # This gives lists of original indices, grouped by timestamp
                indices_split_by_time = np.split(sorted_indices, split_points[1:])
                # Create the map from unique time to its corresponding array of indices
                time_indices_map = dict(zip(unique_times_sorted, indices_split_by_time))
                # Verify unique_times consistency (optional check)
                if not np.array_equal(np.sort(unique_times), unique_times_sorted):
                    print(
                        "Warning: Unique times from initial scan and sorting differ. Using sorted unique times."
                    )
                    unique_times = unique_times_sorted  # Use the times derived from sorting
                    
                job_args = []
                # Prepare arguments for each parallel job
                job_args = [
                    (
                        time_indices_map[timestamp],
                        uv_distances,
                        data,
                        flag,
                        n_chan,
                        n_pol,
                        threshold,
                        num_bins,
                        binning_type,
                    )
                    for timestamp in unique_times
                    if len(time_indices_map[timestamp]) >= 10  # Filter small timestamps here
                ]

                if len(job_args)>0:  # Only run if there are jobs
                    if verbose:
                        print(f"Starting {njobs} parallel flagging jobs.")
                    results = Parallel(n_jobs=njobs, backend="threading")(
                        jobdelayed(_process_timestamp)(*args) for args in job_args
                    )
                    # --- Combine Results ---
                    # Create a copy of flags to update, or update in place if acceptable
                    for time_indices, timestamp_flags in results:
                        # Update the main flags array using the indices and the returned flags
                        # Ensure shapes match before assignment
                        if flag[time_indices, :, :].shape == timestamp_flags.shape:
                            flag[time_indices, :, :] = np.logical_or(
                                flag[time_indices, :, :], timestamp_flags
                            )
                            del timestamp_flags
                        else:
                            print(
                                f"Warning: Shape mismatch when combining flags for indices {time_indices}. Skipping update."
                            )
                            print(
                                f"Expected shape: {flag[time_indices, :, :].shape}, Got shape: {timestamp_flags.shape}"
                            ) 
                mm_flag[start_pol_index:end_pol_index,start_chan:end_chan,...] |= flag.T 
                del flag, data                        
                      
        mstool.done()                    
        mstool.close() 
        # Number of additional flagged data points
        n_final_flagged = np.sum(mm_flag)
        n_additional_flagged = n_final_flagged - old_flag

        ################################
        # Putting flags
        ################################
        tb=table()
        tb.open(msname,nomodify=False)
        nrows = tb.nrows()
        mm_flag = mm_flag.reshape(npol,nchan,-1)
        if absmem>0:
            per_row_size = (npol*nchan)/GB
            writing_chunk = math.ceil(absmem/per_row_size)
            for i in range(0,nrows,writing_chunk):
                start_row = i
                end_row = min(nrows,i+writing_chunk)
                if verbose:
                    print(f"Writing final flag rows: {start_row}:{end_row}")
                tb.putcol("FLAG",mm_flag[...,start_row:end_row],startrow=start_row,nrow=end_row-start_row)
                tb.flush()
        else:
            if verbose:
                print("Writing final flags.")
            tb.putcol("FLAG",mm_flag[:])
            tb.flush()
        tb.close()
        return 0, n_final_flagged, n_additional_flagged
    except Exception:
        traceback.print_exc()
        return 1, None, None
    
