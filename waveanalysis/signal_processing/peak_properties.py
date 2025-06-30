import warnings
import numpy as np
import scipy.signal as sig

warnings.filterwarnings("ignore") # Ignore warnings

def calc_indv_peak_props_workflow(bin_values:np.ndarray, img_props:dict) -> dict:
    '''
    Calculate individual peak properties for each channel and bin.

    Args:
        bin_values (np.ndarray): The input array of bin values.
        img_props (dict): A dictionary containing image properties.

    Returns:
        dict: A dictionary containing the calculated individual peak properties, with the following keys:
            - "peak_widths" (np.ndarray): Mean peak widths for each channel and bin.
            - "peak_maxs" (np.ndarray): Mean peak maxima for each channel and bin.
            - "peak_mins" (np.ndarray): Mean peak minima for each channel and bin.
            - "peak_offsets" (np.ndarray): Mean peak offsets from the midpoint between bases.
            - "incr_rates" (np.ndarray): Mean rate of signal increase from left base to peak.
            - "dec_rates" (np.ndarray): Mean rate of signal decrease from peak to right base.
            - "ddx_maxs" (np.ndarray): Maximum first derivative (d(signal)/dt) on rising slope.
            - "ddx_mins" (np.ndarray): Minimum first derivative (d(signal)/dt) on falling slope.
            - "peak_props" (dict): Dictionary of full individual peak properties (signal, peaks, prominences, base positions, etc.) per channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']

    # Initialize arrays to store the individual peak properties
    indv_peak_widths = np.full((num_channels, num_bins), np.nan)
    indv_peak_maxs = np.full((num_channels, num_bins), np.nan)
    indv_peak_mins = np.full((num_channels, num_bins), np.nan)
    indv_peak_offsets = np.full((num_channels, num_bins), np.nan)
    indv_incr_rates = np.full((num_channels, num_bins), np.nan)
    indv_dec_rates = np.full((num_channels, num_bins), np.nan)
    indv_ddx_maxs = np.full((num_channels, num_bins), np.nan)
    indv_ddx_mins = np.full((num_channels, num_bins), np.nan)
    indv_peak_props = {}

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
            signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)                 
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # Find the first derivatve for this signal function
            signal_ddx = np.gradient(signal)
            signal_ddx = sig.savgol_filter(signal_ddx, window_length = 11, polyorder = 2)    

            # Add a constant (mean signal) to the derivate values to put it on a similar y-axis value for plotting
            avg_sig = np.mean(signal)
            signal_ddx_corr = signal_ddx * 2 + avg_sig

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                # Calculate the peak properties
                # IMPORTANT: leftIndex and rightIndex refer to the indexes of the left and right midpoints of the peak
                widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                # IMPORTANT: leftTrough and rightTrough refer to the indexes of the left and right troughs, or bases, of the peak
                proms, leftTrough, rightTrough = sig.peak_prominences(signal, peaks, 61) #TODO: currently this is hardcoded to the value 61, should have some sort of calculation to detemine the interval

                # Helper method that calculates the extrema for the derivative of the signal plot
                # (ddx is used as a shorthand to refer to the derivative plot)
                ddx_maxs, ddx_mins = compute_derivative_props(signal_ddx, peaks, leftTrough, rightTrough)

                # For each peak, calculate the differences in signal and time between the left trough, peak, and right trough
                left_signal_difference = signal[peaks] - signal[leftTrough]
                right_signal_difference = signal[peaks] - signal[rightTrough]
                left_trough_to_peak_distances = peaks - leftTrough
                peaks_to_right_trough_distances = peaks - rightTrough

                # Calculate the average rate of signal change by dividing the "rise" (change in signal) by the "run" (change in time)
                incr_rates = left_signal_difference / left_trough_to_peak_distances
                dec_rates = right_signal_difference / peaks_to_right_trough_distances
                
                # Calculate the mean of the peak widths, maximums, and minimums
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)

                # Calculate the left and right bases of the peaks, then midpoints and peak offsets
                _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=.99)
                midpoints = (leftIndex + rightIndex) / 2
                peak_offsets = peaks - midpoints

                # Check if one peak entirely encompasses another
                for i in range(len(peaks)):
                    for j in range(len(peaks)):
                        if i != j:  # Avoid self-comparison
                            if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                                # Peak j is entirely encompassed by peak i
                                left_bases[i] = np.nan
                                right_bases[i] = np.nan
                                peak_offsets[i] = np.nan
                                midpoints[i] = np.nan
                
                # Drop NaN values because it will mess up the mean calculation
                valid_indices = ~np.isnan(peak_offsets)
                valid_offsets = peak_offsets[valid_indices]
                valid_indices = ~np.isnan(ddx_maxs)
                valid_ddx_maxs = ddx_maxs[valid_indices]
                valid_indices = ~np.isnan(ddx_mins)
                valid_ddx_mins = ddx_mins[valid_indices]

                # Calculate the mean of valid peak offsets
                mean_offset = np.nanmean(valid_offsets)

                # Calculate the mean of valid peak to base values
                mean_incr_rate = np.nanmean(incr_rates)
                mean_dec_rate = np.nanmean(dec_rates)

                # Calculate the mean of valid rate extrema
                mean_ddx_max = np.nanmean(valid_ddx_maxs)
                mean_ddx_min = np.nanmean(valid_ddx_mins)

            else:
                # If no peaks detected, return NaNs
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                mean_offset = np.nan
                mean_incr_rate = np.nan
                mean_dec_rate = np.nan
                mean_ddx_max = np.nan
                mean_ddx_min = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftIndex = np.nan
                rightIndex = np.nan
                midpoints = np.nan
                peak_offsets = np.nan
                left_bases = np.nan
                right_bases = np.nan

            # Store the mean peak properties in the arrays
            indv_peak_widths[channel, bin] = mean_width
            indv_peak_maxs[channel, bin] = mean_max
            indv_peak_mins[channel, bin] = mean_min
            indv_peak_offsets[channel, bin] = mean_offset
            indv_incr_rates[channel, bin] = mean_incr_rate
            indv_dec_rates[channel, bin] = mean_dec_rate
            indv_ddx_maxs[channel, bin] = mean_ddx_max
            indv_ddx_mins[channel, bin] = mean_ddx_min

            # Store the individual peak properties in the dictionary
            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                                'derivative': signal_ddx,
                                                                'derivativeCorrected': signal_ddx_corr,
                                                                'averageSignal': avg_sig,
                                                                'peaks': peaks,
                                                                'proms': proms, 
                                                                'heights': heights, 
                                                                'leftIndex': leftIndex, 
                                                                'rightIndex': rightIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_base': left_bases,
                                                                'right_base': right_bases}
                        
                        # TODO: rename the keys to be more descriptive
    return {
        "peak_widths": indv_peak_widths,
        "peak_maxs": indv_peak_maxs,
        "peak_mins": indv_peak_mins,
        "peak_offsets": indv_peak_offsets,
        "incr_rates": indv_incr_rates,
        "dec_rates": indv_dec_rates,
        "ddx_maxs": indv_ddx_maxs,
        "ddx_mins": indv_ddx_mins,
        "peak_props": indv_peak_props
    }

def calc_indv_peak_props_rolling(signal: np.ndarray) -> tuple:
    '''
    Calculate the individual peak properties of a signal using rolling window.

    Parameters:
        signal (np.ndarray): The input signal.

    Returns:
        tuple: A tuple containing the mean width, mean maximum, mean minimum, and mean offset of the peaks. If no peaks are detected, NaN values are returned.
    '''
    # Calculate the peak properties
    signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)                 
    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

    # If peaks detected, calculate properties, otherwise return NaNs
    if len(peaks) > 0:
        # Calculate the peak properties
        widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
        proms, _, _ = sig.peak_prominences(signal, peaks)
        # Calculate the mean of the peak widths, maximums, and minimums
        mean_width = np.mean(widths, axis=0)
        mean_max = np.mean(signal[peaks], axis = 0)
        mean_min = np.mean(signal[peaks]-proms, axis = 0)

        # calculate the left and right bases of the peaks, then midpoints and peak offsets
        _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=.99)
        midpoints = (leftIndex + rightIndex) / 2
        peak_offsets = peaks - midpoints
        # Check if one peak entirely encompasses another
        for i in range(len(peaks)):
            for j in range(len(peaks)):
                if i != j:  # Avoid self-comparison
                    if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                        # Peak j is entirely encompassed by peak i
                        left_bases[i] = np.nan
                        right_bases[i] = np.nan
                        peak_offsets[i] = np.nan
                        midpoints[i] = np.nan
        
        # Drop NaN values because it will mess up the mean calculation
        valid_indices = ~np.isnan(peak_offsets)
        valid_offsets = peak_offsets[valid_indices]
        # Calculate the mean of valid peak offsets
        mean_offset = np.nanmean(valid_offsets)
    else:
        # If no peaks detected, return NaNs
        mean_width = np.nan
        mean_max = np.nan
        mean_min = np.nan
        mean_offset = np.nan

    return mean_width, mean_max, mean_min, mean_offset


def compute_derivative_props(signal_ddx, peaks, left_troughs, right_troughs):
    """
    Compute peak derivative extrema for all peaks.

    Returns:
        ddx_maxs (np.ndarray): Max d(signal)/dt from left trough to peak
        ddx_mins (np.ndarray): Min d(signal)/dt from peak to right trough
    """

    ddx_mins = np.full(peaks.shape, np.nan)
    ddx_maxs = np.full(peaks.shape, np.nan)

    len_of_peaks = len(peaks)

    for i in range(1, len_of_peaks - 1):
        left_segment = signal_ddx[int(left_troughs[i]):int(peaks[i])]
        right_segment = signal_ddx[int(peaks[i]):int(right_troughs[i])]
        left_segment = left_segment[~np.isnan(left_segment)]
        right_segment = right_segment[~np.isnan(right_segment)]

        if left_segment.size > 0 and right_segment.size > 0:
            ddx_maxs[i] = np.max(left_segment)
            ddx_mins[i] = np.min(right_segment)

    return ddx_maxs, ddx_mins
