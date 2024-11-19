import numpy as np
import torch

def calc_labits(xs, ys, ts, framesize, t_range, num_bins=5, norm=True):
    """
    Calculate the global bidirectional time surface for each event:
    For each pixel in the entire CMOS, find the timestamp of the event that is closest to the current event on the time axis in the past.
    If there is no event in the past at this pixel, find the timestamp of the event that is closest to the current event on the time axis in the future.
    Args:
        xs: x coordinates of events, numpy array, shape: (N,)
        ys: y coordinates of events, numpy array, shape: (N,)
        ts: timestamps of events, a sorted numpy array, shape: (N,)
        framesize: the size of the CMOS, tuple, shape: (H, W)
        t_range: the time range of the time surface, int
        num_bins: the number of bins for the time surface, int
        norm: whether to normalize the time surface, bool
    Returns:
        time_surface: the local bidirectional time surface for the current event, numpy array, shape: (2*r+1, 2*r+1)
    """
    H, W = framesize
    # print("INSIDE CALC LABITS: t_range: ", t_range, "num_bins: ", num_bins)
    
    # Return pure -1 if no events are present
    if t_range == 0:
        time_surface = np.full((num_bins, H, W), -1)
        return time_surface
    else:
        # Array dimensions for the time surface
        time_surface = np.full((num_bins, H, W), -np.inf)  # Use np.inf as a placeholder for unset values
    
    # Calculate the relative positions of ts
    t_cur = 0
    
    # Get indices before and after the current event
    cur_idx = 0 # np.searchsorted(ts[prev_idx:], t_cur, side='right') + prev_idx
    past_indices = np.array([], dtype=int)
    
    for bidx in range(num_bins):
        # Array dimensions for the time surface of the current bin
        time_surface_bin = np.full((H, W), -np.inf)  # Use np.inf as a placeholder for unset values
        t_cur = t_cur.item() if isinstance(t_cur, torch.Tensor) else t_cur
        t_norm = ts - t_cur 
        # Get indices before and after the current event
        after_idx = np.searchsorted(ts[cur_idx:], t_cur + t_range, side='right') + cur_idx
        future_indices = np.arange(cur_idx, after_idx)
    
        # Update time_surface for past events (choose minimum time difference for each pixel)
        if len(past_indices) > 0:
            np.maximum.at(time_surface_bin, (ys[past_indices], xs[past_indices]), t_norm[past_indices])

        # Temporary array to store future time differences, keeping only those cells that are still inf in time_surface
        if len(future_indices) > 0:
            future_time_surface_bin = np.full_like(time_surface_bin, np.inf)
            np.minimum.at(future_time_surface_bin, (ys[future_indices], xs[future_indices]), t_norm[future_indices])

        # Combine past and future times, only filling future times where past times were not updated
        mask = np.isinf(time_surface_bin)  # Find where past updates have not occurred
        if len(future_indices) > 0:
            time_surface_bin[mask] = future_time_surface_bin[mask]

        # Replace any remaining np.inf with -t_range (indicating no events found in either direction)
        time_surface_bin[np.isinf(time_surface_bin)] = -t_range
    
        time_surface[bidx] = time_surface_bin
        t_cur += t_range
        cur_idx = after_idx
        past_indices = future_indices
    
    # Normalize to [-1, 1], while keep the empty cells as -1
    if norm:
        time_surface = time_surface / t_range
    
    return time_surface