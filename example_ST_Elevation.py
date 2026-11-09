"""
ST-Elevation of icECG signal
============================

This example shows how to extract the ST-Elevation of an ECG signal

Author: Frédéric Waldmann <frederic.waldmann@fhnw.ch>
Date: 20.11.
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils import load_csv
from scipy.signal import butter, filtfilt, find_peaks

def find_QRS_peak(y, fs, f_hp=10, min_QRS_height=0.2, min_QRS_distance=0.3, moving_avg_len_sec=0.02):
    """
    Find QRS in ECG signal

    Find peaks in the high passed absolute signal with constraints

    Parameters
    ----------
    y : array_like
        signal
    fs : int
        sampling frequency
    f_hp : int, optional
        cutoff frequency of second order highpass filter, default=10
    min_QRS_height : float, optional
        peak threshold of absolute signal
    min_QRS_distance : float, optional
        min. distance between peaks, default=0.3
    moving_avg_len_sec: float, optional
        length of moving average filter in seconds, default=0.02

    Returns
    -------
    indexes : np.ndarray
        array of QRS indexes

    """

    # filter QRS signal with High-Pass
    b, a = butter(N=2, Wn=f_hp, btype='highpass', fs=fs)
    y_filt = filtfilt(b, a, y)

    # find peaks
    m = int(moving_avg_len_sec * fs)
    y_filt = np.convolve(y_filt, np.ones(m) / m, 'same')
    indexes, _ = find_peaks(np.abs(y_filt), height=min_QRS_height, distance=min_QRS_distance * fs)

    return indexes


def find_j_point(y, fs, QRS_indexes, min_bend=1.5e-3, lcr_threshold=0.4, min_delay_R2JP_sec=1.5e-2, max_delay_R2JP_sec=8e-2,
                 ab_sec=(-.01, 0.05), gs_sec=(.01, 0.1), show_debug_plot=False):
    """
    Finds J-Point in an ECG signal

    Algorithm uses an edge detection and a constraint peak find detection

    Parameters
    ----------
    y : array_like
        signal
    fs : int
        sampling frequency
    QRS_indexes : array_like
        indexes of QRS position in ecg signal `y`
    min_bend : float, optional
        minimum curve bending, default = 1.5e-3
    lcr_threshold : float, optional
        lcr threshold, default = 0.4
    min_delay_R2JP_sec: float, optional
        minimum delay between R-Peak and J-Point in seconds, default = 1.5e-2
    max_delay_R2JP_sec: float, optional
        maximum delay between R-Peak and J-Point in seconds, default = 8e-2
    ab_sec : tuple(float, float), optional
        length of left and right sided window for edge detection in seconds, default = (-0.01, 0.1)
    g_sec : tuple(float, float), optional
        area of left and right sided window for edge detection in seconds, default = (0.01, 0.1)
    show_debug_plot : bool, optional
        show the debug
    """

    K = len(y)
    ab = int(ab_sec[0] * fs), int(ab_sec[1] * fs)
    gs = int(gs_sec[0] * fs), int(gs_sec[1] * fs)
    cost = lm.TSLM.create_cost(ab, gs)

    # minimize squared error with linear constraint
    rls = lm.RLSAlssm(cost)
    rls.filter(y)
    x_0 = rls.minimize_x(lm.TSLM.H_Straight)
    x_1 = rls.minimize_x(lm.TSLM.H_Continuous)

    # LCR-Log-Cost Ratio
    J_0 = rls.eval_errors(x_0)
    J_1 = rls.eval_errors(x_1)

    lcr = -1 / 2 * np.log(np.divide(J_1, J_0))
    peak_indexes, _ = find_peaks(lcr, height=lcr_threshold)

    # Create Mask for JP Detection
    mask_bend = np.abs(x_1[:, 1] - x_1[:, 3]) >= min_bend  # edge slope change
    mask_peak = np.zeros((K,), dtype=bool)
    mask_peak[peak_indexes] = True  # peaks
    mask = np.bitwise_and(mask_bend, mask_peak)

    r2jp_min = int(min_delay_R2JP_sec * fs)
    r2jp_max = int(max_delay_R2JP_sec * fs)

    indexes_jpoints = []
    for QRS_index in QRS_indexes:
        k_start = max(QRS_index + r2jp_min, 0)
        k_end = min(QRS_index + r2jp_max, K - 1)
        tmp_ = mask[k_start:k_end]
        if any(tmp_):
            idx_ = k_end - np.argmax(np.flip(tmp_)) - 1
            indexes_jpoints.append(idx_)

    if show_debug_plot:

        yhat = cost.eval_alssm_output(x_0, alssm_weights=[0, 1])
        trajs_straight = lm.map_trajectories(cost.trajectories(x_0[indexes_jpoints]), indexes_jpoints, K, merge_ks=True, merge_seg=True)
        trajs_continuous = lm.map_trajectories(cost.trajectories(x_1[indexes_jpoints]), indexes_jpoints, K, merge_ks=True, merge_seg=True)
        time = np.arange(K)/fs
        mask_search_region = np.zeros(K, dtype=bool)
        for QRS_index in QRS_indexes:
            k_start = max(QRS_index + r2jp_min, 0)
            k_end = min(QRS_index + r2jp_max, K - 1)
            mask_search_region[k_start:k_end] = True

        fig_, (ax1, ax2) = plt.subplots(2, sharex='all')

        ax1.set_title('ECG with J-point using Edge Detection', fontsize=10)
        ax1.plot(time, y, c='grey', lw=1, label='ECG')
        ax1.plot(time, trajs_straight, c='blue', lw=1, label='Trajectory Straight')
        ax1.plot(time, trajs_continuous, c='red', lw=1, label='Trajectory Continuous')
        ax1.scatter(time[indexes_jpoints], yhat[indexes_jpoints],s=30, c='r', label='j-Point')

        ax2.set_title('Log Cost Ratio', fontsize=10)
        ax2.plot(time, lcr, c='k', lw=1, label='LCR')
        ax2.fill_between(time, 0, 1, where=mask_search_region, color='green', alpha=0.5, transform=ax2.get_xaxis_transform(), label='search area')
        ax2.scatter(time[indexes_jpoints], lcr[indexes_jpoints],s=30, c='r', label='j-Point detection')

        for ax in [ax1, ax2]:
            ax.legend(loc=1, fontsize=8)
        ax2.set_xlabel('Time [s]', fontsize=8)

    return indexes_jpoints


def find_iso_potential_line(y, fs, QRS_indexes, f_hp=0.5,
                            line_window_params=(-0.025, 0.025, 0.5),
                            isop_window_params=(-float('inf'), float('inf'), 0.125),
                            thd_energy=1.0, percentile_error=30,
                            buffer_len_sec=2.0, buffer_overlap=0.5,
                            duty_cycle_QRS=0.6,
                            lambda_reg=0.5,
                            show_debug_plot=False):

    """
    Finds iso-potential line of ECG Signal

    Parameters
    ----------
    y : array_like
        signal
    fs : int
        sampling frequency
    QRS_indexes : list of int
        QRS indexes
    f_hp : float, optional
        cut of frequency of 4 Order high-pass filter, default=0.5
    line_window_params: tuple(float, float, float), optional
        window parameters for line model, default=(-0.05, 0.001, 0.5)
        first and second parameter is length of the left and right window, resp.
        last parameter is the area of the window for both sides
    isop_window_params: tuple(float, float, float), optional
        window parameters for iso-potential line model, default=(-0.05, 0.001, 0.5)
        first and second parameter is length of the left and right window, resp.
        last parameter is the area of the window for both sides
    thd_energy : float, optional
        threshold for energy mask. default=0.005
    percentile_error : scalar, optional
        percentile of the error mask. default=30
    buffer_len_sec : float, optional
        length of buffer in seconds for adaptive error thresholding. default=2.0
    buffer_overlap : float, optional
        percentage of buffer overlap
    duty_cycle_QRS : float, optional
        percentage of duty cycle mask
    lambda_reg : float, optional
        regularization factor on second derivative for iso-potential line model
        (second degree polynomial), default=0.5
    show_debug_plot : bool, optional
        Show debug plot

    Returns
    -------
    y_iso_potential : np.ndarray, shape=(K,)
        Array of the iso-potential values
    """
    K = len(y)

    # filter QRS signal with High-Pass
    b, a = butter(N=4, Wn=f_hp, btype='highpass', fs=fs)
    y_filt = filtfilt(b, a, y)
    # approximate horizontal line model on filtered signal
    seg_l = lm.Segment(int(line_window_params[0] * fs), -1, lm.FW, int(line_window_params[2] * fs))
    seg_r = lm.Segment(0, int(line_window_params[1] * fs) - 1, lm.BW, int(line_window_params[2] * fs))
    cost_hl = lm.CompositeCost([lm.AlssmPoly(0)], [seg_l, seg_r], F=[[1, 1]])
    rls = lm.RLSAlssm(cost_hl)
    xs = rls.filter_minimize_x(y_filt)

    # mask where low signal energy
    mask_energy = rls.kappa < (thd_energy * fs)

    # create a mask where the error is low (adaptive thresholding)
    J = rls.eval_errors(xs)

    buffer_len = int(buffer_len_sec * fs)  # Buffer for recent data
    buffer_ov_len = int(buffer_len * buffer_overlap)
    buffer_range = np.arange(-buffer_len // 2, buffer_len // 2)
    buffer_thds = np.full(K, np.nan)

    for k in np.arange(buffer_ov_len, K - buffer_ov_len, buffer_ov_len):
        buffer_thds[k + buffer_range] = np.nanpercentile(J[k + buffer_range], percentile_error)

    mask_error = J < buffer_thds

    # Create QRS mask set a percentage of an QRS intervall to True
    mask_qrs = np.zeros(K, dtype=bool)
    ind_mid = np.add(np.diff(QRS_indexes, append=0) * duty_cycle_QRS, QRS_indexes).astype(int)
    for i, j in zip(ind_mid, QRS_indexes[1:]):
        mask_qrs[i:j] = True

    # estimate iso-potential line. This model approximation uses regularization on the second derivative
    # to minimize instability due to sparse signal.
    alssm_line = lm.AlssmPoly(2, C=np.array([[1, 0, 0], [0, 0, lambda_reg]]))
    a = int(isop_window_params[0] * fs) if ~ np.isinf(isop_window_params[0]) else -np.inf
    b = int(isop_window_params[1] * fs) if ~ np.isinf(isop_window_params[1]) else np.inf
    seg_l = lm.Segment(a, -1, lm.FW, int(isop_window_params[2] * fs))
    seg_r = lm.Segment(0, b - 1, lm.BW, int(isop_window_params[2] * fs))
    cost = lm.CompositeCost([alssm_line], [seg_l, seg_r], F=[[1, 1]])
    rls = lm.RLSAlssm(cost)


    # create regularization and sparse mask
    y_dtdt = np.diff(np.diff(y, append=y[-2:]))
    y_stack = np.column_stack([y, y_dtdt])

    mask_iso_potential = np.bitwise_and(mask_error, mask_energy)
    mask_iso_potential = np.bitwise_and(mask_iso_potential, mask_qrs)

    y_iso_potential = rls.filter_minimize_yhat(y_stack, mask_iso_potential)[:, 0]


    if show_debug_plot:
        time = np.arange(K)/fs
        y_hat = cost_hl.eval_alssm_output(xs)

        fig_, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')

        ax1.set_title('Approximate horizontal line', fontsize=10)
        ax1.plot(time, y_filt, c='k', lw=1, label='Filtered Signal')
        ax1.plot(time, y_hat, c='b', lw=1, label='approx. Signal')
        ax1.fill_between(time, 0, 1, where=mask_error, color='green', alpha=0.1, transform=ax1.get_xaxis_transform(), label='search area')

        ax2.set_title('Find good fit of horizontal line', fontsize=10)
        ax2.plot(time, J, c='k', lw=1, label='SE horizontal line')
        ax2.plot(time, buffer_thds, c='b', lw=1, label='adaptive threshold')

        ax3.set_title('Approximate sparse ECG signal with line (regulation on second derivative for stability)', fontsize=10)
        ax3.plot(time, y, c='grey', lw=0.5, label='ECG')
        ax3.scatter(time[mask_iso_potential], y[mask_iso_potential], s=7, c='k', label='masekd ECG')
        ax3.plot(time, y_iso_potential, c='b', lw=1, label='iso-potential line')
        ax3.fill_between(time, 0, 1, where=mask_iso_potential, color='green', alpha=0.1, transform=ax3.get_xaxis_transform(), label='search area')

        for ax in [ax1, ax2, ax3]:
            ax.legend(loc=1, fontsize=8)
        ax3.set_xlabel('Time [s]', fontsize=8)

    return y_iso_potential


SHOW_DEBUG_PLOT = True
fs = 2000
y = load_csv('icECG_raw_FS2000.csv')

QRS_indexes = find_QRS_peak(y, fs)
JPoint_indexes = find_j_point(y, fs, QRS_indexes, show_debug_plot=SHOW_DEBUG_PLOT)
y_iso_line = find_iso_potential_line(y, fs, QRS_indexes, show_debug_plot=SHOW_DEBUG_PLOT)
ST_elevation = y[JPoint_indexes]-y_iso_line[JPoint_indexes]

time = np.arange(len(y))/fs
fig, axs = plt.subplots(2 , sharex='all')
axs[0].plot(time, y, c='k', lw=1, label='ECG')
axs[0].axvline(time[QRS_indexes[0]], c='lightblue', lw=5, label='QRS', zorder=-1)
for QRS_index in QRS_indexes[1:]:
    axs[0].axvline(time[QRS_index], c='lightblue', lw=5, zorder=-1)

axs[0].scatter(time[JPoint_indexes], y[JPoint_indexes], c='m', label='JPoint')
axs[0].plot(time, y_iso_line, c='b', ls='--', label='Iso-Potential')
axs[1].plot(time[JPoint_indexes], ST_elevation, 'x-k', label='ST-Elevation')
axs[1].set_ylim([-1, 1])
for ax in axs:
    ax.legend(loc=1, fontsize=8)
    ax.grid('minor')
    ax.set_xlabel('Time [s]', fontsize=8)
plt.show()
