import matplotlib.pyplot as plt
import numpy as np

from analysis_tools.image_analysis import get_rod_borders

# get rod border positions
def get_positions(img_lst) :
    outer_left_pos = []; inner_left_pos = []; inner_right_pos = []

    for i in range(len(img_lst)) :
        oL, iL, iR, _, _ = get_rod_borders(img_lst[i])
        outer_left_pos.append(oL); inner_left_pos.append(iL); inner_right_pos.append(iR)

        print(f"\ranalyzing images {i+1}/{len(img_lst)}", end="", flush=True)
    print()
    return outer_left_pos, inner_left_pos, inner_right_pos

def nitinol_stiffness(l_rod) :
    return (0.75 * np.pi * 75e9 * (0.155 * 1e-3)**4) / (l_rod * 1e-3)**3

def get_displacements_strains_forces(f_pos, r_pos, rigid_stop, w_gel, l_rod, end_type) :
    x_plot = np.arange(len(f_pos))

    # get displacements and convert from pixels to μm
        # 1 μm / 265 pixels
    f_d = [(n - f_pos[0])/265 for n in f_pos]; r_d = [(n - r_pos[0])/265 for n in r_pos]

    # find starting frame 
    for start_idx in range(len(r_d)) :
        if r_d[start_idx] > 0 : 
            start_idx -= 1
            break

    # keep only positive displacements, prepare for fitting
    r_x_fit = []; r_y_fit = []
    for i in range(start_idx, rigid_stop - 1) :
        if r_d[i] > 0 :
            r_x_fit.append(x_plot[i])
            r_y_fit.append(r_d[i])

    # get fitted displacements for rigid rod, assumed to be linear
    r_slope, y_int = np.polyfit(r_x_fit, r_y_fit, deg=1)
    r_fit = np.multiply(x_plot, r_slope) + y_int

    # get fitted positions for rigid rod and convert to pixels to calculate lengths
    r_fit_pos = [r_pos[start_idx] + (n * 265) for n in r_fit]

    # calculate lengths
    lengths = [((r_fit_pos[i] - f_pos[i])/265) - 1 for i in range(len(r_fit_pos))]

    # calculate strain
    strain = [(l-lengths[0])/lengths[0] for l in lengths]

    # find stress cutoff frame
    if end_type != "stretch" :
        strain_diff = np.diff(strain) 
        for stress_cutoff_idx in range(len(strain_diff)) :
            if strain_diff[stress_cutoff_idx] == max(strain_diff) : break
    else : stress_cutoff_idx = -1

    # calculate stress
    k = nitinol_stiffness(l_rod)
    stress = [(k * d)/(2 * w_gel) for d in f_d] if not np.isnan(w_gel) else [0]

    # plotting
    analysis_fig, axs = plt.subplots(1, 3 if not np.isnan(w_gel) else 2, figsize=(15,5))
    
    axs[0].plot(x_plot[start_idx:], f_d[start_idx:], label="flexible", marker=".", color="blue", alpha=0.5)
    axs[0].plot(x_plot[start_idx:], r_d[start_idx:], label="rigid", marker=",", color="red", alpha=0.5)
    axs[0].plot(x_plot[start_idx:], r_fit[start_idx:], color="red", linestyle="--", linewidth=2)

    axs[0].set_xticks(np.arange(0, len(f_pos), 5)); axs[0].set_xlabel("time (frame)")
    axs[0].set_yticks(np.arange(0, max(r_fit), 0.25)); axs[0].set_ylabel("rod displacement (mm)")
    axs[0].grid()
    axs[0].legend()
    
    axs[1].plot(np.arange(len(strain)), strain, label="strain", marker=".", color="blue")
    axs[1].plot(np.arange(len(stress)), stress, label="stress", marker=".", color="red")

    axs[1].set_xticks(np.arange(0, len(f_pos), 5)); axs[1].set_xlabel("time (frame)")
    axs[1].set_yticks(np.arange(0, max([max(stress), max(strain)]), 0.1))
    axs[1].grid()
    axs[1].legend()

    if len(stress) > 1 :
        axs[2].plot(strain, stress, marker=".", color="blue")
        axs[2].plot(strain[:stress_cutoff_idx], stress[:stress_cutoff_idx], color="red")
        axs[2].set_xlabel("strain"); axs[2].set_ylabel("stress")
        axs[2].grid()

    plt.tight_layout()
    #plt.show()

    return (
        {
            "flexible position" : f_pos,
            "rigid position" : r_pos, 
            "flexible displacement" : f_d,
            "rigid displacement" : r_d,
            "fitted rigid displacement" : r_fit,
            "lengths" : lengths,
            "strains" : strain,
            "stress" : stress if len(stress) > 1 else [np.nan] * len(strain),
            "lower bound" : [0] + [np.nan] * (len(stress) - 1),
            "upper bound" : [stress_cutoff_idx] + [np.nan] * (len(stress) - 1)
        }, 
        analysis_fig
    )

# get 1 y-value per x-value
def sort_unique_strain_stress(x,y) :
    sort_idx = np.argsort(x)
    sorted_x = x[sort_idx]; sorted_y = np.array(y)[sort_idx]

    unique_x, unique_idx, counts = np.unique(sorted_x, return_index=True, return_counts=True)

    # use the mean to get 1 y-value
    unique_y = np.array([
        np.mean(sorted_y[idx : idx + count])
        for idx, count in zip(unique_idx, counts)
    ])
    return unique_x, unique_y