from typing import Tuple

import numpy as np
import pandas as pd

from reachaq.core.reach_event import ReachEventOutcome, ReachEventMethod
from reachaq.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)


def segment_reaches_f11(
    *,
    df_3d: pd.DataFrame,
    frame_rate: int,
    pellet_home: Tuple[float, float, float],
    debug: int,
):
    #define dist and velo for each reach sequence
    pellet_xyz_p = df_3d['Pellet']
    pellet_p = pellet_xyz_p['p']
    pellet_x_vals = pellet_xyz_p['x'].values
    pellet_y_vals = pellet_xyz_p['y'].values
    pellet_z_vals = pellet_xyz_p['z'].values
    dist_p = np.sqrt(
          (pellet_x_vals - pellet_home[0]) ** 2
        + (pellet_y_vals - pellet_home[1]) ** 2
        + (pellet_z_vals - pellet_home[2]) ** 2
    )

    triangle_xyz_p = df_3d['Triangle']
    # triangle_p = triangle_xyz_p['p']
    triangle_x_vals = triangle_xyz_p['x'].values
    triangle_y_vals = triangle_xyz_p['y'].values
    triangle_z_vals = triangle_xyz_p['z'].values
    star_xyz_p = df_3d['Star']
    # star_p = star_xyz_p['p']
    dist_st = np.sqrt(
          (star_xyz_p['x'].values - triangle_x_vals) ** 2
        + (star_xyz_p['y'].values - triangle_y_vals) ** 2
        + (star_xyz_p['z'].values - triangle_z_vals) ** 2
    )

    # dist_tpX = triangle_x_vals - pellet_x_vals
    # dist_tpY = triangle_y_vals - pellet_y_vals
    # dist_tpZ = triangle_z_vals - pellet_z_vals

    tongue_mid_xyz_p = df_3d['Tongue_mid']
    tongue_mid_p = tongue_mid_xyz_p['p']
    dist_tvpp = np.sqrt((tongue_mid_xyz_p['x'].values-pellet_home[0])**2+
                            (tongue_mid_xyz_p['y'].values-pellet_home[1])**2+
                            (tongue_mid_xyz_p['z'].values-pellet_home[2])**2)

    r_hand_xyz_p = df_3d['R_Hand']
    r_hand_p = r_hand_xyz_p['p']
    dist_hvpp_R = np.sqrt((r_hand_xyz_p['x'].values-pellet_home[0])**2+
                            (r_hand_xyz_p['y'].values-pellet_home[1])**2+
                            (r_hand_xyz_p['z'].values-pellet_home[2])**2)

    l_hand_xyz_p = df_3d['L_Hand']
    l_hand_p = l_hand_xyz_p['p']
    dist_hvpp_L = np.sqrt((l_hand_xyz_p['x'].values-pellet_home[0])**2+
                              (l_hand_xyz_p['y'].values-pellet_home[1])**2+
                              (l_hand_xyz_p['z'].values-pellet_home[2])**2)
    # velocity_h_L = np.diff(dist_hvpp_L)*(frame_rate/1000)
    # velocity_h_filt_L = filtfilt(coeffs, [1], velocity_h_L)
    # Z_dist_h_L = df_3d['L_Hand']['z'].values-pellet_home[2]

    Z_dist_p = pellet_z_vals - pellet_home[2]
    Z_dist_p[pellet_p == 0] = np.nan
    # Y_dist_p = np.abs(df_3d['Pellet']['y'].values-pellet_home[1])

    ############################
    #### Pellet-related variables
    ############################
    # Duration in seconds that a pellet must be near the origin to be considered 'placed'
    time2place = .05
    n_frames_2_place = time2place * frame_rate

    # Maximum distance (mm) a pellet can be from origin and still be 'placed'
    max_dist_from_orig = 2

    # Duration in seconds that a pellet must be away from the origin to be considered 'lost'
    time2lost = 0.1

    # Minimum duration in seconds between 'lost' and 'placed'
    min_inter_pellet_interval = 5

    # Minimum distance (mm) a hand must be from the pellet to call it 'grabbed' when 'lost'
    min_dist_for_grab = 15 # 8

    frames_on_found = []
    frames_on_lost = []
    frm_counter = -1
    frame_at_count_begin = 0
    count = 0
    pellet_state = 0 # 0 is lost, 1 is placed
    pellet_events = []

    #

    for dp, st, pp in zip(dist_p, dist_st, pellet_p):
        frm_counter += 1
        idx = frm_counter
        if debug >= 3:
            print(
                f"{frm_counter} > {dp=:.1f} {st=:.1f} {pp=:.1f} \n"  # {tpX=:.1f} {tpY=:.1f} {tpZ=:.1f}\n"
                f"    P=({pellet_x_vals[idx]:.1f}, {pellet_y_vals[idx]:.1f}, {pellet_z_vals[idx]:.1f})"
                f"    T=({triangle_x_vals[idx]:.1f}, {triangle_y_vals[idx]:.1f}, {triangle_z_vals[idx]:.1f})"
            )

        if pellet_state == 0: # Searching for placement
            testA = dp <= max_dist_from_orig
            testB = pp == 1 # is the pellet detected in frame
            testC = st > 12 or np.isnan(st) # was the cover open or not installed?

            if not (testA and testB and testC):  #  and testD:
                if count != 0:
                    if debug >= 1:
                        print(f"tests failed: {count}: {testA=} {testB=} {testC=}")
                    count = 0
                    frame_at_count_begin = frm_counter
                continue

            count += 1
            if count >= n_frames_2_place:
                pellet_dict = {
                    'placed': frame_at_count_begin,
                    'lost': None,
                    'method': ReachEventMethod.NONE,
                    'outcome': ReachEventOutcome.NONE,
                }
                pellet_events.append(pellet_dict)
                frames_on_found.append(frame_at_count_begin)
                pellet_state = 1
                count = 0

        elif pellet_state == 1: # Searching for pellet lost
            if dp > max_dist_from_orig or pp == 0:
                count += 1
            else:
                if debug >= 1:
                    print(f"state==1: {count=} ; {dp:.3f} > {max_dist_from_orig} ; {pp=}")
                count = 0
                frame_at_count_begin = frm_counter

            if count >= time2lost * frame_rate:
                frames_on_lost.append(frame_at_count_begin)
                pellet_dict['lost'] = frame_at_count_begin
                right_test = dist_hvpp_R[frame_at_count_begin] < min_dist_for_grab
                right_test = right_test and r_hand_p[frame_at_count_begin] == 1
                left_test = dist_hvpp_L[frame_at_count_begin] < min_dist_for_grab
                left_test = left_test and l_hand_p[frame_at_count_begin] == 1
                tongue_test = tongue_mid_p[frame_at_count_begin] == 1
                RVL_test = dist_hvpp_R[frame_at_count_begin] < dist_hvpp_L[frame_at_count_begin]
                TVR_test = dist_tvpp[frame_at_count_begin] < dist_hvpp_L[frame_at_count_begin]
                TVL_test = dist_tvpp[frame_at_count_begin] < dist_hvpp_R[frame_at_count_begin]
                pellet_dict['outcome'] = ReachEventOutcome.EATEN
                if TVR_test and TVL_test and tongue_test:
                    pellet_dict['method'] = ReachEventMethod.TONGUE
                elif RVL_test and right_test:
                    pellet_dict['method'] = ReachEventMethod.RIGHT_HAND
                elif not RVL_test and left_test:
                    pellet_dict['method'] = ReachEventMethod.LEFT_HAND
                else:
                    pellet_dict['method'] = ReachEventMethod.OTHER
                    pellet_dict['outcome'] = ReachEventOutcome.DROPPED
                if debug >= 1:
                    print(f"Right hand : {dist_hvpp_R[frame_at_count_begin]} at {frame_at_count_begin}")
                    print(f"Left hand : {dist_hvpp_L[frame_at_count_begin]}")
                    print(f"Tongue : {dist_tvpp[frame_at_count_begin]}")
                    print(f"R/L/T conf : {r_hand_p[frame_at_count_begin]}/{l_hand_p[frame_at_count_begin]}/{tongue_mid_p[frame_at_count_begin]}")
                pellet_events[-1] = pellet_dict
                pellet_state = 2

        elif pellet_state == 2: # Waiting minimum inter-pellet interval
            count += 1
            if count >= min_inter_pellet_interval * frame_rate:
                pellet_state = 0
                count = 0

    logger.verbose("segment_reaches: pellet_home=%s frames_on_found=%s pellet_events=%s",
                   pellet_home, frames_on_found, pellet_events)
    # end big for
    return dist_p, Z_dist_p, dist_hvpp_R, pellet_events, pellet_home, frames_on_found
