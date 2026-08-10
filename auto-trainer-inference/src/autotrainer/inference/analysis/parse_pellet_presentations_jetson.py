import inspect
import math
import os
import glob
import pickle
import time
from collections import OrderedDict
from typing import Tuple, Dict, Union, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, filtfilt

from autotrainer.core import get_verbose_logger, Offset3DTuple, ProjectInfo
from autotrainer.core.reach_event import (
    ReachEvent,
    ReachEventOutcome,
    ReachEventMethod,
)
from autotrainer.inference.analysis import prepare_jetson_data as prep_jet
from autotrainer.inference.analysis._segment_reaches_f1 import segment_reaches_f11

logger = get_verbose_logger(__name__)


def get_coeffs():
    # Savitzky-Golay Smoothing filter parameters
    window_length = 9
    poly_order = 3
    # Obtain Savitzky-Golay filter coefficients
    coeffs = savgol_coeffs(window_length, poly_order)

    return coeffs


def find_last_placement(frmq, pellet_events):
    for i, pe in enumerate(pellet_events[:-1]):
        if i == (len(pellet_events)-2):
            break
        elif pe['placed'] < frmq < pellet_events[i+1]['placed']:
            break
    return i


def get_ln():
    """Return the current line number."""
    return inspect.currentframe().f_back.f_lineno


def segment_reaches(
    *,
    project_info: ProjectInfo,
    session,
    center_method,
    df_lr,
    df_3d,
    frame_rate: int,
    overwrite: bool = True,
    debug: int = 0,
) -> Dict[str, Union[float, int, Dict, List]]:
    results_dict = {
        'pellets_presented': 0,
        'total_reaches': 0,
        'pellets_consumed': 0,
        'successful_reaches': 0,
        'rh_max_vp_list': [],
        'reach_events': [],
        'other_events': [],
    }
    if df_3d is None:
        return results_dict

    # Find video files
    mp4_list = os.path.join(session, f"*.{project_info.video_write_ext}")
    videoList = glob.glob(mp4_list)

    # Extract relevant video paths in order
    videoOrder = [
        project_info.camera_names[0],  # primary camera
        project_info.camera_names[1],  # secondary camera
    ]
    video_paths = [video for key in videoOrder for video in videoList if key in video]
    if not video_paths:
        logger.warning('No Videos found!')
        return results_dict
    vid_name_base, vid_dir = prep_jet.get_vid_name_base(video_paths[0])

    save_file_path = os.path.join(vid_dir, vid_name_base + '_eventSegmentation.pickle')
    if os.path.isfile(save_file_path) and not overwrite:
        logger.notice('Previous analysis found for %s', vid_name_base)
        # TODO: read the file and returns its results dicts..
        return results_dict

    coeffs = get_coeffs()

    dist_p, Z_dist_p, dist_hvpp_R, pellet_events, pellet_home, frames_on_found = segment_reaches_f1(
        project=project_info,
        df_3d=df_3d,
        frame_rate=frame_rate,
        coeffs=coeffs,
        center_method=center_method,
        debug=debug,
    )

    if debug >= 1:
        logger.verbose("segment_reaches_f1: events=%s", pellet_events)

    pellets_consumed, pellets_presented, successful_reaches, total_reaches, rh_max_vp_list, reach_events = segment_reaches_f2(
        project=project_info,
        fps=frame_rate,
        df_3d=df_3d,
        coeffs=coeffs,
        vid_dir=vid_dir,
        vid_name_base=vid_name_base,
        pellet_events=pellet_events,
        pellet_home=pellet_home,
        dist_p=dist_p,
        Z_dist_p=Z_dist_p,
        frame_rate=frame_rate,
        frames_on_found=frames_on_found,
        dist_hvpp_R=dist_hvpp_R,
        debug=debug,
    )

    with open(save_file_path, 'wb') as f:
        pickle.dump(reach_events, f)

    def save_reach_events_h5():
        p_before = time.perf_counter()
        # could be done
        reach_event_h5_path = project_info.get_reach_event_path()
        reach_df = pd.DataFrame(reach_events, columns=['init', 'end', 'max', 'outcome', 'delay_since_presented'])
        reach_df.to_hdf(reach_event_h5_path, key="reach", mode="w")  # use mode='w' for first write,
        # subsequent writes to it will be with mode='a'

        column_names = ("left", "right", "3d")
        index_labels = ("x", "y", "z")
        columns = pd.MultiIndex.from_product(
            [column_names, index_labels],
            names=["view", "coordinates"])

        pellet_dfs = []
        reaches_dfs = []
        for reach in reach_events:
            r_end = reach['end']
            r_init = reach['init']
            l = df_lr[0]['R_Hand'][['x', 'y']].assign(z=0).iloc[r_init:r_end].reset_index(drop=True)
            r = df_lr[1]['R_Hand'][['x', 'y']].assign(z=0).iloc[r_init:r_end].reset_index(drop=True)
            three_d = df_3d['R_Hand'][['x', 'y', 'z']].iloc[r_init:r_end].reset_index(drop=True)
            reaches_dfs.append(pd.DataFrame(
                {
                    (w, c): o[c]
                    for c in 'xyz'
                    for w, o in (("left", l), ("right", r), ("3d", three_d))
                },
                columns=columns,
            ))
            l = df_lr[0]['Pellet'][['x', 'y']].assign(z=0).iloc[r_init:r_end].reset_index(drop=True)
            r = df_lr[1]['Pellet'][['x', 'y']].assign(z=0).iloc[r_init:r_end].reset_index(drop=True)
            three_d = df_3d['Pellet'][['x', 'y', 'z']].iloc[r_init:r_end].reset_index(drop=True)
            pellet_dfs.append(pd.DataFrame(
                {
                    (w, c): [o[c].loc[:30].mean()]
                    for c in 'xyz'
                    for w, o in (("left", l), ("right", r), ("3d", three_d))
                },
                columns=columns,
            ))

        if len(pellet_dfs) == 0:
            pellet_dfs = pd.DataFrame(columns=columns)
        else:
            pellet_dfs = pd.concat(pellet_dfs, ignore_index=True)
        pellet_dfs.to_hdf(reach_event_h5_path, key="pellet", mode="a")

        if len(reaches_dfs) == 0:
            trajectory_dfs = pd.DataFrame(columns=columns)
        else:
            trajectory_dfs = pd.concat(OrderedDict((k, trj) for k, trj in enumerate(reaches_dfs)))
        trajectory_dfs.to_hdf(reach_event_h5_path, key="trajectory", mode="a")
        logger.verbose("saved in %.1fs reach_events df to %s",
                       time.perf_counter() - p_before,
                       reach_event_h5_path)

    save_reach_events_h5()

    results_dict['pellets_presented'] = pellets_presented
    results_dict['total_reaches'] = total_reaches
    results_dict['pellets_consumed'] = pellets_consumed
    results_dict['successful_reaches'] = successful_reaches
    results_dict['rh_max_vp_list'] = rh_max_vp_list
    results_dict['reach_events'] = reach_events
    other_events = results_dict['other_events'] = []
    for pellet_event in pellet_events[:-1]:
        if pellet_event['method'] not in {ReachEventMethod.RIGHT_HAND, ReachEventMethod.LEFT_HAND}:
            other_events.append(pellet_event)
    return results_dict


def segment_reaches_f1(
    *,
    project: ProjectInfo,
    df_3d: pd.DataFrame,
    frame_rate: int,
    coeffs: np.ndarray,
    center_method: Tuple[int, str],
    debug: int,
):
    # Calculate distance and speed
    # bodyparts = df_3d.columns.get_level_values('bodyparts').unique()
    bp4speed = ['R_Hand','L_Hand','Pellet']
    for bp in bp4speed:
        df_bp = df_3d[bp]
        dist_vec = np.sqrt(
              np.diff(df_bp['x']) ** 2
            + np.diff(df_bp['y']) ** 2
            + np.diff(df_bp['z']) ** 2
        )  # calculate distance
        dist_vec = np.concatenate(([dist_vec[0]], dist_vec))  # adjust size
        speed_vec = dist_vec * (frame_rate / 1000)  # convert to speed in mm/ms

        # Apply filter using filtfilt for smoothing
        speed_vec_filt = filtfilt(coeffs, [1], speed_vec)
        # speed_vec_filt[df_3d[bp]['p'] == 0] = np.nan
        df_3d.loc[:, (bp, 'speed')] = speed_vec_filt

    df_3d_pellet = df_3d["Pellet"]
    t_delivered = project.get_t_pellet_delivered_or_default()
    t_presented = project.get_t_pellet_presented_or_default()
    f_delivered = int(t_delivered * frame_rate)
    f_presented = int(t_presented * frame_rate)

    # todo; using up to 25 frames arbitrarily, we might want be more precise
    r = df_3d_pellet.iloc[f_delivered:f_delivered + 25].loc[df_3d_pellet['p'] == 1]
    pellet_home = tuple(r[pos].median() for pos in "xyz")

    logger.verbose("segment_reaches: using pellet_home=%s ; f_deliv=%s f_pres=%s t_deliv=%.3f t_pres=%.3f",
                   pellet_home, f_delivered, f_presented, t_delivered, t_presented)

    return segment_reaches_f11(
        df_3d=df_3d,
        frame_rate=frame_rate,
        pellet_home=pellet_home,
        debug=debug,
    )


def segment_reaches_f2(
    *,
    project: ProjectInfo,
    fps: int,
    df_3d: pd.DataFrame,
    coeffs,
    vid_dir,
    vid_name_base,
    pellet_events,
    pellet_home,
    dist_p,
    Z_dist_p,
    frame_rate,
    frames_on_found,
    dist_hvpp_R,
    debug,
) -> Tuple[int, int, int, int, List[Offset3DTuple], List[Dict]]:
    frm_ct = np.shape(df_3d)[0]
    ############################
    #### Reach-related variables
    ############################
    # Maximum reach duration (seconds)
    max_reach_dur = 1

    # Minimum reach duration (seconds)
    min_reach_dur = 0.1

    # The hand must come within this distance to the pellet (mm)
    min_dist_from_pellet = 15

    batch_frm = 10
    batch_dist = 5

    # can cause some (quickly succeeding) reaches count to be missed though
    position_window = 100
    speed_window = 100

    batch_speed = 5
    batch_stall = 25
    reach_init_speed = -0.025
    reach_dirchange_speed = 0.025
    pellet_drop_speed = 0.275  # 0.225
    pellet_drop_dist = -5
    dist_thresh_end = 5
    confidence = 0.9

    if len(pellet_events) == 0:
        logger.warning("Pellet never found for %s", vid_name_base)

    for pellet_event in pellet_events:
        lost = pellet_event['lost']
        if lost is not None and lost >= 0:
            end_search = lost + position_window
            if end_search >= frm_ct:
                end_search = frm_ct - 1
            # print(np.round((Z_dist_p[pellet_event['placed']:end_search])))
            # print(pellet_event['placed'],end_search)
            if np.nanmin(Z_dist_p[pellet_event['placed']:end_search]) <= pellet_drop_dist:
                if debug >= 2:
                    logger.debug('DROP: Z dist at %s', get_ln())
                pellet_event['outcome'] = ReachEventOutcome.DROPPED

    pellet_dict = {
        'x': pellet_home[0],
        'y': pellet_home[1],
        'z': pellet_home[2],
    }
    pellet_events.append(pellet_dict)

    # ########
    # with open(pellet_file_path, 'wb') as f:
    #     pickle.dump(pellet_events, f)
    # print(pellet_events)
    # return
    # ########

    pellet_data = df_3d['Pellet']
    pellet_p = pellet_data['p']
    pellet_speed = pellet_data['speed']

    r_hand_data = df_3d['R_Hand']
    r_hand_p = r_hand_data['p']
    r_hand_speed = r_hand_data['speed']

    velocity_h_R = np.diff(dist_hvpp_R)*(frame_rate/1000)
    velocity_h_filt_R = filtfilt(coeffs, [1], velocity_h_R)
    Z_dist_h_R = r_hand_data['z'].values - pellet_home[2]

    reach_events = []
    dist_list = []
    max_frm = 0

    for frmindex, start_frame in enumerate(frames_on_found):

        # NB: pellet_events is same size/length than frames_on_found, actually both could be merged.
        # also use directly pellet_event instead of find_last_placement(..) below.
        pellet_event = pellet_events[frmindex]

        search_status = 1
        food_was_dropped = False
        frame = start_frame - 20 # in case reach begins prior to pellet placement
        pellet_detected = False
        while frame < frm_ct - batch_frm:             # test if we reached the next pellet placement
            if frmindex < len(frames_on_found)-1 and frame >= frames_on_found[frmindex+1]:
                if search_status == 1:
                    if debug >= 2:
                        if not pellet_detected:
                            logger.debug('No pellet detected')
                        else:
                            logger.debug('No (additional) reach detected')
                break
            testA = np.sum(pellet_p[frame:frame+batch_frm] > confidence)/batch_frm > 0.75 # test if pellet is there
            if testA:
                pellet_detected = True
            else:
                pellet_detected = False

            if search_status == 1:                  # search for a reach initiation
                if pellet_detected:
                    testD = np.sum(r_hand_p[frame:frame+batch_frm] > confidence)/batch_frm > 0.75 # test if hand is there
                    if testD:
                        testA = Z_dist_h_R[frame] > -4
                        testB = np.mean(velocity_h_filt_R[frame:frame+batch_speed]) < reach_init_speed

                        if testA and testB:
                            if debug >= 2:
                                logger.debug('reach began at frame %d!', frame)
                            delay_since_presented = frame / fps - project.t_pellet_presented
                            reach_dict = {
                                'init': frame,
                                'max': None,
                                'end': None,
                                'method': pellet_event['method'],
                                'outcome': pellet_event['outcome'],
                                'delay_since_presented': delay_since_presented,
                            }
                            search_status = 2
                            food_was_dropped = False
                            speed_hvh_init = np.diff(dist_hvpp_R - dist_hvpp_R[frame])*(frame_rate/1000)
                            speed_hvh_init = filtfilt(coeffs, [1], speed_hvh_init)

            elif search_status == 2:                #search for reach max
                speed_seg = np.asarray(pellet_speed[frame:frame + speed_window])
                if np.sum(speed_seg > pellet_drop_speed) > 1:
                    food_was_dropped = True
                if np.any(Z_dist_p[frame:frame+position_window] < pellet_drop_dist): #food dropped if pellet is too low -
                    z_dist_indices = np.where(Z_dist_p[frame:frame+position_window] < pellet_drop_dist)
                    testD = np.any(pellet_p[frame:frame+position_window].iloc[z_dist_indices] > confidence)
                    if testD:
                        food_was_dropped = True
                # print(np.mean(speed_hvh_init[frame:frame+batch_speed]))
                testB = np.mean(speed_hvh_init[frame:frame+batch_speed]) > reach_dirchange_speed
                if testB:
                    if debug >= 2:
                        logger.debug('reach max at frame %d!', int(frame+3))
                    # reach_events.append(('reachMax', int(frame+3)))
                    reach_dict['max'] = int(frame+3)
                    max_frm = frame+3

                    search_status = 3
                    frame += 3

            elif search_status == 3:                # search for reach end
                keep_looking = True
                # print(np.mean(speed_hvh_init[frame:frame+batch_speed]))
                testA = np.mean(dist_hvpp_R[frame:frame + batch_dist]) > dist_thresh_end  # test if hand is far enough away and moving away from pellet
                testB = np.mean(speed_hvh_init[frame:frame+batch_speed]) < reach_dirchange_speed
                testC = np.mean(velocity_h_filt_R[frame:frame+batch_speed]) < reach_init_speed
                testD = np.allclose(np.mean(dist_hvpp_R[frame:frame+batch_stall]), dist_hvpp_R[frame], atol = 2)
                testE = np.isclose(np.mean(r_hand_speed[frame:frame + batch_stall]), 0, atol = 0.025)
                testF = np.all(dist_hvpp_R[frame:frame + batch_stall] < 6) #4
                speed_seg = np.asarray(pellet_speed[frame:frame + speed_window])
                if np.sum(speed_seg > pellet_drop_speed) > 1:
                    # print(np.sum(speed_seg > pellet_drop_speed))
                    food_was_dropped = True
                    if debug >= 2:
                        logger.debug('DROP: speed drop - line %s - frame %s', get_ln(), frame)

                if np.any(Z_dist_p[frame:frame+position_window] < pellet_drop_dist): #food dropped if pellet is too low
                    z_dist_indices = np.where(Z_dist_p[frame:frame+position_window] < pellet_drop_dist)
                    testD = np.any(pellet_p[frame:frame+position_window].iloc[z_dist_indices] > confidence)
                    if testD:
                        food_was_dropped = True
                        if debug >= 1:
                            logger.verbose('DROP: pellet too low - line %s - frame %s', get_ln(), frame)

                if testA and testC and not food_was_dropped: #and pellet_detected:
                    if debug >= 2:
                        logger.debug('reach ended at frame %d!: NEW REACH', int(frame-1))
                    reach_dict['end'] = int(frame-1)
                    reach_dict['outcome'] = ReachEventOutcome.MISSED
                    # reach_events.append(('reachEnd_missed', int(frame-1)))
                    search_status = 1
                    dist_list.append(dist_hvpp_R[max_frm])

                elif testD and testE and testF: # and not food_was_dropped:
                    if debug >= 2:
                        logger.debug('reach stalled')
                    # reach_events.append(('reachEnd_stalled', int(frame+10)))
                    reach_dict['end'] = int(frame+10)
                    reach_dict['outcome'] = ReachEventOutcome.STALLED
                    search_status = 1
                    # keep_looking = True
             
                elif testA and testB:
                    pTest = np.mean(dist_p[frame:frame+batch_frm]) < 2 # pellet wasnt dropped and still in original position 
                    if food_was_dropped:
                        if debug >= 2:
                            logger.debug('reach ended at frame %d!: DROPPED', int(frame+2))
                        # reach_events.append(('reachEnd_dropped', int(frame+2)))
                        reach_dict['end'] = int(frame+2)
                        reach_dict['outcome'] = ReachEventOutcome.DROPPED
                        # lp = find_last_placement(frame+2, pellet_events)
                        # pellet_events[lp]['outcome'] = 'dropped'
                        pellet_event['outcome'] = ReachEventOutcome.DROPPED
                        keep_looking = False
                    elif pellet_detected and pTest:
                        if debug >= 2:
                            logger.debug('reach ended at frame %d!: MISSED', int(frame+2))
                        # reach_events.append(('reachEnd_missed', int(frame+2)))
                        reach_dict['end'] = int(frame+2)
                        reach_dict['outcome'] = ReachEventOutcome.MISSED
                        dist_list.append(dist_hvpp_R[max_frm])
                        frame += 2
                        search_status = 1  
                    else:
                        if debug >= 2:
                            logger.debug('reach ended at frame %d!: GRABBED', int(frame+2)) #alt: pellet position close to hand(within some threshold)
                        # reach_events.append(('reachEnd_grabbed', int(frame+2)))
                        reach_dict['end'] = int(frame+2)
                        reach_dict['outcome'] = ReachEventOutcome.GRABBED
                        # lp = find_last_placement(frame+2, pellet_events)
                        # pellet_events[lp]['outcome'] = 'eaten'
                        keep_looking = False 
                 
                if reach_dict['end'] is not None:
                    start_query = reach_dict['init']
                    max_query = reach_dict['max']
                    end_query = reach_dict['end']
                    if start_query is not None and end_query is not None:
                        testX = end_query - start_query < min_reach_dur * frame_rate
                        testY = end_query - start_query > max_reach_dur * frame_rate
                    else:
                        testX = True
                        testY = True
                    if end_query is not None:
                        testZ = dist_hvpp_R[max_query] > min_dist_from_pellet
                    else:
                        testZ = False
                    if not (testX or testY or testZ):
                        reach_events.append(reach_dict)

                if not keep_looking:
                    break
            frame += 1

    pellet_file_path = os.path.join(vid_dir, vid_name_base + '_pelletHistory.pickle')
    with open(pellet_file_path, 'wb') as f:
        pickle.dump(pellet_events, f)
    
    if debug >= 1:
        logger.verbose("pellet_events=%s", pellet_events)

    pellets_presented = len(pellet_events) - 1
    total_reaches = 0
    successful_reaches = 0
    pellets_consumed = 0

    if len(pellet_events) > 0:
        total_reaches += len(reach_events)

        for p in pellet_events[:-1]:
            if p['outcome'] == ReachEventOutcome.EATEN:
                pellets_consumed += 1
                if p['method'] == ReachEventMethod.RIGHT_HAND:
                    successful_reaches += 1

    fail_ct = 0
    rh_max_vp_list = []

    # loop over all reach events, and check for/get what we need:
    for r in reach_events:
        if r['outcome'] in {ReachEventOutcome.MISSED, ReachEventOutcome.DROPPED}:
            fail_ct += 1
            r_max = r['max']
            if r_max is None or r_max < 0:
                o = None
            else:
                o = Offset3DTuple(
                    r_hand_data['x'][r_max] - pellet_home[0],
                    r_hand_data['y'][r_max] - pellet_home[1],
                    r_hand_data['z'][r_max] - pellet_home[2],
                )
            rh_max_vp_list.append(o)

    if debug >= 1:
        logger.verbose("reach_events=%s", reach_events)

    return pellets_consumed, pellets_presented, successful_reaches, total_reaches, rh_max_vp_list, reach_events
