import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate

from lib.diag_common.numpy_helpers import gen_range


def save_to_excel(input: dict, file_name):
    '''
    Function for saving the outputs of the model to excel
    :param output:
    :return:
    '''
    # COMMENTS AND FUTURE WORK
    # use as columns name something that can be easily ordered like T_001, T_002, ..., T_099
    # include delta t , needing to deal with decimal seconds like 0.3, in this case needing to make the
    # model return also the time side by the outputs

    # output = output.numpy()
    # num_temperatures = output.shape[-1]
    #
    # name_columns = ['time'] + [f'T_{i}' for i in range(num_temperatures)]
    # time = np.arange(output.shape[0])
    # time = np.expand_dims(time, 1)
    # data_to_save = np.hstack((time, output))
    #
    # df = pd.DataFrame(data=data_to_save, columns=name_columns)
    # #df.drop_duplicates(keep=False, inplace=True)
    #
    # df.to_excel(excel_writer=file_name, index=False)
    time = input['time_series'].numpy()
    temperatures = input['temperatures'].numpy()
    name_columns = ['time'] + [f'T_{i}' for i in range(temperatures.shape[-1])]
    data_to_save = np.hstack((time, temperatures))
    df = pd.DataFrame(data=data_to_save, columns=name_columns)
    df.drop_duplicates(keep=False, inplace=True)
    df.to_excel(excel_writer=file_name, index=False)


def sampling_heat_data(file, starting_point, num_steps_predicted):
    df = pd.read_excel(file, index_col=0)

    initial_temperatures_set_up = df.iloc[starting_point]
    initial_temperatures_set_up = tf.convert_to_tensor(initial_temperatures_set_up, dtype=tf.float32)

    predictions = df.iloc[starting_point + 1: starting_point + num_steps_predicted + 1, :]
    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)

    return (initial_temperatures_set_up, predictions)


def sample_generator(*, ds_filepath: str, variables: list[str],
                     samples_per_frame: int, time_step: float,
                     starting_time: float = None, duration: float = None,
                     stride: float = None, max_frames: int = None,
                     ) -> list[dict]:
    '''

    Returns a collection of frames, where each frame is in the ofrm of dictionary.
    Each frame contains `samples_per_frame` 'rows' of data, so that it can be directly used for models, which simulates multiple steps at once and
    return a series of output values.

    :param variables:
    :param ds_filepath: file path from which extract the data points ( it is supposed to be an excel file, coming form Timber, where the columns
                are named starting from the "time" then followed by temperatures ordered in a sequential way
    :param starting_time: starting time point is seconds
    :param duration: duration of the period of interest in seconds
    :param stride: distance between two samples
    :param max_frames: number of samples to extract from the file
    :param samples_per_frame: number of samples, that should be included in each frame returned. This is related to number of steps of the model (model_num_steps + 0/1 depending if the model is also returning the first fix or not)
    :param time_step: "delta t" between each sample point
    :param include_initial_starting_point: True or False to include the initial starting point, this is because also the model
                                            has the option to include or not the initial starting point
    :return: TO BE DECIDED ( can be a dictionary for example)

    WARNINGS: THE RESULT CAN BE DIFFERENT FROM WHAT SPECIFIED:
             * IF THE USER WANTS TO START FROM A POINT WHICH COMES TEMPORALLY
               BEFORE THE FIRST RECORDED VALUE , THE FUNCTION AUTOMATICALLY STARTS FROM THE FIRST RECORDED VALUE, WITHOUT
               RAISING ANY ERROR
             * IF THE USER WANTS TO START FROM A POINT WHICH COMES TEMPORALLY
               AFTER THE LAST RECORDED VALUE, THE FUNCTION AUTOMATICALLY RETURNS AN EMPTY DICTIONARY,
               WITHOUT RAISING ANY ERROR
    '''

    TIME_COL_NAME = 'time'

    frame_duration = (samples_per_frame - 1) * time_step

    df = pd.read_excel(ds_filepath)

    min_time_in_ds = df[TIME_COL_NAME].iloc[0]
    max_time_in_ds = df[TIME_COL_NAME].iloc[-1]

    if starting_time is None:
        starting_time = min_time_in_ds

    if duration is None:
        duration = max_time_in_ds - starting_time

    if duration < frame_duration:
        raise ValueError(f"Requested DURATION for sampling is LESS than a frame-duration! "
                         f"Not a single frame can fit that duration! Requested duration: {duration}, frame-duration: {frame_duration} ")

    if starting_time >= max_time_in_ds:
        raise ValueError(
            f"Requested starting_time is BEYOND the max_time in DataSet! starting_time = {starting_time}, max_time_in_ds = {max_time_in_ds}")

    ending_time = starting_time + duration

    first_row_idx_of_interest = int(np.amax(df.index[df['time'] <= starting_time].to_list(), initial=0))
    last_row_idx_of_interest = int(np.amin(df.index[df['time'] > ending_time].to_list(), initial=df.index[-1]))

    df_to_extract = df.iloc[first_row_idx_of_interest:last_row_idx_of_interest + 1]

    if len(df_to_extract) <= 0:
        raise ValueError(
            f"(EE) sampler_generator() :: STARTING POINT GREATER THAN THE LAST RECORDED VALUE: starting_time={starting_time}, last time in DataSet: {df['time'][-1]} ")

    ds_min_time = df_to_extract[TIME_COL_NAME].iloc[0]
    ds_max_time = df_to_extract[TIME_COL_NAME].iloc[-1]

    # -- extracting, which columns form which variables:
    # col_names_for_variables = {}
    # columns_name = list(df_to_extract.columns)
    # all_used_col_names = set()
    # for var_name in variables:
    #     qualified_col_names = sorted([col_name for col_name in columns_name if col_name.startswith(var_name)])
    #     col_names_for_variables[var_name] = qualified_col_names
    #     all_used_col_names |= set(qualified_col_names)

    col_names_for_variables, all_used_col_names = extract_col_from_var(df=df_to_extract, variables=variables)

    # -- Initialization of interpolators:
    interpolators = {}
    for col_name in all_used_col_names:
        xs = df_to_extract[TIME_COL_NAME].values
        ys = df_to_extract[col_name].values
        interpolators[col_name] = interpolate.interp1d(xs, ys)

    if not stride:
        stride = frame_duration

    all_frames: list[dict] = []

    curr_start_time = max(starting_time, ds_min_time)

    while True:
        # -- now we try to extract frame: [curr_start_time, curr_end_time] -- IF it does NOT fit inside the interpolator range, quit!
        curr_range = gen_range(num=samples_per_frame, start=curr_start_time, step=time_step, dtype=np.float32,
                               shape=(-1, 1))
        curr_end_time = np.amax(curr_range)
        if curr_end_time > ds_max_time:
            break

        curr_frame = {
            'time_step': time_step,
            TIME_COL_NAME: curr_range
        }

        for var_name, col_names in col_names_for_variables.items():
            cols_ys = [interpolators[col_name](curr_range) for col_name in col_names]
            curr_frame[var_name] = np.hstack(cols_ys).astype(np.float32)
            # ys = np.reshape(interpolators['T[00]'](curr_range), newshape=(-1, 1))

        all_frames.append(curr_frame)
        curr_start_time += stride

        if (max_frames is not None) and (max_frames > 0) and (len(all_frames) >= max_frames):
            break

        # ^-- end of WHILE loop

    return all_frames


def load_row_from_dataset(file, starting_time_point, duration):
    '''
    The purpose of this function is to extract all the data points given from a given file ,
    from a starting time point given in "seconds" and given a proper time duration
    :param file: excel file
    :param starting_time_point: starting point for the data extraction, this value should be in "seconds"
    :param duration: duration in seconds of the period of interest
    :return: dataframe containing the sample points "to be fitted"
    '''

    df = pd.read_excel(file)
    ending_time_point = starting_time_point + duration
    df_to_extract = df.loc[(df['time'] >= starting_time_point) & (df['time'] <= ending_time_point)]

    return df_to_extract


def extract_col_from_var(df, variables: list[str]):
    """
    :param df: pandas DataFrame
    :param variables: list of string containing the variables we want to extract
    :return: dictionary of type variable: all df columns for that variable, set of df columns' name which form the variables
    """

    col_names_for_variables = {}
    columns_name = list(df.columns)
    all_used_col_names = set()
    for var_name in variables:
        qualified_col_names = sorted([col_name for col_name in columns_name if col_name.startswith(var_name)])
        col_names_for_variables[var_name] = qualified_col_names
        all_used_col_names |= set(qualified_col_names)

    return col_names_for_variables, all_used_col_names
