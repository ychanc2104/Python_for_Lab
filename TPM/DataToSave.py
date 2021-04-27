
### import used modules first
import math
import random
import string
import numpy as np
import os
import datetime
import pandas as pd

### Use for data saving and data reshaping
class DataToSave:
    # data: np.array, path_folder: string path
    def __init__(self, data, localization_results, path_folder, frame_start, med_fps, window, factor_p2n, BM_lower=10, BM_upper=200, random_string=''):
        self.med_fps = med_fps
        self.BM_lower = BM_lower
        self.BM_upper = BM_upper
        self.columns = self.__get_df_sheet_names()
        self.localization_results = localization_results
        self.df = pd.DataFrame(data=data, columns=self.columns)
        self.path_folder = path_folder
        self.sheet_names = self.__get_analyzed_sheet_names() + self.__get_reshape_sheet_names()
        self.filename_time = self.__get_date()
        self.bead_number = int(max(1 + self.df['aoi']))
        self.frame_acquired = int(len(self.df['x']) / self.bead_number)
        self.frame_start = frame_start # starting frame for statistics
        self.frame_n = self.frame_acquired # frame number for statistics
        self.time = self.__get_time()[frame_start:frame_start+self.frame_n]
        self.time_sliding, self.time_fixing = self.__get_sftime(window=window)
        self.df_reshape = self.__get_reshape_data(self.df, med_fps)
        self.x_2D = np.array(self.df_reshape['x'])
        self.y_2D = np.array(self.df_reshape['y'])
        self.sx_2D = np.array(self.df_reshape['sx'])
        self.sy_2D = np.array(self.df_reshape['sy'])
        self.df_reshape_analyzed = self.get_analyzed_data(self.df_reshape, window, med_fps, factor_p2n)
        self.random_string = random_string
        self.random_string = self.__gen_random_code(3)
    ##  save four files
    def Save_four_files(self):
        # random_string = self.__gen_random_code(3)
        self.save_fitresults_to_csv()
        self.save_all_dict_df_to_excel()
        self.save_selected_dict_df_to_excel()
        self.save_removed_dict_df_to_excel()

    ##  save fitresults to csv
    def save_fitresults_to_csv(self):
        random_string = self.random_string
        df = self.df
        path_folder = self.path_folder
        filename_time = self.filename_time
        df.to_csv(os.path.join(path_folder, f'{filename_time}-{random_string}-fitresults.csv'), index=False)

    ##  save all dictionary of DataFrame to excel sheets
    def save_all_dict_df_to_excel(self, title=''):
        random_string = self.random_string
        df_reshape_analyzed = self.df_reshape_analyzed
        path_folder = self.path_folder
        filename_time = self.filename_time
        filename = title + '-fitresults_reshape_analyzed.xlsx'
        sheet_names = self.sheet_names

        writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'))
        for sheet_name in sheet_names:
            df_save = df_reshape_analyzed[sheet_name]
            df_save.to_excel(writer, sheet_name=sheet_name, index=True)
        writer.save()

    ##  save selected dictionary of DataFrame to excel sheets
    def save_selected_dict_df_to_excel(self):
        random_string = self.random_string
        df_reshape_analyzed = self.df_reshape_analyzed
        path_folder = self.path_folder
        filename_time = self.filename_time
        filename = 'fitresults_reshape_analyzed_selected.xlsx'
        criteria = self.get_criteria(df_reshape_analyzed)
        sheet_names = self.sheet_names

        writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'))
        for sheet_name in sheet_names:
            df_save = df_reshape_analyzed[sheet_name]
            if sheet_name != 'med_attrs' and sheet_name != 'avg_attrs' and sheet_name != 'std_attrs':
                df_save_selected = df_save.T[criteria].T
            else:  # for avg_attrs and std_attrs sheets
                df_save_selected = df_save[criteria]
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=True)
        writer.save()

    ##  save removed dictionary of DataFrame to excel sheets
    def save_removed_dict_df_to_excel(self):
        random_string = self.random_string
        df_reshape_analyzed = self.df_reshape_analyzed
        path_folder = self.path_folder
        filename_time = self.filename_time
        filename = 'fitresults_reshape_analyzed_removed.xlsx'
        criteria = self.get_criteria(df_reshape_analyzed)
        sheet_names = self.sheet_names

        writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-{random_string}-{filename}'))
        for sheet_name in sheet_names:
            df_save = df_reshape_analyzed[sheet_name]
            if sheet_name != 'med_attrs' and sheet_name != 'avg_attrs' and sheet_name != 'std_attrs':
                df_save_selected = df_save.T[~criteria].T
            else:  # for avg_attrs and std_attrs sheets
                df_save_selected = df_save[~criteria]
            df_save_selected.to_excel(writer, sheet_name=sheet_name, index=True)
        writer.save()

    ##  get selection criteria
    def get_criteria(self, df_reshape_analyzed):
        BM_lower = self.BM_lower
        BM_upper = self.BM_upper
        ratio = df_reshape_analyzed['med_attrs'][['xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared']]
        BMx_med = df_reshape_analyzed['med_attrs']['BMx_sliding']
        ratio = np.nan_to_num(ratio)
        BMx_med = np.array(BMx_med).reshape((BMx_med.shape[0], 1))
        c_ratio = ((ratio > 0.80) & (ratio < 1.2))
        c_BM = (BMx_med > BM_lower) & (BMx_med < BM_upper)
        c = np.append(c_ratio, c_BM, axis=1)
        criteria = []
        for row_boolean in c:
            criteria += [all(row_boolean)]

        return np.array(criteria)

    ## get anaylyzed data, BM, sxsy,xy ratio...
    def get_analyzed_data(self, df_reshape, window, med_fps, factor_p2n):
        bead_number = self.bead_number
        analyzed_data = self.append_analyed_data(factor_p2n, med_fps, window)
        analyzed_sheet_names = self.__get_analyzed_sheet_names()
        df_reshape_analyzed = df_reshape.copy()
        # save data to dictionary of DataFrame
        for data, sheet_name in zip(analyzed_data, analyzed_sheet_names):
            if sheet_name == 'avg_attrs':
                df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data,
                                                               columns=self.__get_attrs_col(name='avg_attrs')).set_index(self.__get_columns('bead', data.shape[0])[1:])
            elif sheet_name == 'std_attrs' or sheet_name == 'med_attrs':
                df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data, 
                                                               columns=self.__get_attrs_col()).set_index(self.__get_columns('bead', data.shape[0])[1:])
            else:
                df_reshape_analyzed[sheet_name] = pd.DataFrame(data=data,
                                                               columns=self.__get_columns(sheet_name, bead_number)).set_index('time')
        return df_reshape_analyzed


    ##  append BM, sx_sy, xy_ratio, mean, std to analyzed_data
    def append_analyed_data(self, factor_p2n, med_fps, window):
        x_2D = self.x_2D
        y_2D = self.y_2D
        sx_2D = self.sx_2D
        sy_2D = self.sy_2D
        frame_acquired = self.frame_acquired
        med_fps = self.med_fps

        BMx_sliding, BMx_fixing = self.calBM_2D(x_2D, factor_p2n=factor_p2n, window=window)
        BMy_sliding, BMy_fixing = self.calBM_2D(y_2D, factor_p2n=factor_p2n, window=window)
        sx_sy = sx_2D * sy_2D
        xy_ratio = self.__get_xy_ratio([BMx_sliding, BMy_sliding], [BMx_fixing, BMy_fixing], [sx_2D ** 2, sy_2D ** 2])

        data_analyzed_med, data_analyzed_avg, data_analyzed_std = self.__med_avg_std_operator(BMx_sliding, BMx_fixing,
                                                                                        BMy_sliding, BMy_fixing, sx_sy,
                                                                                        xy_ratio[0][:int(2*med_fps)], xy_ratio[1][:int(2*med_fps/20)],
                                                                                        xy_ratio[2][:int(2*med_fps)])
        data_reshaped_med, data_reshaped_avg, data_reshaped_std = self.__df_reshape_med_avg_std_operator(self.df_reshape)
        # append data or time together
        data_reshaped_avg = np.append(data_reshaped_avg, self.localization_results, axis=1)
        data_med_2D = np.append(data_analyzed_med, data_reshaped_med, axis=1)
        data_avg_2D = np.append(data_analyzed_avg, data_reshaped_avg, axis=1)
        data_std_2D = np.append(data_analyzed_std, data_reshaped_std, axis=1)

        analyzed_data = [BMx_sliding, BMy_sliding, BMx_fixing, BMy_fixing, sx_sy, xy_ratio[0], xy_ratio[1], xy_ratio[2]]
        analyzed_data = self.__append_time(analyzed_data, med_fps, frame_acquired, window=20)
        analyzed_data = analyzed_data + [data_med_2D, data_avg_2D, data_std_2D]
        return analyzed_data

    ##  get time axis for each frame
    def __get_time(self):
        med_fps = self.med_fps
        path_folder = self.path_folder

        path_header_time = os.path.join(path_folder, 'header-time.txt')
        exist = os.path.exists(path_header_time)
        if exist == True:
            df_time = pd.read_csv(path_header_time, sep='\t')
            time = np.array(df_time)
        else:
            time = np.array([i / med_fps for i in range(self.frame_acquired)])
        return time

    ##  get time-axis for sliding and fixing window
    def __get_sftime(self, window=20):
        time = self.time
        frame_acquired = self.frame_acquired
        time_sliding = []
        iteration_sliding = frame_acquired - window + 1
        for i in range(iteration_sliding):
            time_pre = time[i: i + window]
            time_sliding += [np.mean(time_pre)]
        time_fixing = []
        iteration_fixing = int(frame_acquired / window)  # fix window
        for i in range(iteration_fixing):
            time_pre = time[i * window: (i + 1) * window]
            time_fixing += [np.mean(time_pre)]
        self.time_sliding = np.array(time_sliding)
        self.time_fixing = np.array(time_fixing)
        return np.array(time_sliding), np.array(time_fixing)


    ### data:1D numpy array for a bead, BM: 1D numpy array
    def __calBM_1D(self, data, window=20, factor_p2n=10000/180, method='sliding'):
        time = self.time
        if method == 'sliding':  # overlapping
            iteration = len(data) - window + 1  # silding window
            BM_s = []
            for i in range(iteration):
                data_pre = data[i: i + window]
                try:
                    BM_s += [factor_p2n * np.std(data_pre[data_pre > 0], ddof=1)]
                except:
                    BM_s += [0]
            BM = BM_s
        else:  # fix, non-overlapping
            iteration = int(len(data) / window)  # fix window
            BM_f = []
            for i in range(iteration):
                data_pre = data[i * window: (i + 1) * window]
                try:
                    BM_f += [factor_p2n * np.std(data_pre[data_pre > 0], ddof=1)]
                except:
                    BM_f += [0]
            BM = BM_f
        return np.array(BM)

    ##  cal BM of multiple beads, data_2D: (row, col)=(frames, beads)
    def calBM_2D(self, data_2D, window=20, factor_p2n=10000/180):
        ##  get BM of each beads
        BM_sliding = []
        BM_fixing = []
        for data_1D in data_2D.T:
            BM_sliding += [self.__calBM_1D(data_1D, window=window, factor_p2n=factor_p2n, method='sliding')]
            BM_fixing += [self.__calBM_1D(data_1D, window=window, factor_p2n=factor_p2n, method='fixing')]
        BM_sliding = np.array(BM_sliding).T
        BM_fixing = np.array(BM_fixing).T
        return BM_sliding, BM_fixing

    ##  cal ratio fo a len=2 list ratio
    def __get_xy_ratio(self, *args):
        xy_ratio = []
        for data in args:
            ratio = data[0] / data[1]
            ratio[ratio > 99999] = 0
            xy_ratio += [ratio.astype('float32')]
        return xy_ratio

    ##  data average operator for multiple columns(2D-array), output: (r,c)=(beads,attrs)
    def __med_avg_std_operator(self, *args):
        data_med_2D = []
        data_avg_2D = []
        data_std_2D = []
        for data_2D in args:
            data_med = []
            data_avg = []
            data_std = []
            for data in data_2D.T: # input row vector in each iteration
                data_med += [np.median(data, axis=0)]
                data_avg += [np.mean(data, axis=0)]
                data_std += [np.std(data, axis=0, ddof=1)]
            data_med_2D += [np.array(data_med)]
            data_avg_2D += [np.array(data_avg)]
            data_std_2D += [np.array(data_std)]
        return np.nan_to_num(data_med_2D).T, np.nan_to_num(data_avg_2D).T, np.nan_to_num(data_std_2D).T

    ##  get avg and std for reshaped DataFrame
    def __df_reshape_med_avg_std_operator(self, df_reshape):
        data_med = []
        data_avg = []
        data_std = []
        for i, sheet_name in enumerate(self.columns):
            if i > 1:
                data = np.array(df_reshape[sheet_name])
                data_med += [np.median(data, axis=0)]
                data_avg += [np.mean(data, axis=0)]
                data_std += [np.std(data, axis=0, ddof=1)]
        return np.array(data_med).T, np.array(data_avg).T, np.array(data_std).T

    ## get reshape data all
    def __get_reshape_data(self, df, med_fps):
        frame_acquired = self.frame_acquired
        df_reshape = dict()
        for i, sheet_name in enumerate(df.columns):
            if i > 1:
                df_reshape[sheet_name] = self.__gather_reshape_sheets(df, sheet_name, frame_acquired, med_fps)
        return df_reshape

    ##  save each attributes to each sheets, data:2D array
    def __gather_reshape_sheets(self, df, sheet_name, frame_acquired, med_fps):
        bead_number = self.bead_number
        name = self.__get_columns(sheet_name, bead_number)
        data = self.__get_attrs(df[sheet_name], bead_number, frame_acquired)
        data = np.array(self.__append_time([data], med_fps, frame_acquired))
        data = np.reshape(data, (frame_acquired, bead_number + 1))
        df_reshape = pd.DataFrame(data=data, columns=name).set_index('time')
        return df_reshape

    ##  add time axis into first column, data: list of 2D array,(r,c)=(frame,bead)
    def __append_time(self, analyzed_data, med_fps, frames_acquired, window=20):
        time_sliding = self.time_sliding
        time_fixing = self.time_fixing
        dt = (window-1)/med_fps/2
        analyzed_append_data = []
        for data in analyzed_data:
            if len(data) == len(time_sliding):
                time = time_sliding
            elif len(data) == len(time_fixing):
                time = time_fixing
            else:
                time = dt + np.arange(0, data.shape[0]) / med_fps * math.floor(frames_acquired / data.shape[0])
            time = np.reshape(time, (-1, 1))
            analyzed_append_data += [np.append(time, data, axis=1)]
        return analyzed_append_data

    ##  get columns for avg_attrs, med_attrs or std_attrs
    def __get_attrs_col(self, name='med_attrs'):
        analyzed_col = self.__get_analyzed_sheet_names()[:-3]
        reshape_col = self.__get_reshape_sheet_names()
        if name == 'med_attrs' or name == 'std_attrs':
            return analyzed_col + reshape_col
        else:
            return analyzed_col + reshape_col + ['bead_radius']

    ### input 1D array data, output: (row, column) = (frame, bead)
    def __get_attrs(self, data_col, bead_number, frame_acquired):
        data_col = np.array(data_col)
        data_col_reshape = np.reshape(data_col, (frame_acquired, bead_number))
        return data_col_reshape

    ### get name and bead number to be saved, 1st col is time
    def __get_columns(self, name, bead_number):
        columns = ['time'] + [f'{name}_{i}' for i in range(bead_number)]
        return np.array(columns)

    ### getting date
    def __get_date(self):
        filename_time = datetime.datetime.today().strftime('%Y-%m-%d')  # yy-mm-dd
        return filename_time

    ### get analyzed sheet names, add median
    def __get_analyzed_sheet_names(self):
        return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
                'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
                'med_attrs', 'avg_attrs', 'std_attrs']

    ### get reshape sheet names
    def __get_reshape_sheet_names(self):
        return ['amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity', 'intensity_integral', 'ss_res']

    ##  get df sheet names(tracking_results)
    def __get_df_sheet_names(self):
        return ['frame', 'aoi', 'amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity',
                'intensity_integral', 'ss_res']

    ##  add 2n-word random texts(n-word number and n-word letter)
    def __gen_random_code(self, n):
        random_string = self.random_string
        if len(random_string) == 0:
            digits = "".join([random.choice(string.digits) for i in range(n)])
            chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
            return digits + chars
        else:
            return random_string