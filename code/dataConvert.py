import pandas as pd
import os


class DataConvert:
    def __init__(self, data_path, output_path, province_name="1"):
        self.data_path = data_path
        self.output_path = output_path
        self.province_name = province_name

    def readData(self):
        data = pd.read_csv(self.data_path)
        return data

    def ConvertData(self):
        # 检查输出文件夹是否存在
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        df = self.readData()
        target = df["v1"].iloc[0]
        first_col_name = "Station_Code"
        # 1. 筛选臭氧数据（假设 v1 列是污染物类型，O3 表示臭氧）
        df_target = df[df["v1"] == target].copy()

        # 2. 找出所有日期范围（全局 min_date 和 max_date）
        all_dates = pd.to_datetime(df_target["Date"]).sort_values()
        min_date, max_date = all_dates.min(), all_dates.max()
        date_range = pd.date_range(min_date, max_date, freq="D")
        date_columns = [d.strftime("%Y-%m-%d") for d in date_range]

        # 3. 按站点分组，并检查数据长度是否匹配最长站点
        max_length = len(date_range)
        result = []

        for station_code, group in df_target.groupby(first_col_name):
            # 检查该站点数据是否覆盖全部日期
            if len(group) == max_length:
                # 按日期排序并提取臭氧浓度
                group_sorted = group.sort_values("Date")
                o3_values = (
                    group_sorted.set_index("Date")[target].reindex(date_columns).values
                )
                result.append([station_code] + o3_values.tolist())

        # 4. 创建最终 DataFrame
        final_df = pd.DataFrame(result, columns=[first_col_name] + date_columns)
        final_df.to_csv(
            os.path.join(self.output_path, "province_station_" + target + ".csv"),
            index=False,
        )
        # 1. 计算每个日期的平均值（按列计算）
        daily_avg = final_df.drop(columns=[first_col_name]).mean(axis=0)

        # 2. 转换为 DataFrame，并添加 "省份" 列
        result_df = pd.DataFrame(
            {
                "Province": [self.province_name],  # 固定值
                **daily_avg.to_dict(),  # 展开日期和平均值
            }
        )

        # 3. 调整列顺序（省份列放第一列）
        cols = ["Province"] + [col for col in result_df.columns if col != "Province"]
        result_df = result_df[cols]

        # 保存结果
        result_df.to_csv(
            os.path.join(self.output_path, "province_ave_" + target + ".csv"), index=False
        )
        return os.path.join(self.output_path, "province_station_" + target + ".csv"), os.path.join(self.output_path, "province_ave_" + target + ".csv")
    def ConvertWeatherData(self, data_path):
        df = pd.read_csv(data_path)

        first_col_name = df.columns[0]

        target = df["v1"].iloc[0]
        
        # 1. 找出数据最完整的站点（数据条数最多的）
        station_counts = df[first_col_name].value_counts()
        max_count = station_counts.max()
        valid_stations = station_counts[station_counts == max_count].index.tolist()

        print(f"保留的站点数: {len(valid_stations)}")
        # print(f"每个站点的数据天数: {max_count}")
        df_valid = df[df[first_col_name].isin(valid_stations)].copy()
        # 6. 计算各省份各指标平均值
        numeric_cols = ['Mean_Pressure', 'Mean_Humidity', 'Mean_Temperature', 
                    'Mean_Wind_Speed', 'Max_Wind_Direction', target]
        result = df_valid.groupby(['Province', 'Date', 'v1'])[numeric_cols].mean().reset_index()
        # print(result.head())
        # 7. 保存结果
        result.to_csv(
            os.path.join(self.output_path, "province_match_ave.csv"), index=False
        )
        return os.path.join(self.output_path, "province_match_ave.csv")



    def ConvertWeatherRemoveData(self, data_path, province_mode=False):
        df = pd.read_csv(data_path)

        if(province_mode==False):
            first_col_name = "Station_Code"
        else:
            first_col_name = "Province"

        target = df["v1"].iloc[0]
        
        # 1. 找出数据最完整的站点（数据条数最多的）
        station_counts = df[first_col_name].value_counts()
        max_count = station_counts.max()
        valid_stations = station_counts[station_counts == max_count].index.tolist()

        print(f"保留的站点数: {len(valid_stations)}")
        print(f"每个站点的数据天数: {max_count}")

        
        # 2. 筛选出数据完整的站点
        df_valid = df[df[first_col_name].isin(valid_stations)].copy()

        # 3. 按省份和日期分组，计算平均值
        result = df_valid.groupby(['Province', 'Date']).agg({
            'Mean_Pressure': 'mean',
            'Mean_Humidity': 'mean',
            'Mean_Temperature': 'mean',
            'Mean_Wind_Speed': 'mean',
            'Max_Wind_Direction': 'mean',
            target: 'mean',
            'human_effect': 'mean',
            'natural_effect': 'mean'
        }).reset_index()

        # 4. 添加时间相关列（从原始数据中保留）
        time_cols = ['year', 'month', 'day', 'hour', 'weekday', 'dayofyear']
        for col in time_cols:
            # 取每个日期第一次出现的值（所有站点同一天的值相同）
            date_values = df_valid.drop_duplicates('Date').set_index('Date')[col]
            result[col] = result['Date'].map(date_values)

        # 5. 重新排列列顺序
        cols_order = ['Province', 'Date'] + time_cols + [
            'Mean_Pressure', 'Mean_Humidity', 'Mean_Temperature',
            'Mean_Wind_Speed', 'Max_Wind_Direction', target,
            'human_effect', 'natural_effect'
        ]
        result = result[cols_order]

        result.to_csv(
            os.path.join(self.output_path, "province_weather_remove_ave_" + target + ".csv"), index=False
        )
        return os.path.join(self.output_path, "province_weather_remove_ave_" + target + ".csv")


