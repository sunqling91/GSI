# -*- coding: utf-8 -*-
"""
AGSI模型 - 逐像元物候日期计算
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd


# 定义AGSI模型所需的函数
def calculate_iPre(Pre, Premin, Premax):
    return np.where(Pre <= Premin, 0,
                    np.where(Pre >= Premax, 1, (Pre - Premin) / (Premax - Premin)))


def calculate_iTmax(Tmax, Ta_min, Ta_max):
    iTmax = np.where(Tmax <= Ta_min, 0,
                     np.where((Ta_min < Tmax) & (Tmax < 15), (Tmax - Ta_min) / (15 - Ta_min),
                              np.where((15 <= Tmax) & (Tmax < 20), 1,
                                       np.where((20 < Tmax) & (Tmax < Ta_max), (Ta_max - Tmax) / (Ta_max - 20),
                                                np.where(Tmax >= Ta_max, 0, 0)))))
    return iTmax


def calculate_iTmin(Tmin, Tm_min, Tm_max):
    iTmin = np.where(Tmin <= Tm_min, 0,
                     np.where((Tm_min < Tmin) & (Tmin < Tm_max), (Tmin - Tm_min) / (Tm_max - Tm_min),
                              np.where(Tmin >= Tm_max, 1, 0)))
    return iTmin


def declination_angle(day_of_year):
    return np.arcsin(0.39795 * np.cos(2 * np.pi / 365.25 * (day_of_year - 173)))


def calculate_daylength(lat, day_of_year, p=0.833):
    lat_rad = np.radians(lat)  # Convert latitude to radians
    phi = declination_angle(day_of_year)  # Calculate declination angle in radians

    # Calculate the argument for arccos, ensuring it's within [-1, 1]
    acos_argument = (np.sin(np.radians(p)) + np.sin(lat_rad) * np.sin(phi)) / (np.cos(lat_rad) * np.cos(phi))
    acos_argument = np.clip(acos_argument, -1, 1)

    # Calculate daylength using the given formula
    daylength = 24 - (24 / np.pi) * np.arccos(acos_argument)
    return daylength


def calculate_iPhoto(daylength, Photomin, Photomax):
    return np.where(daylength <= Photomin, 0,
                    np.where(daylength >= Photomax, 1, (daylength - Photomin) / (Photomax - Photomin)))


def calculate_igsi(iTmax, iTmin, iPhoto, iPre):
    return iTmax * iTmin * iPhoto * iPre


def get_phenology_dates(gsi_series, threshold):
    """
    计算物候日期。

    参数:
    - gsi_series: 已经进行过滑动平均的GSI时间序列 (numpy array)
    - threshold: 阈值

    返回:
    - spring_date: 春季日期
    - fall_date: 秋季日期
    """
    # 计算春季日期
    spring_condition = gsi_series >= threshold
    if np.any(spring_condition):
        spring_date = np.argmax(spring_condition) + 1  # +1 因为索引从0开始
    else:
        spring_date = np.nan

    # 计算秋季日期为全年中最后一个满足条件的日期
    fall_condition = gsi_series >= threshold
    if np.any(fall_condition):
        fall_date = np.argmax(fall_condition[::-1])  # 倒序后找到第一个满足条件的索引
        fall_date = len(gsi_series) - fall_date  # 转换为正序日期
    else:
        fall_date = np.nan

    return spring_date, fall_date


def main():
    # 定义参数值
    params = {
        'Ta_min': 2.72,
        'Ta_max': 27.58,
        'Tm_min': -12.58,
        'Tm_max': 4.95,
        'Photomin': 11.86,
        'Photomax': 14.14,
        'Premin': 0.01,
        'Premax': 2.32,
        'threshold': 0.04  # 根据注释调整阈值为0.5
    }

    # 定义文件夹路径
    pre_folder = 'F:/GSIdata/grassland_Means_pre/2000/'
    Tmax_folder = 'F:/GSIdata/grassland_Tmax_output/2000/'
    Tmin_folder = 'F:/GSIdata/grassland_Tmin_output/2000/'
    output_folder = 'F:/GSIdata/Phenology_grassland/平均参数AGSI/'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 定义年份范围
    start_year = 2000
    end_year = 2000
    years = range(start_year, end_year + 1)

    # 读取一个样本文件以获取空间参考和尺寸
    # 更新样本文件路径为新的命名格式
    sample_file_path = os.path.join(pre_folder, f"pre_{start_year}_1_30mean.tif")
    if not os.path.exists(sample_file_path):
        raise FileNotFoundError(f"Sample file not found: {sample_file_path}")

    with rasterio.open(sample_file_path) as sample_file:
        profile = sample_file.profile
        rows, cols = sample_file.shape
        transform = sample_file.transform
        crs = sample_file.crs

    # 计算每个像元的纬度
    # 通过行号和仿射变换获取每行的Y坐标（纬度）
    y_indices = np.arange(rows)
    _, y_coords = rasterio.transform.xy(transform, y_indices, np.zeros(rows), offset='ul')
    latitudes = np.array(y_coords)  # 假设Y坐标是纬度

    # 初始化纬度矩阵
    latitude_matrix = np.tile(latitudes[:, np.newaxis], (1, cols))

    # 循环处理每个年份
    for year in tqdm(years, desc="Processing Years"):
        print(f"Processing Year: {year}")

        # 判断是否为闰年
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_year = 366
        else:
            days_in_year = 365

        # 初始化日序列数组
        iTmax_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)
        iTmin_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)
        iPre_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)
        iPhoto_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)

        # 处理每一天
        for day in tqdm(range(1, days_in_year + 1), desc=f"Year {year} - Days", leave=False):
            # 构建日期对象
            date = datetime(year, 1, 1) + timedelta(days=day - 1)

            # 构建文件名不再使用date_str，而使用新的命名格式
            # Pre 文件
            pre_filename = f"pre_{year}_{day}_30mean.tif"
            pre_path = os.path.join(pre_folder, pre_filename)
            if os.path.exists(pre_path):
                with rasterio.open(pre_path) as src_pre:
                    Pre = src_pre.read(1).astype(np.float32)
                iPre = calculate_iPre(Pre, params['Premin'], params['Premax'])
                iPre_series[day - 1] = iPre
            else:
                print(f"Pre file not found: {pre_path}")
                # 保持 iPre 为 0

            # Tmax 文件
            Tmax_filename = f"Tmax_{year}_{day}_clipped.tif"
            Tmax_path = os.path.join(Tmax_folder, Tmax_filename)
            if os.path.exists(Tmax_path):
                with rasterio.open(Tmax_path) as src_Tmax:
                    Tmax = src_Tmax.read(1).astype(np.float32)
                iTmax = calculate_iTmax(Tmax, params['Ta_min'], params['Ta_max'])
                iTmax_series[day - 1] = iTmax
            else:
                print(f"Tmax file not found: {Tmax_path}")
                # 保持 iTmax 为 0

            # Tmin 文件
            Tmin_filename = f"Tmin_{year}_{day}_clipped.tif"
            Tmin_path = os.path.join(Tmin_folder, Tmin_filename)
            if os.path.exists(Tmin_path):
                with rasterio.open(Tmin_path) as src_Tmin:
                    Tmin = src_Tmin.read(1).astype(np.float32)
                iTmin = calculate_iTmin(Tmin, params['Tm_min'], params['Tm_max'])
                iTmin_series[day - 1] = iTmin
            else:
                print(f"Tmin file not found: {Tmin_path}")
                # 保持 iTmin 为 0

            # 计算 daylength
            day_of_year = day
            daylength = calculate_daylength(latitude_matrix, day_of_year)

            # 计算 iPhoto
            iPhoto = calculate_iPhoto(daylength, params['Photomin'], params['Photomax'])
            iPhoto_series[day - 1] = iPhoto

        # 计算 iGSI
        iGSI = calculate_igsi(
            iTmax=iTmax_series,
            iTmin=iTmin_series,
            iPhoto=iPhoto_series,
            iPre=iPre_series
        )

        # 计算 GSI（21天滑动平均）
        # 使用 Pandas 的 rolling 平滑 GSI
        # 由于 apply_along_axis 对大数据量很慢，考虑优化方法
        # 这里继续使用 apply_along_axis，若性能成为瓶颈，可进一步优化

        GSI = np.apply_along_axis(
            lambda x: pd.Series(x).rolling(window=21, center=True).mean().fillna(method='bfill').fillna(
                method='ffill').values, axis=0, arr=iGSI)
        # 由于滑动平均后仍为多维数组，保持形状不变

        # 计算物候日期
        spring_dates = np.full((rows, cols), np.nan, dtype=np.float32)
        fall_dates = np.full((rows, cols), np.nan, dtype=np.float32)

        # 遍历每个像元计算春秋物候日期
        for r in tqdm(range(rows), desc=f"Year {year} - Pixels Rows", leave=False):
            for c in range(cols):
                gsi_series = GSI[:, r, c]
                spring_date, fall_date = get_phenology_dates(gsi_series, params['threshold'])
                spring_dates[r, c] = spring_date
                fall_dates[r, c] = fall_date

        # 保存结果为TIF文件
        spring_output_path = os.path.join(output_folder, f"spring_date_{year}.tif")
        fall_output_path = os.path.join(output_folder, f"fall_date_{year}.tif")

        # 更新 profile 以适应输出数据类型
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        # 保存春季日期
        with rasterio.open(spring_output_path, 'w', **profile) as dst_spring:
            dst_spring.write(spring_dates, 1)

        # 保存秋季日期
        with rasterio.open(fall_output_path, 'w', **profile) as dst_fall:
            dst_fall.write(fall_dates, 1)

        print(f"Year {year} processing completed. Output saved.")

    print("All years processed successfully.")


if __name__ == "__main__":
    main()
