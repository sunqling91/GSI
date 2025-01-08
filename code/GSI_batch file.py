# -*- coding: utf-8 -*-
"""
AGSI模型 - 逐像元物候日期计算（自动多年份处理）
"""

import os
import numpy as np
import rasterio
from rasterio.transform import Affine
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd

# 定义AGSI模型所需的函数
def calculate_iVPD(VPD, VPDmin, VPDmax):
    return np.where(VPD <= VPDmin, 1, np.where(VPD >= VPDmax, 0, 1 - (VPD - VPDmin) / (VPDmax - VPDmin)))

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

def calculate_igsi(iVPD, iTmin, iPhoto):
    return iVPD * iTmin * iPhoto

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
        fall_date = len(gsi_series) - np.argmax(fall_condition[::-1])
    else:
        fall_date = np.nan

    return spring_date, fall_date

def main():
    # 定义参数值
    params = {
        'Tm_min': -7.45,
        'Tm_max': 5.88,
        'Photomin': 11.18,
        'Photomax': 14.08,
        'VPDmin': 0.50,
        'VPDmax': 1.17,
        'threshold': 0.14  # 根据注释调整阈值为0.5
    }

    # 定义父文件夹路径
    VPD_parent_folder = 'F:/GSIdata/grassland_VPD_output/'
    Tmin_parent_folder = 'F:/GSIdata/grassland_Tmin_output/'
    output_folder = 'F:/GSIdata/Phenology_grassland/GSI0/'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 定义年份范围
    start_year = 1960
    end_year = 1999
    years = range(start_year, end_year + 1)

    # 读取一个样本文件以获取空间参考和尺寸
    # 使用第一个年份的文件作为样本
    sample_year = str(start_year)
    sample_Tmin_folder = os.path.join(Tmin_parent_folder, sample_year)
    sample_file_path = os.path.join(sample_Tmin_folder, f"Tmin_{sample_year}_1_clipped.tif")
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
    # rasterio.transform.xy返回的是(x, y)，我们只需要y坐标
    _, y_coords = rasterio.transform.xy(transform, y_indices, np.zeros(rows), offset='ul')
    latitudes = np.array(y_coords)  # 假设Y坐标是纬度

    # 初始化纬度矩阵
    latitude_matrix = np.tile(latitudes[:, np.newaxis], (1, cols))

    # 遍历每个年份
    for year in tqdm(years, desc="Processing Years"):
        year_str = str(year)
        print(f"Processing Year: {year_str}")

        # 定义当前年份的VPD和Tmin文件夹路径
        VPD_folder = os.path.join(VPD_parent_folder, year_str)
        Tmin_folder = os.path.join(Tmin_parent_folder, year_str)

        # 检查当前年份的文件夹是否存在
        if not os.path.isdir(VPD_folder):
            print(f"VPD folder not found for year {year_str}: {VPD_folder}. Skipping this year.")
            continue
        if not os.path.isdir(Tmin_folder):
            print(f"Tmin folder not found for year {year_str}: {Tmin_folder}. Skipping this year.")
            continue

        # 判断是否为闰年
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_year = 366
        else:
            days_in_year = 365

        # 初始化日序列数组
        iTmin_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)
        iVPD_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)
        iPhoto_series = np.zeros((days_in_year, rows, cols), dtype=np.float32)

        # 处理每一天
        for day in tqdm(range(1, days_in_year + 1), desc=f"Year {year_str} - Days", leave=False):
            # 构建日期对象
            date = datetime(year, 1, 1) + timedelta(days=day - 1)
            date_str = date.strftime('%Y-%m-%d')  # 格式化为 YYYY-MM-DD

            # 构建VPD文件路径
            vpd_filename = f"VPD_{date_str}_clipped.tif"
            vpd_path = os.path.join(VPD_folder, vpd_filename)
            if os.path.exists(vpd_path):
                with rasterio.open(vpd_path) as src_vpd:
                    VPD = src_vpd.read(1).astype(np.float32)
                iVPD = calculate_iVPD(VPD, params['VPDmin'], params['VPDmax'])
                iVPD_series[day - 1] = iVPD
            else:
                print(f"VPD file not found: {vpd_path}. Setting iVPD to 0 for this day.")
                # 保持 iVPD 为 0

            # 构建Tmin文件路径
            Tmin_filename = f"Tmin_{year_str}_{day}_clipped.tif"
            Tmin_path = os.path.join(Tmin_folder, Tmin_filename)
            if os.path.exists(Tmin_path):
                with rasterio.open(Tmin_path) as src_Tmin:
                    Tmin = src_Tmin.read(1).astype(np.float32)
                iTmin = calculate_iTmin(Tmin, params['Tm_min'], params['Tm_max'])
                iTmin_series[day - 1] = iTmin
            else:
                print(f"Tmin file not found: {Tmin_path}. Setting iTmin to 0 for this day.")
                # 保持 iTmin 为 0

            # 计算 daylength
            day_of_year = day
            daylength = calculate_daylength(latitude_matrix, day_of_year)

            # 计算 iPhoto
            iPhoto = calculate_iPhoto(daylength, params['Photomin'], params['Photomax'])
            iPhoto_series[day - 1] = iPhoto

        # 计算 iGSI
        iGSI = calculate_igsi(
            iVPD=iVPD_series,
            iTmin=iTmin_series,
            iPhoto=iPhoto_series
        )

        # 计算 GSI（21天滑动平均）
        # 使用 Pandas 的 rolling 平滑 GSI
        # 为了提高性能，将 GSI 数据重塑为二维数组进行批量处理
        GSI_reshaped = iGSI.reshape(days_in_year, -1)
        GSI_smoothed = pd.DataFrame(GSI_reshaped).rolling(window=21, center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill').values
        # 将平滑后的数据重塑回原来的形状
        GSI = GSI_smoothed.reshape(days_in_year, rows, cols)

        # 计算物候日期
        spring_dates = np.full((rows, cols), np.nan, dtype=np.float32)
        fall_dates = np.full((rows, cols), np.nan, dtype=np.float32)

        # 遍历每个像元计算春秋物候日期
        for r in tqdm(range(rows), desc=f"Year {year_str} - Pixels Rows", leave=False):
            for c in range(cols):
                gsi_series = GSI[:, r, c]
                spring_date, fall_date = get_phenology_dates(gsi_series, params['threshold'])
                spring_dates[r, c] = spring_date
                fall_dates[r, c] = fall_date

        # 保存结果为TIF文件
        spring_output_path = os.path.join(output_folder, f"spring_date_{year_str}.tif")
        fall_output_path = os.path.join(output_folder, f"fall_date_{year_str}.tif")

        # 更新 profile 以适应输出数据类型
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        # 保存春季日期
        with rasterio.open(spring_output_path, 'w', **profile) as dst_spring:
            dst_spring.write(spring_dates, 1)

        # 保存秋季日期
        with rasterio.open(fall_output_path, 'w', **profile) as dst_fall:
            dst_fall.write(fall_dates, 1)

        print(f"Year {year_str} processing completed. Output saved.")

    print("All years processed successfully.")

if __name__ == "__main__":
    main()
