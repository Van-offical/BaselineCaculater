# main.py
import os
import glob
from init import load_config, create_directories, get_data_paths, get_weather_config
import dataManager
# 以下模块请确保已实现并在 PYTHONPATH 中
import dataMatching
import weatherRemove
import weatherRemovalBaselineCaculator
import baselineCaculator
import dataConvert

def _safe_glob(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return files[0]

def main():
    # 加载配置
    config = load_config("config.yaml")
    
    # 显示已有数据情况
    dataManager.display_available_data()
    
    # 询问是否添加新数据
    add_choice = input("是否添加新的数据文件? (y/n): ").strip().lower()
    if add_choice == "y":
        dataManager.interactive_add_data(config)
    
    # 选择用于计算的数据组合
    year, pollutant = dataManager.select_calculation_data()
    if year is None or pollutant is None:
        print("无可供计算的数据组合，退出。")
        return
    
    # 更新配置中的污染物（气象处理时会用到）
    config["weather_config"]["target_col"] = pollutant
    
    # 初始化输出目录和数据路径
    output_dirs = create_directories(int(year), pollutant, config)
    paths = get_data_paths(int(year), pollutant, config)

    # 是否跳过数据匹配
    skip_matching = input("是否跳过数据匹配处理? (y/n): ").strip().lower() == "y"
    # 检查output_dirs[0]文件夹以及子文件夹内是否有csv文件
    if not skip_matching and not os.path.exists(os.path.join(output_dirs[0], "*", "*.csv")):
        # 数据匹配处理
        print("无匹配数据开始数据匹配处理...")
        skip_matching = False
    if not skip_matching:
        # 数据匹配处理
        print("开始数据匹配处理...")
        province_list = dataMatching.process_data(
            station_data_path=paths["station_data"],
            data_prs_path=paths["prs"],
            data_rhu_path=paths["rhu"],
            data_tem_path=paths["tem"],
            data_win_path=paths["win"],
            value_path=paths["value"],
            output_dir=output_dirs[0],
            value_name=pollutant,
        )
    else: 
        # 获取省份列表
        province_list = [f for f in os.listdir(output_dirs[0]) if os.path.isdir(os.path.join(output_dirs[0], f))]
    # 处理数据选择
    print(f"共有{len(province_list)}个省份供计算：")
    for i, province in enumerate(province_list):
        print(f"{i+1}. {province}")
    choice = input("请选择要计算的省份编号(0表示全部)：").strip()
    if choice == "0":
        province_list = province_list
    elif choice.isdigit() and 1 <= int(choice) <= len(province_list):
        province_list = [province_list[int(choice)-1]]
    else:
        print("无效的选择，退出。")
        return
    
    for province in province_list:
        # 气象数据去除处理
        # 选择是否跳过气象数据去除处理
        skip_weather_remove = input(f"是否跳过{province}各站点的气象数据去除处理? (y/n): ").strip().lower() == "y"
        if skip_weather_remove:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[1], province, "analysis_results.csv")):
                print(f"不存在{province}各站点的气象数据去除结果，继续处理。")
                skip_weather_remove = False
        if not skip_weather_remove:
            print(f"开始{province}各站点的气象数据去除处理...")
            # 设置输出文件夹
            weather_config = get_weather_config(os.path.join(output_dirs[1], province), pollutant, config)
            # 设置输入数据
            matched_data = _safe_glob(os.path.join(output_dirs[0], province, "*.csv"))
            weatherRemove.main(input_path=matched_data, config=weather_config)

        Convert = dataConvert.DataConvert(
            data_path= _safe_glob(os.path.join(output_dirs[0], province, "*.csv")),
            output_path=os.path.join(output_dirs[4], province),
            province_name=province,
        )
        Convert.readData()
        province_station_csv, province_ave_csv = Convert.ConvertData()
        province_weather_remove_csv = Convert.ConvertWeatherRemoveData(_safe_glob(os.path.join(output_dirs[1], province, "*.csv")))

        province_match_ave_csv = Convert.ConvertWeatherData(_safe_glob(os.path.join(output_dirs[0], province, "*.csv")))

        # 全省去气象处理
        # 选择是否跳过全省去气象处理
        skip_weather_remove_ave = input(f"是否跳过{province}的全省去气象处理? (y/n): ").strip().lower() == "y"
        if skip_weather_remove_ave:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[7], province, "analysis_results.csv")):
                print(f"不存在{province}的全省去气象处理结果，继续处理。")
                skip_weather_remove_ave = False
        if not skip_weather_remove_ave:
            print(f"开始{province}的全省去气象处理...")
            # 设置输出文件夹
            weather_config = get_weather_config(os.path.join(output_dirs[7], province), pollutant, config)
            # 设置输入数据
            weatherRemove.main(input_path=province_match_ave_csv, config=weather_config, province_mode=True)

        # 基线计算
        # 选择是否跳过基线计算
        skip_baseline = input(f"是否跳过{province}各站点的基线计算? (y/n): ").strip().lower() == "y"
        if skip_baseline:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[3], province, "infl_draw_1")):
                print(f"不存在{province}各站点的基线计算结果，继续处理。")
                skip_baseline = False
        if not skip_baseline:
            print(f"开始{province}各个站点基线计算...")
            baseline_processor = baselineCaculator.BaseLineCalculator(
                year=year,
                pollutant=pollutant,
                target_dir_pre=os.path.join(output_dirs[3], province),
            )
            baseline_processor.init_raw_data_from_csv(province_station_csv)
            baseline_processor.full_process()
        
        # 去气象基线计算
        # 选择是否跳过去气象基线计算
        skip_weather_baseline = input(f"是否跳过{province}各站点的去气象基线计算? (y/n): ").strip().lower() == "y"
        if skip_weather_baseline:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[2], province, "infl_draw_1")):
                print(f"不存在{province}各站点的去气象基线计算结果，继续处理。")
                skip_weather_baseline = False
        if not skip_weather_baseline:
            print(f"开始{province}各个站点去气象基线计算...")
            processor = weatherRemovalBaselineCaculator.EnhancedBaseLineCalculator(
                year=year,
                pollutant=pollutant,
                target_dir=os.path.join(output_dirs[2], province),
            )
            processor.load_data(f"{os.path.join(output_dirs[1], province)}/analysis_results.csv")
            processor.generate_reports()

        # 全省平均基线计算
        # 选择是否跳过全省平均基线计算
        skip_ave_baseline = input(f"是否跳过{province}的全省平均基线计算? (y/n): ").strip().lower() == "y"
        if skip_ave_baseline:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[5], province, "infl_draw_1")):
                print(f"不存在{province}的全省平均基线计算结果，继续处理。")
                skip_ave_baseline = False
        if not skip_ave_baseline:
            print(f"开始{province}全省平均基线计算...")
            ave_processor = baselineCaculator.BaseLineCalculator(
                year=year,
                pollutant=pollutant,
                target_dir_pre=os.path.join(output_dirs[5], province),
            )
            ave_processor.init_raw_data_from_csv(province_ave_csv)
            ave_processor.full_process()

        # 去气象全省平均基线计算
        # 选择是否跳过去气象全省平均基线计算
        skip_weather_ave_baseline = input(f"是否跳过{province}的去气象全省平均基线计算? (y/n): ").strip().lower() == "y"
        if skip_weather_ave_baseline:
            # 检查是否已有结果
            if not os.path.exists(os.path.join(output_dirs[6], province, "infl_draw_1")):
                print(f"不存在{province}的去气象全省平均基线计算结果，继续处理。")
                skip_weather_ave_baseline = False
        if not skip_weather_ave_baseline:
            print(f"开始{province}全省平均去气象基线计算...")
            ave_processor = weatherRemovalBaselineCaculator.EnhancedBaseLineCalculator(
                year=year,
                pollutant=pollutant,
                target_dir=os.path.join(output_dirs[6], province),
            )
            ave_processor.load_data(province_weather_remove_csv)
            ave_processor.generate_reports()

        
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
