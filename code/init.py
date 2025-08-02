# init.py
import os
import glob
from typing import Dict, Any
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载外部配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_directories(year: int, value: str, config: Dict[str, Any]) -> tuple:
    """创建所有输出目录"""
    base_result_dir = config.get("base_result_dir", "../result")
    output_dirs = {
        "output_match": os.path.join(
            base_result_dir,
            config["output_directories"]["Matched_Data"],
            str(year),
            value,
        ),
        "output_weather": os.path.join(
            base_result_dir,
            config["output_directories"]["WeatherRemove_Data"],
            str(year),
            value,
        ),
        "output_baseline_with_weatherRemove": os.path.join(
            base_result_dir,
            config["output_directories"]["Baseline_Data_With_WeatherRemove"],
            str(year),
            value,
        ),
        "output_baseline": os.path.join(
            base_result_dir,
            config["output_directories"]["Baseline_Data"],
            str(year),
            value,
        ),
        "output_convert": os.path.join(
            base_result_dir,
            config["output_directories"]["Converted_Data"],
            str(year),
            value,
        ),
        "output_baseline_by_province": os.path.join(
            base_result_dir,
            config["output_directories"]["Province_Baseline_Data"],
            str(year),
            value,
        ),
        "output_baseline_with_weatherRemove_by_province": os.path.join(
            base_result_dir,
            config["output_directories"]["Province_Baseline_Data_With_WeatherRemove"],
            str(year),
            value,
        ),
        "output_province_match": os.path.join(
            base_result_dir,
            config["output_directories"]["Province_Matched_Data"],
            str(year),
            value,
        ),
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return tuple(output_dirs.values())


def _safe_glob(pattern: str) -> str:
    """安全地获取匹配文件的第一个文件路径"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return files[0]


def get_data_paths(year: int, value: str, config: Dict[str, Any]) -> dict:
    """根据配置获取输入数据文件的路径"""
    base_data_dir = config.get("base_data_dir", "../data")
    return {
        "station_data": _safe_glob(os.path.join(base_data_dir, "StationList", "*.csv")),
        "prs": _safe_glob(
            os.path.join(base_data_dir, "MeteorologicalData", str(year), "PRS", "*.csv")
        ),
        "rhu": _safe_glob(
            os.path.join(base_data_dir, "MeteorologicalData", str(year), "RHU", "*.csv")
        ),
        "tem": _safe_glob(
            os.path.join(base_data_dir, "MeteorologicalData", str(year), "TEM", "*.csv")
        ),
        "win": _safe_glob(
            os.path.join(base_data_dir, "MeteorologicalData", str(year), "WIN", "*.csv")
        ),
        "value": _safe_glob(
            os.path.join(base_data_dir, "MeteorologicalData", str(year), value, "*.csv")
        ),
    }


def get_weather_config(
    output_dir: str, value: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """获取气象数据处理的配置参数"""
    weather_config = config.get("weather_config", {})
    weather_config["output_dir"] = output_dir
    weather_config["target_col"] = value
    return weather_config
