import osmnx as ox
import pandas as pd
import numpy as np
import time

# --- 配置 ---

# 1. 允许缓存以加快重试速度，并开启控制台日志
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 180  # 设置超时为3分钟，防止大区域（如离岛）下载中断

# 2. 香港 18 个行政区完整列表
DISTRICTS = [
    "Central and Western District, Hong Kong", # 中西区
    "Wan Chai District, Hong Kong",           # 湾仔区
    "Eastern District, Hong Kong",            # 东区
    "Southern District, Hong Kong",           # 南区
    "Yau Tsim Mong District, Hong Kong",      # 油尖旺区
    "Sham Shui Po District, Hong Kong",       # 深水埗区
    "Kowloon City District, Hong Kong",       # 九龙城区
    "Wong Tai Sin District, Hong Kong",       # 黄大仙区
    "Kwun Tong District, Hong Kong",          # 观塘区
    "Kwai Tsing District, Hong Kong",         # 葵青区
    "Tsuen Wan District, Hong Kong",          # 荃湾区
    "Tuen Mun District, Hong Kong",           # 屯门区
    "Yuen Long District, Hong Kong",          # 元朗区
    "North District, Hong Kong",              # 北区
    "Tai Po District, Hong Kong",             # 大埔区
    "Sha Tin District, Hong Kong",            # 沙田区
    "Sai Kung District, Hong Kong",           # 西贡区
    "Islands District, Hong Kong"             # 离岛区
]

# 3. 定义需要抓取的 POI 类型 (映射到测试场景所需的 category)
TAGS = {
    'amenity': ['cafe', 'fast_food', 'school', 'hospital'],
    'shop': ['supermarket'],
    'leisure': ['park']
}

def fetch_full_hk_data():
    print(f"--- 开始下载香港全境 18 区数据 ---")
    all_pois = []
    
    for place in DISTRICTS:
        # 提取短名用于显示和存储 (例如 "Central and Western")
        district_short = place.split(" District")[0]
        print(f"正在下载: {district_short}...", end=" ", flush=True)
        
        try:
            # 使用 osmnx 获取几何特征
            gdf = ox.features_from_place(place, tags=TAGS)
            
            if gdf.empty:
                print("[无数据]")
                continue

            # --- 数据清洗 ---
            
            # 1. 几何处理：无论是 Point 还是 Polygon，统一取中心点坐标
            gdf['lat'] = gdf.geometry.centroid.y
            gdf['lon'] = gdf.geometry.centroid.x
            
            # 2. 类别映射：将 OSM 的复杂 tag 映射为我们需要的简单 category
            gdf['category'] = np.nan
            for key, values in TAGS.items():
                if key in gdf.columns:
                    for val in values:
                        # 如果该行的 key 列等于 val，则标记 category
                        gdf.loc[gdf[key] == val, 'category'] = val
            
            # 3. 过滤掉没有映射成功的行
            gdf = gdf.dropna(subset=['category'])
            
            # 4. 统一名称，防止空值
            if 'name' not in gdf.columns:
                gdf['name'] = "Unknown"
            else:
                gdf['name'] = gdf['name'].fillna("Unknown")
            
            # 5. 选取并重命名列
            cols = ['name', 'category', 'lat', 'lon']
            df_temp = gdf[cols].copy()
            df_temp['district'] = district_short 
            
            all_pois.append(df_temp)
            print(f"[成功] 获取 {len(df_temp)} 条")
            
        except Exception as e:
            print(f"[失败] 错误信息: {e}")
        
        # 暂停 2 秒，礼貌抓取
        time.sleep(2)

    if not all_pois:
        raise ValueError("所有区域均下载失败，请检查网络连接 (需访问 OSM API)。")

    print("\n正在合并并最终清洗数据...")
    final_df = pd.concat(all_pois, ignore_index=True)
    
    # 最终去重 (防止行政区边界处的重叠点)
    initial_len = len(final_df)
    final_df = final_df.drop_duplicates(subset=['lat', 'lon', 'category'])
    print(f"去除边界重叠点: {initial_len - len(final_df)} 条")

    # 生成简单的整数 ID (用于测试场景中的 Q4)
    final_df['id'] = range(1, len(final_df) + 1)
    
    #调整列顺序
    final_df = final_df[['id', 'name', 'category', 'lat', 'lon', 'district']]

    # 保存 CSV
    output_file = "/home/luobeier/cave-bench/benchmarks/data_analysis/Geospatial/OSM_POI/osm_pois_hk.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n成功! 数据已保存至: {output_file}")
    print(f"总记录数: {len(final_df)}")
    print("列清单: id, name, category, lat, lon, district")
    
    return final_df

if __name__ == "__main__":
    fetch_full_hk_data()