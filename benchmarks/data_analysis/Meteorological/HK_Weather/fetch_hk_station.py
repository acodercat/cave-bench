import psycopg2
import pandas as pd
import sys

# ================= 配置区域 =================
# 数据库连接信息
DB_HOST = "159.138.155.17"
DB_PORT = "5433"          
DB_NAME = "hkgai_air"
DB_USER = "HKGAI_env"    
DB_PASS = "nCFtx2GCF1asdHKGAI_envsNVC7"      

# 输出文件名
OUTPUT_FILE = "/home/luobeier/cave-bench/benchmarks/data_analysis/Meteorological/HK_Weather/hk_weather_stations.csv"


SQL_QUERY = """
SELECT
    id,
    name_en,
    station_code,
    ST_Y(coordinate::geometry) AS latitude,  -- 转换坐标为纬度
    ST_X(coordinate::geometry) AS longitude, -- 转换坐标为经度
    altitude,
    station_type
FROM weather_station
ORDER BY id ASC;
"""
# ===========================================

def download_stations():
    conn = None
    try:
        print(f"正在连接数据库 ({DB_HOST}:{DB_PORT})...")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        
        print("连接成功，正在转换坐标并下载站点信息...")
        
        # 使用 pandas 读取
        df = pd.read_sql(SQL_QUERY, conn)
        
        print(f"下载完成，共获取 {len(df)} 个气象站点。")
        
        # 保存为 CSV
        print(f"正在保存为 {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print("✅ 成功！文件已保存。")

    except psycopg2.OperationalError as e:
        print(f"❌ 连接数据库失败: {e}")
        print("请检查网络、端口或密码。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        if conn:
            conn.close()
            print("数据库连接已关闭。")

if __name__ == "__main__":
    download_stations()