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
OUTPUT_FILE = "/home/luobeier/cave-bench/benchmarks/data_analysis/Meteorological/HK_Weather/hk_weather_2025.csv"

# SQL 查询语句
SQL_QUERY = """
SELECT
    id,
    station_id,
    temperature,
    visibility,
    pressure,
    humidity,
    wind_degree,
    wind_speed,
    wind_gust,
    datetime
FROM station_weather
WHERE
    datetime >= '2025-01-01 00:00:00'
    AND datetime <= '2025-12-08 23:59:59'
ORDER BY datetime ASC;
"""
# ===========================================

def download_data():
    conn = None
    try:
        print(f"正在连接数据库 ({DB_HOST}:{DB_PORT})...")
        # 建立连接
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        
        print("连接成功，正在执行 SQL 并下载数据...")
        
        # 使用 pandas 直接读取 SQL 结果为 DataFrame
        # parse_dates=['datetime'] 会自动把 datetime 列转为时间格式，方便后续分析
        df = pd.read_sql(SQL_QUERY, conn, parse_dates=['datetime'])
        
        print(f"下载完成，共获取 {len(df)} 条数据。")
        
        # 保存为 CSV
        print(f"正在保存为 {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig 防止中文乱码（虽然这里主要是数字）
        
        print("✅ 成功！文件已保存。")

    except psycopg2.OperationalError as e:
        print(f"❌ 连接数据库失败: {e}")
        print("请检查主机、端口、用户名或密码是否正确。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        if conn:
            conn.close()
            print("数据库连接已关闭。")

if __name__ == "__main__":
    download_data()