import pandas as pd

# 直接读取官方下载的 parquet 文件
df = pd.read_parquet("/home/luobeier/cave-bench/benchmarks/data_analysis/Geospatial/NYC_TCL/yellow_tripdata_2025-01.parquet")

# 如果非要保存为 csv
df.to_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Geospatial/NYC_TCL/yellow_tripdata_2025_01.csv", index=False)