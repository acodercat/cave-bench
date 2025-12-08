import yfinance as yf
import pandas as pd

msft = yf.Ticker("MSFT")

# 1. 获取数据
bs = msft.balance_sheet        # 资产负债表
inc = msft.income_stmt         # 利润表 (Annual)
cf = msft.cashflow             # 现金流量表

# 2. 转置数据
bs_t = bs.T
inc_t = inc.T
cf_t = cf.T

# 查看一下列名，看看有哪些指标可以用
print(bs_t.columns)
print(inc_t.columns)
print(cf_t.columns)

# 3. 保存数据到CSV文件
bs_t.to_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Finance/MSFT/msft_balance_sheet.csv")
inc_t.to_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Finance/MSFT/msft_income_statement.csv")
cf_t.to_csv("/home/luobeier/cave-bench/benchmarks/data_analysis/Finance/MSFT/msft_cash_flow.csv")