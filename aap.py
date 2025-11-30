import yfinance as yf

# Download AAPL data for 2024
data = yf.download("AAPL", start="2015-01-01", end="2025-11-29")

# Reset index to make Date a column
data = data.reset_index()

# Keep only the columns you need
aapl_df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

# Save to CSV
data.to_csv("AAPL_20100101_20251129.csv", index=False)


# import pandas as pd

# aapl_df = pd.read_csv("AAPL_20240101_20241231.csv")

# # Ensure the Date column is in datetime format and set as index
# aapl_df['Date'] = pd.to_datetime(aapl_df['Date'])
# aapl_df.set_index('Date', inplace=True)

# # Resample by month and sum the Volume
# monthly_volume = aapl_df['Volume'].resample('M').sum()

# # Find the month with the highest volume
# highest_vol_month_idx = monthly_volume.idxmax()
# highest_vol_month_volume_val = monthly_volume.max()

# # Format the month as 'YYYY-MM'
# highest_vol_month_str = highest_vol_month_idx.strftime('%Y-%m')

# print(highest_vol_month_str)
# print(highest_vol_month_volume_val)