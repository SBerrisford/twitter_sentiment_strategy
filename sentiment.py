## Twitter Sentiment Stock Portfolio Strategy

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
import yfinance as yf
import os
import matplotlib.ticker as mtick

# Set the plotting style to 'ggplot'
plt.style.use('ggplot')

# Load the sentiment data from a CSV file
sentiment_df = pd.read_csv(os.path.join('sentiment_data.csv'))

# Convert the 'date' column to datetime format
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Set the DataFrame index to be a MultiIndex with 'date' and 'symbol'
sentiment_df = sentiment_df.set_index(['date', 'symbol'])

# Calculate the engagement ratio as the ratio of comments to likes
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments']/sentiment_df['twitterLikes'] 

# Filter out rows where 'twitterLikes' is less than or equal to 20 or 'twitterComments' is less than or equal to 10
sentiment_df = sentiment_df[(sentiment_df['twitterLikes']>20)&(sentiment_df['twitterComments']>10)]

# Calculate the average engagement ratio for each stock every month
aggragated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='M'), 'symbol'])
                                    [['engagement_ratio']].mean())

# Rank the engagement ratios for each month, with the highest engagement ratio getting the highest rank
aggragated_df['rank'] = (aggragated_df.groupby(level=0)['engagement_ratio']
                        .transform(lambda x: x.rank(ascending=False)))

# Select the top 5 stocks with the highest engagement ratio for each month
filtered_df = aggragated_df[aggragated_df['rank']<6]

# Reset the index for the 'symbol' level to prepare for further processing
filtered_df = filtered_df.reset_index(level=1)

# Offset the index dates by one day to avoid overlap with month-end calculations
filtered_df.index = filtered_df.index+pd.DateOffset(1)

# Reset the index and then set it again with 'date' and 'symbol'
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

# Display the first 20 rows of the filtered DataFrame (for debugging purposes)
filtered_df.head(20)

# Create a list of unique dates from the filtered DataFrame
dates = filtered_df.index.get_level_values('date').unique().tolist()

# Initialize a dictionary to store fixed dates and their corresponding symbols
fixed_dates = {}

# Populate the dictionary with dates as keys and lists of symbols as values
for d in dates:
    
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

# Download fresh stock prices for the shortlisted stocks (excluding 'ATVI')
stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()
stocks_list = [stock for stock in stocks_list if stock != 'ATVI']

# Use yfinance to download historical stock prices for the selected stocks
prices_df = yf.download(tickers = stocks_list,
                        start='2021-01-01',
                        end = '2023-03-01')


# Calculate log returns for the adjusted close prices and drop missing values
returns_df = np.log(prices_df['Adj Close']).diff().dropna()

# Initialize an empty DataFrame to store portfolio returns
portfolio_df = pd.DataFrame()

# Calculate portfolio returns with monthly rebalancing
for start_date in fixed_dates.keys():
    # Define the end date for the current month
    end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    
    # Get the list of symbols for the current month
    cols = fixed_dates[start_date]
    
    # Calculate the mean return for the selected stocks over the current month
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    
    # Concatenate the current month's returns to the portfolio DataFrame
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

# Download historical data for the NASDAQ-100 index (QQQ)
qqq_df = yf.download(tickers='QQQ',
                    start = '2021-01-01',
                    end = '2023-01-01')

# Calculate log returns for the NASDAQ-100 index
qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')

# Merge the portfolio returns with the NASDAQ-100 returns for comparison
portfolio_df = portfolio_df.merge(qqq_ret,
                                left_index=True,
                                right_index=True)

# Calculate cumulative returns for both the portfolio and the NASDAQ-100 index
portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum().sub(1))

# Plot and display the cumulative returns over time
portfolios_cumulative_return.plot(figsize=(16,6))
plt.title('Twitter Sentiment Strategy - Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()




