import pandas as pd

#python 3.6 required!

df = pd.read_csv("crypto_data/LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])

# print(df.head())

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"

main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
for ratio in ratios:
    # print(ratio)
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume']) 

    # rename volumes
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # use onlt price and volume columns

    if len(main_df)==0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)
# print(main_df.head())

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df.head())

