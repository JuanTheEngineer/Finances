import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')


# IDEA: an IDE that views packages as if the files were like photos when you are using File Explorer with the Extra
# Large Icons on. This is groundbreaking because it will change the way developers think when they structure their
# repositories.


def plot_chart(filename, title, xlabel, ylabel, file_iterator):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./images/' + filename + '_v' + str(file_iterator) + '.png')


def complete_df(df_inbound):
    dates = pd.date_range(df_inbound.index.min(), df_inbound.index.max())
    df = pd.DataFrame(index=dates)
    df = df.join(df_inbound['NetWorth'])
    return cleanse_dataframe(df)


def cleanse_dataframe(df):
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df


def bollinger_bands(df, column_name, windows, file_iterator):
    def get_rolling_mean(vals, windows):
        return vals.rolling(window=windows).mean()

    def get_rolling_std(vals, windows):
        return vals.rolling(window=windows).std()

    def get_bollinger_bands(roll_mean, roll_std):
        upper_b = roll_mean + roll_std * 1.7
        lower_b = roll_mean - roll_std * 1.7
        return upper_b, lower_b

    def demoBollingerBands(df, column_name, windows):
        # Compute Bollinger Bands
        # 1. Compute rolling mean
        rm_JPM = get_rolling_mean(df[column_name], windows=windows)

        # 2. Compute rolling standard deviation
        rstd_JPM = get_rolling_std(df[column_name], windows=windows)

        # 3. Compute upper and lower bands
        upper_band, lower_band = get_bollinger_bands(rm_JPM, rstd_JPM)

        # Plot raw SPY values, rolling mean and Bollinger Bands
        bbplot = df[column_name].plot(title="Bollinger Bands", label=column_name, color="tab:gray")  # replace with sma
        rm_JPM.plot(label='Rolling mean', ax=bbplot, color="tab:red")
        upper_band.plot(label='upper band', ax=bbplot, color="tab:blue")
        lower_band.plot(label='lower band', ax=bbplot, color="tab:cyan")

        # Band Width = (Upper Bollinger Band - Lower Bollinger Band) / Middle Bollinger Band
        # https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-band-width

        # Add axis labels and legend
        bbplot.set_xlabel("Date")
        bbplot.set_ylabel("Networth")
        bbplot.legend()
        plt.tight_layout()
        plt.savefig('./images/chase_and_bofa_boll_bands_3m_v' + str(file_iterator) + '.png')

        fig2 = plt.figure()

    demoBollingerBands(df, column_name, windows)


def mint_prep(df):
    df['Amount'] = df['Amount'].where(df['Transaction Type'] == 'credit', other=(-1 * df['Amount']))
    return df


'''------------------------------------------------------------------------------------------------------------------'''


class FinanceFrame:
    DATE = 'Date'
    AMOUNT = 'Amount'
    NETWORTH = 'NetWorth'

    def __init__(self, transaction_file_name, initial_networth, index_col, is_bank=False):
        self.INCOME_THRESHOLD = 3000
        self.transaction_file_name = transaction_file_name
        self.initial_networth = initial_networth
        self.is_bank = is_bank
        self.df = pd.read_csv(transaction_file_name, index_col=index_col)
        self.hard_copy_of_df = pd.read_csv(transaction_file_name, index_col=index_col)

        self.networth_df = None

    def correct_column_names(self):
        self.df.index.rename(self.DATE, inplace=True)
        self.df = self.df.filter([self.DATE, self.AMOUNT], axis=1)

    def correct_data_format(self):
        # convert transactions to floats
        if isinstance(self.df[self.AMOUNT][0], str):
            self.df[self.AMOUNT] = pd.to_numeric(self.df[self.AMOUNT].str.strip().str.replace(",", ""))

        self.df = self.df.groupby([self.DATE]).sum()

    def correct_data_values(self):
        def complete_df_range():
            dates = pd.date_range(self.df.index.min(), self.df.index.max())
            temp = pd.DataFrame(index=dates)
            temp.index.name = self.DATE
            self.df = temp.join(self.df[self.AMOUNT])
            return self.df.fillna(0)

        '''
        Take all values above income threshold, sum them, set them all to zero on all respective dates, then disperse sum 
        equally to those days to smooth income over. In the future, we can target all bank deposits to be more accurate.
        '''

        def flatten_income():
            temp = self.df

            # get sum of all values above threshold for counting as income, then split that amongst all dates to
            # simulate rate of daily income
            total_income = temp[temp[self.AMOUNT] > self.INCOME_THRESHOLD].sum()
            daily_income = total_income / temp.size

            # erase real income days by setting to 0
            temp[self.AMOUNT] = temp[self.AMOUNT].where(temp[self.AMOUNT] <= self.INCOME_THRESHOLD, other=0)

            # spread daily income amongst Amount data
            temp += daily_income

            logging.info("These values should be equal: " + str(temp['Amount'].sum()) + str(self.df['Amount'].sum()))
            self.df = temp

        self.df = self.df.sort_index()
        complete_df_range()
        if self.is_bank:
            flatten_income()

        # add starting networth to first transaction
        self.df[self.AMOUNT].iat[0] = self.df[self.AMOUNT].iat[0] + self.initial_networth

        # [WIP] 'part' is too unstable mathematically
        self.df[self.AMOUNT] = self.df[self.AMOUNT]  # * part

    '''
    Note: The proof has been solved to state that when we want to combine to data sets, we get the same result if 
    we group them as transaction data or networth (cumulative) data. 
    '''
    def calc_cumulative_worth_df(self):
        def complete_dataframe():
            dates = pd.date_range(self.df.index.min(), self.df.index.max())
            df = pd.DataFrame(index=dates)
            self.df = df.join(self.df[self.NETWORTH])

        def cleanse_dataframe():
            self.df.fillna(method="ffill", inplace=True)
            self.df.fillna(method="bfill", inplace=True)

        temp = self.df
        temp = temp.iloc[::-1][self.AMOUNT]
        temp = temp.cumsum()
        self.networth_df = self.df.copy()
        self.networth_df.rename(columns={self.AMOUNT: self.NETWORTH})
        self.networth_df[self.NETWORTH] = temp.iloc[::-1]
        complete_dataframe()
        cleanse_dataframe()


class BofAFrame(FinanceFrame):
    DATE_FORMAT = '%m/%d/%Y'

    def __init__(self, transaction_file_name, initial_networth, index_col, is_bank=False):
        # Look at generic prep in Parent class to see what prep has already been done
        super().__init__(transaction_file_name, initial_networth, index_col, is_bank)

        # Generic Prep
        self.correct_column_names()
        self.correct_data_format()

        # BofA specific prep
        self.prep_bofa_datetime()

        # Generic prep
        self.correct_data_values()
        self.calc_cumulative_worth_df()

    def prep_bofa_datetime(self):
        # Use pandas.to_datetime() to change datetime format
        self.df.index = pd.to_datetime(self.df.index, format=self.DATE_FORMAT)


class ChaseFrame(FinanceFrame):
    DATE_FORMAT = '%m/%d/%y'

    def __init__(self, transaction_file_name, initial_networth, index_col, is_bank=False):
        # Look at generic prep in Parent class to see what prep has already been done
        super().__init__(transaction_file_name, initial_networth, index_col, is_bank)

        # Generic Prep
        self.correct_column_names()
        self.correct_data_format()

        # BofA specific prep
        self.prep_bofa_datetime()

        # Generic prep
        self.correct_data_values()
        self.calc_cumulative_worth_df()

    def prep_bofa_datetime(self):
        # Use pandas.to_datetime() to change datetime format
        self.df.index = pd.to_datetime(self.df.index, format=self.DATE_FORMAT)


def run(file_iterator, isBank):
    # current networth - change in networth since start date
    start_networth = 8648.13 + 6364.33
    """
    # Last 6 months from Nov 16 2022
    bofa_df = pd.read_csv('data/BofA_22-05-16_to_22-11-16.csv', index_col=1)
    control_df = bofa_df
    bofa_df = prep_bofa_or_chase_df(bofa_df, start_networth, part=0.40, bank=isBank)

    # plain version (TEST)
    control_df = prep_bofa_or_chase_df(control_df, start_networth, part=0.40, bank=False)

    chase_df = pd.read_csv('data/Chase_22-05-16_to_22_11_16.csv', index_col=0)
    chase_df = prep_bofa_or_chase_df(chase_df, date_format='%m/%d/%y')  # incorporating initial debit is extra work

    control_df = calc_cumulative_worth_col(pd.concat([control_df, chase_df]).groupby('Date', as_index=True).sum())
    chase_and_bofa_df = calc_cumulative_worth_col(pd.concat([bofa_df, chase_df]).groupby('Date', as_index=True).sum())
    
    grouped_df = pd.concat(final_df).groupby('Date', as_index=True).sum()
    """
    # Last 6 months from Nov 16 2022
    mint_df = pd.read_csv('data/MInt_22-08-20_to_22-11-18.csv', index_col=0)
    control_df = mint_df
    mint_prep(mint_df)
    mint_df = prep_bofa_or_chase_df(mint_df, start_networth, part=0.40, bank=isBank)

    # plain version (TEST)
    control_df = prep_bofa_or_chase_df(control_df, start_networth, part=0.40, bank=False)

    grouped_df = mint_df
    control_df = calc_cumulative_worth_col(control_df)
    chase_and_bofa_df = calc_cumulative_worth_col(grouped_df)

    chase_and_bofa_df['NetWorth'].plot(color="tab:red")
    control_df['NetWorth'].plot(color="tab:gray")
    plot_chart(filename='chase_and_bofa_net_3m',
               title='Net Worth 3 months BofA',
               xlabel='Date', ylabel='Value',
               file_iterator=file_iterator)
    fig1 = plt.figure()

    bollinger_bands(chase_and_bofa_df, 'NetWorth', windows=14, file_iterator=file_iterator)


### NEXT: step thru code and validate data
if __name__ == '__main__':
    run(0, False)
    run(1, True)
