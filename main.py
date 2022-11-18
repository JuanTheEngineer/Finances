import pandas as pd
import matplotlib.pyplot as plt


def prep_bofa_or_chase_df(df, start_net_worth=0, date_format='%m/%d/%Y'):
    df.index.rename('Date', inplace=True)
    df = df.filter(['Date', 'Amount'], axis=1)

    # convert transactions to floats
    if isinstance(df['Amount'][0], str):
        df['Amount'] = pd.to_numeric(df['Amount'].str.strip().str.replace(",", ""))

    df = df.groupby(['Date']).sum()

    # Use pandas.to_datetime() to change datetime format
    df.index = pd.to_datetime(df.index, format=date_format)

    df = df.sort_index()

    # add starting network to first transaction
    df['Amount'].iat[-1] = df['Amount'].iat[-1] + start_net_worth

    return df


def calc_cumulative_worth_col(df, part=1.0):
    temp = df
    temp = temp.iloc[::-1]['Amount'].cumsum()
    df['NetWorth'] = temp.iloc[::-1]
    df['NetWorth'] = df['NetWorth'] * part
    return df


def plot_chart(filename, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./images/' + filename + '.png')

def complete_df(df_inbound):
    dates = pd.date_range(df_inbound.index.min(), df_inbound.index.max())
    df = pd.DataFrame(index=dates)
    df = df.join(df_inbound['NetWorth'])
    return df

### NEXT: step thru code and validate data
if __name__ == '__main__':
    # Last 6 months from Nov 16 2022
    bofa_df = pd.read_csv('data/BofA_22-05-16_to_22-11-16.csv', index_col=1)
    bofa_df = prep_bofa_or_chase_df(bofa_df, 8648.13 + 3132.23)
    bofa_df = calc_cumulative_worth_col(bofa_df)


    chase_df = pd.read_csv('data/Chase_22-05-16_to_22_11_16.csv', index_col=0)
    chase_df = prep_bofa_or_chase_df(chase_df, 8648.13 + 3132.23, date_format='%m/%d/%y')
    chase_df = calc_cumulative_worth_col(chase_df, part=0.40)

    chase_and_bofa_df = pd.concat([bofa_df, chase_df]).groupby('Date', as_index=True).sum()

    chase_and_bofa_df = complete_df(chase_and_bofa_df)

    chase_and_bofa_df =

    chase_and_bofa_df['NetWorth'].plot(color="tab:purple")
    plot_chart(filename='chase_and_bofa_net_3m', title='Net Worth 3 months BofA', xlabel='Date', ylabel='Value')
    fig1 = plt.figure()
    print()