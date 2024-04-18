import datetime
import time

import streamlit as st
import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from joypy import joyplot
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap


from helpers import tab



st.set_page_config(layout="wide")




def clean_hash(x):
    return x.strip('#')

def get_div_tot(x):
    if pd.isna(x):
        return np.nan
    else:
        return int(x.split('/')[-1])

def get_div_place(x):
    if pd.isna(x):
        return np.nan
    else:
        return int(x.split('/')[0])

def convert_timedelta(td_str):
    td_str = td_str.strip('*').strip('#').split(' ')[-1]
    if len(td_str) == 5:
        td_str = f"00:{td_str}"
    ret = pd.to_timedelta(td_str)
    return ret

def get_df_m():
    df_m = pd.read_csv('MA_Exer_PikesPeak_Males.txt', sep='\t', encoding='ISO-8859-1', converters={'Net Tim': clean_hash})
    col_map = {'Place':'place', 'Num':'num', 'Name':'name', 'Ag':'age', 'Hometown':'hometown', 'Gun Tim':'time_gun', 'Net Tim':'time_net', 'Pace':'pace'}
    df_m = df_m.rename(columns=col_map).replace('<NA>', np.nan)

    m_bins = [5, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89]
    m_labels = ['U14', 'U19', 'U24', 'U29', 'U34', 'U39', 'U44', 'U49', 'U54', 'U59', 'U64', 'U69', 'U74', 'U79', 'U84', 'U89']

    df_m['age'] = df_m['age'].astype('Int64')
    df_m['time_gun'] = df_m['time_gun'].map(lambda x: convert_timedelta(x))
    df_m['time_net'] = df_m['time_net'].map(lambda x: convert_timedelta(x))
    df_m['div_place'] = (df_m['Div/Tot'].map(lambda x: get_div_place(x))).astype('Int64')
    df_m['div_tot'] = (df_m['Div/Tot'].map(lambda x: get_div_tot(x))).astype('Int64')
    df_m['seconds'] = df_m['time_net'].dt.total_seconds().astype('Int64')
    df_m['division'] = pd.cut(df_m['age'], bins=m_bins, labels=m_labels, right=False)
    df_m['div_tot'] = df_m.groupby('division')['division'].transform('count').astype('Int64')
    df_m['div_place'] = df_m.groupby('division')['time_net'].rank(method='dense', ascending=True).astype('Int64')
    df_m['delta_sec'] = df_m['time_gun'].dt.total_seconds().astype('Int64') - df_m['time_net'].dt.total_seconds().astype('Int64')
    return df_m

def get_df_f():
    df_f = pd.read_csv('MA_Exer_PikesPeak_Females.txt', sep='\t', encoding='ISO-8859-1', converters={'Net Tim': clean_hash})
    col_map = {'Place':'place', 'Num':'num', 'Name':'name', 'Ag':'age', 'Hometown':'hometown', 'Gun Tim':'time_gun', 'Net Tim':'time_net', 'Pace':'pace'}
    df_f = df_f.rename(columns=col_map).replace('<NA>', np.nan)
    f_bins = [5, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
    f_labels = ['U14', 'U19', 'U24', 'U29', 'U34', 'U39', 'U44', 'U49', 'U54', 'U59', 'U64', 'U69', 'U74', 'U79']
    df_f['age'] = df_f['age'].astype('Int64')
    df_f['time_gun'] = df_f['time_gun'].map(lambda x: convert_timedelta(x))
    df_f['time_net'] = df_f['time_net'].map(lambda x: convert_timedelta(x))
    df_f['div_place'] = (df_f['Div/Tot'].map(lambda x: get_div_place(x))).astype('Int64')
    df_f['div_tot'] = (df_f['Div/Tot'].map(lambda x: get_div_tot(x))).astype('Int64')
    df_f['seconds'] = df_f['time_net'].dt.total_seconds().astype('Int64')
    df_f['division'] = pd.cut(df_f['age'], bins=f_bins, labels=f_labels, right=False)
    df_f['div_tot'] = df_f.groupby('division')['division'].transform('count').astype('Int64')
    df_f['div_place'] = df_f.groupby('division')['time_net'].rank(method='dense', ascending=True).astype('Int64')
    df_f['delta_sec'] = df_f['time_gun'].dt.total_seconds().astype('Int64') - df_f['time_net'].dt.total_seconds().astype('Int64')
    return df_f

def get_male_ridgeplot(df_m):
    df_m['time_net_minutes'] = (df_m['seconds'] / 60).astype(float)  # Convert to total minutes for easier plotting
    _df_m = df_m.dropna(axis='rows').copy(deep=True)
    is_numeric_m = pd.to_numeric(_df_m['time_net_minutes'], errors='coerce').notna().all()
    print("Column time_net_minutes is all numeric male:", is_numeric_m)
    print(tab(_df_m.sort_values('division').tail()))
    fig, axes = joyplot(data=_df_m[['division','time_net_minutes']], by='division', column='time_net_minutes', colormap=plt.cm.viridis, figsize=(10, 8), overlap=2, normalize=True)

    for ax in axes:
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim([_df_m['time_net_minutes'].min(), _df_m['time_net_minutes'].max()])
        ax.set_axisbelow(False)
    plt.title("Men's Net Time by Division")
    # plt.show()
    return fig


def get_female_ridgeplot(df_f):
    df_f['time_net_minutes'] = (df_f['seconds'] / 60).astype(float)
    _df_f = df_f.dropna(axis='rows', how='any').copy(deep=True)
    is_numeric_f = pd.to_numeric(_df_f['time_net_minutes'], errors='coerce').notna().all()
    print("Column time_net_minutes is all numeric female:", is_numeric_f)
    # _df_f['time_net_minutes'] = 
    print(tab(_df_f.sort_values('division').tail()))
    # _df_f = df_f#.drop(['U84','U89'], axis=0)
    fig, axes = joyplot(data=_df_f[['division','time_net_minutes']], by='division', column='time_net_minutes', colormap=plt.cm.viridis, figsize=(10, 8), overlap=2, normalize=True) #, bw_method=0.3)
    for ax in axes:
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim([_df_f['time_net_minutes'].min(), _df_f['time_net_minutes'].max()])
        ax.set_axisbelow(False)
    plt.title("Women's Net Time by Division")
    # plt.show()
    return fig

def get_histogram(df_m, df_f):
    def format_seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_m['seconds'], name='Male', marker_color='dodgerblue', opacity=0.6))
    fig.add_trace(go.Histogram(x=df_f['seconds'], name='Female', marker_color='orange', opacity=0.6))
    fig.update_layout(barmode='overlay')
    male_median = df_m['seconds'].median()
    fig.add_trace(go.Scatter(x=[df_m['seconds'].mean(), df_m['seconds'].mean()], y=[0, 1], mode='lines', line=dict(dash='dashdot', color='#1e90ff'), name='Male Mean'))
    fig.add_trace(go.Scatter(x=[df_m['seconds'].median(), df_m['seconds'].median()], y=[0, 1], mode='lines', line=dict(dash='dot', color='#1e90ff'), name='Male Median'))
    fig.add_trace(go.Scatter(x=[df_f['seconds'].mean(), df_f['seconds'].mean()], y=[0, 1], mode='lines', line=dict(dash='dashdot', color='orange'), name='Female Mean'))
    fig.add_trace(go.Scatter(x=[df_f['seconds'].median(), df_f['seconds'].median()], y=[0, 1], mode='lines', line=dict(dash='dot', color='orange'), name='Female Median'))


    fig.add_shape(type="line", x0=df_m['seconds'].mean(), x1=df_m['seconds'].mean(), y0=0, y1=1, yref="paper", line=dict(color="#1e90ff", width=2, dash="dashdot"), name='Male Mean')
    fig.add_shape(type="line", x0=df_m['seconds'].median(), x1=df_m['seconds'].median(), y0=0, y1=1, yref="paper", line=dict(color="#1e90ff", width=2, dash="dot"), name='Male Median')
    fig.add_shape(type="line", x0=df_f['seconds'].mean(), x1=df_f['seconds'].mean(), y0=0, y1=1, yref="paper", line=dict(color="orange", width=2, dash="dashdot"), name='Female Mean')
    fig.add_shape(type="line", x0=df_f['seconds'].median(), x1=df_f['seconds'].median(), y0=0, y1=1, yref="paper", line=dict(color="orange", width=2, dash="dot"), name='Female Median')

    tick_vals = np.arange(0, df_m['seconds'].max() + 300, 300)
    fig.update_xaxes(tickvals=tick_vals, ticktext=[format_seconds_to_hms(t) for t in tick_vals])
    # fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    # fig.update_layout(height=500, width=700, autosize=False)
    fig.update_layout(autosize=False)
    # fig.show()
    return fig

def format_timedelta(td, fmt_out='str'):
    if pd.isna(td) == False:
        if fmt_out == 'str':
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        elif fmt_out == 'sec':
            return int(td.total_seconds())
    else:
        return np.nan
    
def get_df_summary(df_m, df_f):
    df_all = pd.concat([df_m, df_f])
    df_summary = pd.DataFrame()
    df_summary['male'] = df_m['time_net'].describe()[['count','mean','min','25%','50%','75%','max']].rename({'50%':'median'})
    df_summary['female'] = df_f['time_net'].describe()[['count','mean','min','25%','50%','75%','max']].rename({'50%':'median'})
    df_summary['all'] = df_all['time_net'].describe()[['count','mean','min','25%','50%','75%','max']].rename({'50%':'median'})
    df_summary.loc['mode','male'] = pd.to_timedelta(df_m['time_net'].dt.round('T').mode().values[0])
    df_summary.loc['mode','female'] = pd.to_timedelta(df_f['time_net'].dt.round('T').mode().values[0])
    df_summary.loc['mode','all'] = pd.to_timedelta(df_all['time_net'].dt.round('T').mode().values[0])
    df_summary_T = df_summary.T

    for col in ['mean','min','25%','median','75%','max','mode']:
        df_summary_T[col] = df_summary_T[col].map(lambda x: format_timedelta(x, 'str'))

    df_summary = df_summary_T.T
    return df_summary


class Divisions:

    def __init__(self):
        self.df_m_summary = pd.DataFrame()
        self.df_m_summary_str = pd.DataFrame()
        self.df_m_summary_sec = pd.DataFrame()
        self.df_f_summary = pd.DataFrame()
        self.df_f_summary_str = pd.DataFrame()
        self.df_f_summary_sec = pd.DataFrame()
        self.cols = ['count','mean','min','25%','median','75%','max']
        self.td_cols = ['mean','min','25%','median','75%','max']

    def get_m_summary(self, df_m):
        _df_m = df_m.dropna(axis='rows', how='any').copy(deep=True)
        self.df_m_summary = _df_m.groupby('division')['time_net'].agg([
            pd.NamedAgg(column='mean', aggfunc='mean'), 
            pd.NamedAgg(column='median', aggfunc='median'),
            pd.NamedAgg(column='min', aggfunc='min'),
            pd.NamedAgg(column='max', aggfunc='max'),
            pd.NamedAgg(column='count', aggfunc='count')
        ]).join(_df_m.groupby('division')['time_net'].quantile([0.25, 0.75]).unstack().rename(columns={0.25: '25%', 0.75: '75%'}))
        self.df_m_summary = self.df_m_summary[self.cols]

        idx_before_all = self.df_m_summary.index.tolist()
        self.df_m_summary.loc['All'] = _df_m['time_net'].describe()[['count','mean','min','25%','50%','75%','max']].rename({'50%':'median'})
        self.df_m_summary = self.df_m_summary.loc[['All']+idx_before_all]
        self.df_m_summary_str = pd.DataFrame()
        self.df_m_summary_sec = pd.DataFrame()
        self.df_m_summary_str['count'] = self.df_m_summary['count']
        self.df_m_summary_sec['count'] = self.df_m_summary['count']
        for col in self.td_cols:
            self.df_m_summary_str[col] = self.df_m_summary[col].map(lambda x: format_timedelta(x, 'str'))
            self.df_m_summary_sec[col] = self.df_m_summary[col].map(lambda x: format_timedelta(x, 'sec'))

        return self.df_m_summary

    def get_f_summary(self, df_f):
        _df_f = df_f.dropna(axis='rows', how='any').copy(deep=True)
        self.df_f_summary = _df_f.groupby('division')['time_net'].agg([
            pd.NamedAgg(column='mean', aggfunc='mean'), 
            pd.NamedAgg(column='median', aggfunc='median'),
            pd.NamedAgg(column='min', aggfunc='min'),
            pd.NamedAgg(column='max', aggfunc='max'),
            pd.NamedAgg(column='count', aggfunc='count')
        ]).join(_df_f.groupby('division')['time_net'].quantile([0.25, 0.75]).unstack().rename(columns={0.25: '25%', 0.75: '75%'}))
        self.df_f_summary = self.df_f_summary[self.cols]

        idx_before_all = self.df_f_summary.index.tolist()
        self.df_f_summary.loc['All'] = _df_f['time_net'].describe()[['count','mean','min','25%','50%','75%','max']].rename({'50%':'median'})
        self.df_f_summary = self.df_f_summary.loc[['All']+idx_before_all]
        self.df_f_summary_str = pd.DataFrame()
        self.df_f_summary_sec = pd.DataFrame()
        self.df_f_summary_str['count'] = self.df_f_summary['count']
        self.df_f_summary_sec['count'] = self.df_f_summary['count']
        for col in self.td_cols:
            self.df_f_summary_str[col] = self.df_f_summary[col].map(lambda x: format_timedelta(x, 'str'))
            self.df_f_summary_sec[col] = self.df_f_summary[col].map(lambda x: format_timedelta(x, 'sec'))

        return self.df_f_summary


def get_mens_heatmap(df_m_summary, df_m_summary_sec, df_m_summary_str):
    m_heat_scaler = MinMaxScaler()
    df_m_summary_scaled = pd.DataFrame(m_heat_scaler.fit_transform(df_m_summary_sec), columns=df_m_summary_sec.columns)
    plt.figure(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list("custom_red_green", ["red", "green"], N=256)
    ax = sns.heatmap(df_m_summary_scaled, annot=df_m_summary_str, fmt='', cmap='YlGnBu', cbar=False)
    ax.set_yticklabels(df_m_summary.index, rotation=0)
    plt.title("Men's Net Times by Division")
    # plt.show()
    return plt.gcf()

def get_womens_heatmap(df_f_summary, df_f_summary_sec, df_f_summary_str):
    f_heat_scaler = MinMaxScaler()
    df_f_summary_scaled = pd.DataFrame(f_heat_scaler.fit_transform(df_f_summary_sec), columns=df_f_summary_sec.columns)
    plt.figure(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list("custom_red_green", ["red", "green"], N=256)
    ax = sns.heatmap(df_f_summary_scaled, annot=df_f_summary_str, fmt='', cmap='YlGnBu', cbar=False)
    ax.set_yticklabels(df_f_summary.index, rotation=0)
    plt.title("Women's Net Times by Division")
    # plt.show()
    return plt.gcf()

def get_seconds_delta_fig(df_):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    df_[['time_net','delta_sec']].set_index('time_net').plot(ax=ax)
    ax.set_xlabel('Total Net Time')
    ax.set_ylabel('Gun Time - Net Time (in seconds)')
    ax.set_title('Gun Time - Net Time Differences by Race Finish Net Time')
    return fig

def get_chris_doe_df(df_m):
    df_u49 = df_m[df_m['division'] == 'U49']
    df_u49[['div_place','div_tot']] = df_u49[['div_place','div_tot']].astype(int)
    df_chrisdoe = df_m[df_m['name'] == 'Chris Doe']
    df_doe = pd.DataFrame()
    df_doe['Chris Doe'] = df_chrisdoe[['div_place','div_tot','time_net']].T
    df_doe['10th percentile'] = df_u49[['div_place','div_tot','time_net']].quantile(0.1)
    df_doe_T = df_doe.T
    df_doe_T['div_place'] = df_doe_T['div_place'].astype('Int64')
    df_doe_T['div_tot'] = df_doe_T['div_tot'].astype('Int64')
    df_doe = df_doe_T.T
    df_doe['Differential'] =  df_doe['Chris Doe'] - df_doe['10th percentile']
    return df_doe


def main():
    st.title("2006 Pike's Peak 10k Race")

    df_m = get_df_m()
    df_f = get_df_f()

    st.header('Q1: Results Distribution by Gender')
    

    hist_fig = get_histogram(df_m, df_f)
    st.plotly_chart(hist_fig, use_container_width=True, height=500, width=1100)
    
    df_summary = get_df_summary(df_m, df_f)
    st.dataframe(df_summary)





    st.header('Q2: Differential of Gun Time and Net Time')
    st.write('Conclusion: There looks to be three start times, and all of the best male and female racers were in the first starting group and have a gun/net time differential of less than 5 seconds.')

    fig_delta_m = get_seconds_delta_fig(df_m)
    fig_delta_f = get_seconds_delta_fig(df_f)
    col1_delta, col2_delta = st.columns(2)

    with col1_delta:
        st.pyplot(fig_delta_m)
    with col2_delta:
        st.pyplot(fig_delta_f)


    st.header('Q3: Chris Doe Results')

    df_doe = get_chris_doe_df(df_m)
    st.dataframe(df_doe)
    

    st.header('Q4: Results Distribution by Age Division')

    male_ridgeplot_fig = get_male_ridgeplot(df_m)
    female_ridgeplot_fig = get_female_ridgeplot(df_f)


    Divs = Divisions()
    df_m_summary = Divs.get_m_summary(df_m)
    df_f_summary = Divs.get_f_summary(df_f)
    m_heatmap_plt = get_mens_heatmap(Divs.df_m_summary, Divs.df_m_summary_sec, Divs.df_m_summary_str)
    f_heatmap_plt = get_womens_heatmap(Divs.df_f_summary, Divs.df_f_summary_sec, Divs.df_f_summary_str)


    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(male_ridgeplot_fig)
        st.pyplot(m_heatmap_plt)
    with col2:
        st.pyplot(female_ridgeplot_fig)
        st.pyplot(f_heatmap_plt)



if __name__ == '__main__':
    main()