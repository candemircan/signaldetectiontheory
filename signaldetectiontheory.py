import pandas as pd
import itertools 
from scipy.stats import norm


def compute_sdt(data,sig,sig_val,no_sig_val,response,group = [], csv = ""): 
    """
    Computes signal detection theory measures

    Parameters
    ----------
    data : pd.DataFrame
        Data you want to analyse. If you have data from multiple participants you want to analyse, make sure you append the dataframes.
        
    sig : str
        Column of data which contains the signal information
        
    sig_val : 
        value for present signal in data

    no_sig_val:
        value for no-signal in data
    
    response: 
        participant's response
        
    group: list, optional 
        a list of factors to group the statistics such as participant no or difficulty level.
        
    csv: string, optional 
        path of the csv output of the dataframe containing the SDT statistics.
    
    Returns
    -------
    pd.DataFrame
        A dataframe of SDT statistics

    """
    main_column_names = ["hit rate","false alarm rate","miss rate","correct rejection rate", "sensitivity (d')","bias (c)"]
    all_column_names = group + main_column_names
    sdt_stats = pd.DataFrame(columns = all_column_names)
    group_expand = []
    
    for factor in group:
        group_expand.append(list(data[factor].unique()))
        
    group_combo = list(itertools.product(*group_expand))
    for combo,row_c in zip(group_combo,range(len(group_combo))):

        stats_to_add =[]

        sdt_stats.loc[row_c,group]=combo

        hit_rate = sum((data[sig] == sig_val) & (data[response] == sig_val) & (data[group]==combo).all(axis="columns")) / \
            sum((data[sig] == sig_val) & (data[group]==combo).all(axis="columns"))
        
        false_alarm_rate = sum((data[sig] == no_sig_val) & (data[response] == sig_val) & (data[group]==combo).all(axis="columns")) / \
            sum((data[sig] == no_sig_val) & (data[group]==combo).all(axis="columns"))
        
        miss_rate = 1-hit_rate
        
        correct_rejection_rate = 1-false_alarm_rate

        sensitivity = norm.ppf(hit_rate)-norm.ppf(false_alarm_rate)

        bias = .5 +(hit_rate-false_alarm_rate)/2
        
        stats_to_add.extend([hit_rate,false_alarm_rate,miss_rate,correct_rejection_rate,sensitivity,bias])

        sdt_stats.loc[row_c,main_column_names] = stats_to_add
    
    if csv:
        sdt_stats.to_csv(csv,index=False)

    return sdt_stats
