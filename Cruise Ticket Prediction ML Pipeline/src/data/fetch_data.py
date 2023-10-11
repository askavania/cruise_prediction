import sqlite3
import pandas as pd

def fetch_pre_purchase_data():
    """
    Fetches the pre-purchase data from the SQLite database and returns it as a DataFrame.
    """
    conn_pre = sqlite3.connect('data/cruise_pre.db')
    pre_purchase_data = pd.read_sql_query('SELECT * FROM cruise_pre', conn_pre)
    pre_df = pd.DataFrame(pre_purchase_data)
    return pre_df

def fetch_post_trip_data():
    """
    Fetches the post-trip data from the SQLite database and returns it as a DataFrame.
    """
    conn_post = sqlite3.connect('data/cruise_post.db')
    post_trip_data = pd.read_sql_query('SELECT * FROM cruise_post', conn_post)
    post_df = pd.DataFrame(post_trip_data)
    return post_df
