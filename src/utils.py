# utility functions

from typing import Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd


def create_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create timestamp column
    Engineer new time-related features, including:
    - hour
    - day of week
    - time difference since last transaction grouped by credit card number (cc_num)
    - the above expressed as minutes and seconds
    Returns:
    pd.DataFrame
    """
    df["datetime"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()
    df = df.sort_values(by=["cc_num", "datetime"])
    df["time_since_last"] = df.groupby("cc_num")["datetime"].diff()
    df["time_since_last_seconds"] = df["time_since_last"].dt.total_seconds()
    df["time_since_last_minutes"] = df["time_since_last_seconds"] / 60
    return df


def plot_comparison_histogram(
    df_normal: pd.DataFrame,
    df_fraud: pd.DataFrame,
    col_name: str,
    normal_bins: int = 20,
    fraud_bins: int = 20,
    height: float = 10,
    width: float = 7,
    x_label: str = "x-axis",
    y_label: str = "y-axis",
    max_x: float = None,
) -> Tuple[Figure, List[Axes]]:
    """
    Generate a fraud vs normal side-by-side comparison plot of a numerical feature.
    Returns a matplotlib.figure.Figure object and a list of matplotlib.axes.Axes
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(df_normal[col_name], bins=normal_bins)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_title("Normal")
    axs[1].hist(df_fraud[col_name], bins=fraud_bins, color="orange")
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_title("Fraud")
    if max_x:
        axs[0].set_xlim(left=0, right=max_x)
        axs[1].set_xlim(left=0, right=max_x)
    return fig, axs
