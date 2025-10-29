import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean
from typing import Dict, List, Tuple, Any
import os
import logging

plt.style.use('seaborn-v0_8-whitegrid')

def _categorize_signal(row: pd.Series, median_x: float, median_y: float) -> str:
    """Categorizes a keyword based on its position relative to medians."""
    is_above_median_x = row['axis_x'] >= median_x
    is_above_median_y = row['axis_y'] >= median_y
    
    if is_above_median_x and is_above_median_y:
        return "Strong Signal"
    elif not is_above_median_x and is_above_median_y:
        return "Weak Signal"
    elif not is_above_median_x and not is_above_median_y:
        return "Latent Signal"
    else: # is_above_median_x and not is_above_median_y
        return "Well-known but not strong"

def generate_signal_map_data(
    df_x: pd.DataFrame, 
    df_y: pd.DataFrame, 
    periods: Dict[str, List[int]]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Generates the data for a signal map (KEM or KIM) for all periods.
    
    This function applies the core logic:
    1. Slices data by period.
    2. Calculates X-axis (mean) and Y-axis (gmean).
    3. Calculates medians and categorizes all keywords.
    
    Args:
        df_x (pd.DataFrame): Data for the X-axis (e.g., TF or DF).
        df_y (pd.DataFrame): Data for the Y-axis (e.g., DoV or DoD).
        periods (Dict[str, List[int]]): Dictionary defining the periods.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
            - A dictionary mapping period names to the resulting DataFrame.
            - A dictionary mapping period names to a short name (e.g., "P1").
    """
    all_period_data = {}
    period_short_names = {}

    for period_key, year_range in periods.items():
        period_name = f"{period_key} ({year_range[0]}-{year_range[1]-1})"
        period_short_names[period_name] = period_key.upper()
        logging.info(f"--- Processing Signal Map for {period_name} ---")
        
        years = list(range(year_range[0], year_range[1]))
        
        # Ensure columns are integers for slicing
        df_x.columns = df_x.columns.astype(int)
        df_y.columns = df_y.columns.astype(int)
        
        df_x_period = df_x[years]
        df_y_period = df_y[years]

        # Calculate axes
        avg_x = df_x_period.mean(axis=1)
        growth_y = gmean(df_y_period + 1e-10, axis=1)

        df_map = pd.DataFrame({ 'axis_x': avg_x, 'axis_y': growth_y })
        
        # Filter out keywords with no presence
        df_map = df_map[(df_map['axis_x'] > 0) & (df_map['axis_y'] > 1e-10)]

        if df_map.empty:
            logging.warning(f"No data for period {period_name}. Skipping.")
            continue

        # Calculate Medians and Categorize
        median_x = df_map['axis_x'].median()
        median_y = df_map['axis_y'].median()
        
        df_map['category'] = df_map.apply(
            lambda row: _categorize_signal(row, median_x, median_y), 
            axis=1
        )
        
        all_period_data[period_name] = df_map
        logging.info(f"Categorization complete for {period_name}.")
        
    return all_period_data, period_short_names

def plot_signal_map(
    df_map: pd.DataFrame, 
    title: str, 
    x_label: str, 
    y_label: str, 
    save_path: str
):
    """
    Generates and saves a standardized signal map plot.
    This follows the plotting logic from your scripts.
    """
    if df_map.empty:
        logging.warning(f"Skipping plot for '{title}' due to empty data.")
        return
        
    median_x = df_map['axis_x'].median()
    median_y = df_map['axis_y'].median()

    fig, ax = plt.subplots(figsize=(16, 12))
    palette = {
        "Strong Signal": "#2ca02c",
        "Weak Signal": "#ff7f0e",
        "Latent Signal": "#d62728",
        "Well-known but not strong": "#1f77b4"
    }
    
    sns.scatterplot(
        data=df_map, x='axis_x', y='axis_y', hue='category',
        palette=palette, s=60, alpha=0.6, ax=ax
    )
    
    # Add median lines
    ax.axhline(median_y, color='grey', linestyle='--', linewidth=1.2)
    ax.axvline(median_x, color='grey', linestyle='--', linewidth=1.2)

    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add Quadrant Labels
    ax.text(ax.get_xlim()[1], ax.get_ylim()[1], 'Strong Signals', ha='right', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], 'Weak Signals', ha='left', va='top', fontsize=14, fontweight='bold', color='black', alpha=0.7)
    ax.text(ax.get_xlim()[0], ax.get_ylim()[0], 'Latent Signals', ha='left', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)
    ax.text(ax.get_xlim()[1], ax.get_ylim()[0], 'Well-known but not strong', ha='right', va='bottom', fontsize=14, fontweight='bold', color='black', alpha=0.7)
    
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(title="Signal Category", title_fontsize='14', fontsize='12')
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    logging.info(f"Saved plot to: {save_path}")

def create_evolution_file(
    all_period_data: Dict[str, pd.DataFrame],
    period_short_names: Dict[str, str],
    category_col_prefix: str,
    save_path: str
):
    """Creates a consolidated file tracking signal evolution over time."""
    all_keywords_index = pd.Index([])
    for df_period in all_period_data.values():
        all_keywords_index = all_keywords_index.union(df_period.index)

    df_evolution = pd.DataFrame(index=all_keywords_index)

    for period_name, df_period in all_period_data.items():
        short_name = period_short_names[period_name]
        col_name = f'{category_col_prefix}_{short_name}'
        
        category_col = df_period[['category']].rename(columns={'category': col_name})
        df_evolution = df_evolution.join(category_col)

    df_evolution = df_evolution.fillna("Not Present")
    df_evolution.to_csv(save_path)
    logging.info(f"Saved evolution file to: {save_path}")