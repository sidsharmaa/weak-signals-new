import pandas as pd
import os
import logging
from src.config import config, get_project_root
from src.analysis.signal_maps import (
    generate_signal_map_data, 
    plot_signal_map, 
    create_evolution_file
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Orchestrates the KIM analysis:
    1. Loads configuration and data (DF and DoD).s
    2. Calls the generalized signal map generator.
    3. Saves all artifacts (plots, data, evolution file).
    """
    logging.info("Starting KIM Analysis...")
    
    # Load all paths from the central config
    PROJECT_ROOT = get_project_root()
    df_path = os.path.join(PROJECT_ROOT, config.metrics.df_path)
    dod_path = os.path.join(PROJECT_ROOT, config.metrics.dod_path)
    reports_dir = os.path.join(PROJECT_ROOT, config.analysis.reports_dir)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Data
    try:
        df_df = pd.read_csv(df_path, index_col=0)
        df_dod = pd.read_csv(dod_path, index_col=0)
        logging.info("Successfully loaded DF and DoD data.")
    except FileNotFoundError as e:
        logging.error(f"Input file not found. {e}. Have you run the metrics scripts?")
        return

    # 2. Call Abstracted Logic
    all_period_data, period_short_names = generate_signal_map_data(
        df_x=df_df,
        df_y=df_dod,
        periods=config.analysis.periods
    )

    # 3. Save Artifacts
    for period_name, df_map in all_period_data.items():
        # Save data
        data_filename = f"kim_data_{period_name.replace(' ', '_')}.csv"
        data_path = os.path.join(reports_dir, data_filename)
        df_map.to_csv(data_path)
        logging.info(f"Saved KIM data for {period_name} to {data_path}")

        # Save plot
        plot_title = f"Keyword Issue Map (KIM) for {period_name}"
        plot_filename = f"KIM_Plot_{period_name.replace(' ', '_')}.png"
        plot_path = os.path.join(reports_dir, plot_filename)
        plot_signal_map(
            df_map=df_map,
            title=plot_title,
            x_label="Average Document Frequency (Diffusion)",
            y_label="DoD Growth Rate (Emergence)",
            save_path=plot_path
        )

    # 4. Save Consolidated Evolution File
    evolution_path = os.path.join(PROJECT_ROOT, config.analysis.kim_evolution_path)
    create_evolution_file(
        all_period_data=all_period_data,
        period_short_names=period_short_names,
        category_col_prefix="kim_category", 
        save_path=evolution_path
    )
    
    logging.info("KIM Analysis Complete.")

if __name__ == "__main__":
    main()