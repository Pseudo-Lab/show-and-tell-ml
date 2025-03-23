import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from PIL import Image
import time

def format_ticks(x, pos):
    if x >= 1000:
        return f'{int(x / 1000)}k'
    return str(int(x))

def generate_scatter_plot(current_time, sampled_df, time_counts, erangel_img, max_time, output_dir, min_time):
    filtered_df = sampled_df[(sampled_df['time'] >= current_time - 30) & (sampled_df['time'] <= current_time)]
    
    current_time_data_count = len(filtered_df[filtered_df['time'] == current_time])
    alpha_weight = time_counts.max() / current_time_data_count if current_time_data_count > 0 else 1
    alpha_weight = np.clip(alpha_weight, 1, 10)
    
    plt.figure(figsize=(7, 6))

    plt.imshow(erangel_img, aspect='auto', extent=[0, 800000, 800000, 0], alpha=0.3)

    if not filtered_df.empty:
        scatter = plt.scatter(filtered_df['victim_position_x'], filtered_df['victim_position_y'],
                              c=filtered_df['time'], cmap='plasma', alpha=0.05 * alpha_weight,
                              norm=mcolors.Normalize(vmin=0, vmax=max_time), s=5)
    else:
        scatter = plt.scatter([], [], c=[], cmap='plasma', alpha=0.05,
                              norm=mcolors.Normalize(vmin=0, vmax=max_time), s=5)

    plt.xlabel('Victim Position X')
    plt.ylabel('Victim Position Y')
    plt.title(f'Scatter Plot of Victim Positions: Time = {current_time}')

    plt.xlim(0, 800000)
    plt.ylim(0, 800000)

    plt.gca().invert_yaxis()

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))

    cbar = plt.colorbar(scatter)
    cbar.set_label('Time')
    cbar.set_ticks([min_time, max_time])
    cbar.ax.axhline(y=current_time, color='tab:red', linewidth=1, linestyle='-')

    filename = os.path.join(output_dir, f"scatter_plot_{current_time:04d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(generate_scatter_plot, 
                              range(min_time, max_time + 1),
                              [sampled_df] * (max_time - min_time + 1),
                              [time_counts] * (max_time - min_time + 1),
                              [erangel_img] * (max_time - min_time + 1),
                              [max_time] * (max_time - min_time + 1),
                              [output_dir] * (max_time - min_time + 1),
                              [min_time] * (max_time - min_time + 1)),
                  desc="Generating scatter plots", unit="frame"))

    print(f"Scatter plots saved in the '{output_dir}' directory.")

if __name__ == '__main__':
    data_path = "../../../Downloads/archive"
    
    data_stats_files = glob(os.path.join(data_path, 'aggregate/*.csv'))
    data_deaths_files = glob(os.path.join(data_path, 'deaths/*.csv'))
    
    for file in data_stats_files + data_deaths_files:
        print(os.path.basename(file))
        
    erangel_img_pil = Image.open(os.path.join(data_path, 'erangel.jpg'))
    erangel_img_resized = erangel_img_pil.resize((256, 256))  # Resize to 1024x1024
    erangel_img = np.array(erangel_img_resized)
    
    start_time = time.perf_counter()
    df = pd.read_csv(data_stats_files[0])
    end_time = time.perf_counter()
    print(f"Time taken to read the agg_match_stats file: {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    death_df = pd.read_csv(data_deaths_files[0])
    end_time = time.perf_counter()
    print(f"Time taken to read the kill_match_stats file: {end_time - start_time:.6f} seconds")

    df = df.dropna()
    death_df = death_df.dropna()

    erangel_death_df = death_df[death_df['map'] == 'ERANGEL']

    max_time = erangel_death_df['time'].max()
    min_time = erangel_death_df['time'].min()

    unique_match_ids = df['match_id'].unique()
    print(f"Number of unique match_id values: {len(unique_match_ids)}")
    
    death_unique_match_ids = erangel_death_df['match_id'].unique()
    print(f"Number of ERANGEL unique match_id values: {len(death_unique_match_ids)}")
    
    valid_match_ids = np.intersect1d(unique_match_ids, death_unique_match_ids)
    print(f"Valid match_ids: {len(valid_match_ids)}")
    
    output_dir = "scatter_plots"
    
    if os.path.exists(output_dir):
        for file_name in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file_name))
    else:
        os.makedirs(output_dir)
        
    sampled_df = erangel_death_df.sample(frac=0.01, random_state=42)
    
    min_time = sampled_df['time'].min()
    max_time = sampled_df['time'].max()
    
    time_counts = sampled_df['time'].value_counts()

    main()
