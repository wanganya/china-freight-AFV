"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Result visualization module
Generates various charts and visualization results for the simulation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging


class ResultVisualizer:
    """Result visualizer for generating various charts and visualization results"""
    
    def __init__(self, config, logger=None):
        """
        Initialize result visualizer
        
        Parameters:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.output_dir = Path(config['path_config']['output_directory'])
        self.figure_dir = self.output_dir / 'figures'
        
        os.makedirs(self.figure_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette('Set2')
    
    def visualize_optimization_history(self, history_file=None):
        """
        Visualize optimization history
        
        Parameters:
            history_file (str, optional): Optimization history file path
        """
        if history_file is None:
            history_file = self.output_dir / "optimization_history.csv"
        

        if not os.path.exists(history_file):
            self.logger.error(f"Optimization history file not found: {history_file}")
            return
        
        try:
            history_df = pd.read_csv(history_file)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Optimization History', fontsize=16)
            ax1 = axes[0, 0]
            ax1.plot(history_df['iteration'], history_df['objective'], 'b-', label='Objective')
            accepted_df = history_df[history_df['accepted']]
            ax1.plot(accepted_df['iteration'], accepted_df['objective'], 'ro', markersize=3, label='Accepted')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Objective Value vs. Iteration')
            ax1.legend()
            
            ax2 = axes[0, 1]
            ax2.plot(history_df['iteration'], history_df['temperature'], 'g-')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Temperature')
            ax2.set_title('Temperature vs. Iteration')
            ax2.set_yscale('log')
            
            ax3 = axes[1, 0]
            ax3.plot(history_df['iteration'], history_df['cost'], 'r-', label='Cost')
            ax3.plot(history_df['iteration'], history_df['travel_time'], 'g-', label='Travel Time')
            ax3.plot(history_df['iteration'], history_df['ghg'], 'b-', label='GHG')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Objective Components')
            ax3.set_title('Objective Components vs. Iteration')
            ax3.legend()
            
            ax4 = axes[1, 1]
            window_size = 50
            accepted_ratio = history_df['accepted'].rolling(window=window_size).mean()
            ax4.plot(history_df['iteration'], accepted_ratio, 'purple')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Acceptance Ratio')
            ax4.set_title(f'Acceptance Ratio (Window Size: {window_size}) vs. Iteration')
            ax4.set_ylim(0, 1)
            
            fig.tight_layout()
            fig_path = self.figure_dir / "optimization_history.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Optimization history chart saved to {fig_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to visualize optimization history: {e}")
    
    def visualize_station_distribution(self, station_file=None):
        """
        Visualize station distribution
        
        Parameters:
            station_file (str, optional): Station configuration file path
        """
        if station_file is None:
            station_file = self.output_dir / "02Station.txt"
        
        if not os.path.exists(station_file):
            self.logger.error(f"Station configuration file not found: {station_file}")
            return
        
        try:
            station_df = pd.read_csv(station_file, sep='\t')
            cs_count = station_df['CS'].sum()
            bss_count = station_df['BSS'].sum()
            hrs_count = (station_df.filter(like='HRS_').sum(axis=1) > 0).sum()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = ['Charging Stations', 'Battery Swap Stations', 'Hydrogen Refueling Stations']
            sizes = [cs_count, bss_count, hrs_count]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.colors)
            ax.axis('equal')
            ax.set_title('Distribution of Station Types')
            
            fig_path = self.figure_dir / "station_distribution.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Station distribution chart saved to {fig_path}")
            total_fcp = station_df['FCP'].sum()
            total_scp = station_df['SCP'].sum()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = ['Fast Charging Posts', 'Slow Charging Posts']
            y = [total_fcp, total_scp]
            ax.bar(x, y, color=self.colors[:2])
            for i, v in enumerate(y):
                ax.text(i, v + 0.1, str(v), ha='center')
            ax.set_xlabel('Type')
            ax.set_ylabel('Count')
            ax.set_title('Charging Post Distribution')
            
            fig_path = self.figure_dir / "charging_post_distribution.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Charging post distribution chart saved to {fig_path}")
            bev_columns = [col for col in station_df.columns if col.startswith('BEV_')]
            bev_counts = []

            has_battery_data = False
            for col in bev_columns:
                count = station_df[col].sum()
                if count > 0:
                    has_battery_data = True
                    bev_counts.append((col, count))
            if has_battery_data and bev_counts:
                fig, ax = plt.subplots(figsize=(12, 6))
                x = [item[0] for item in bev_counts]
                y = [item[1] for item in bev_counts]
                ax.bar(x, y, color=self.colors[2:2+len(bev_counts)])
                for i, v in enumerate(y):
                    ax.text(i, v + 0.1, str(v), ha='center')
                ax.set_xlabel('Battery Type')
                ax.set_ylabel('Count')
                ax.set_title('Spare Battery Distribution (Based on 7-day Maximum Demand)')
                
                fig_path = self.figure_dir / "spare_battery_distribution.png"
                fig.savefig(fig_path, dpi=300)
                plt.close(fig)
                
                self.logger.info(f"Spare battery distribution chart saved to {fig_path}")
            else:
                self.logger.info("Spare battery quantities determined based on simulation demand")
        
        except Exception as e:
            self.logger.error(f"Failed to visualize station distribution: {e}")
    
    def visualize_vehicle_distribution(self, vehicle_file=None):
        """
        Visualize vehicle distribution
        
        Parameters:
            vehicle_file (str, optional): Vehicle allocation file path
        """
        if vehicle_file is None:
            vehicle_file = self.output_dir / "01ElectrifiedVehicle.txt"
        
        if not os.path.exists(vehicle_file):
            self.logger.error(f"Vehicle allocation file not found: {vehicle_file}")
            return
        
        try:
            vehicle_df = pd.read_csv(vehicle_file, sep='\t')
            vehicle_df['Category'] = vehicle_df['Type'].apply(lambda x: 'BEV' if x.startswith('BEV_') else 'HFCV')
            category_counts = vehicle_df['Category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = category_counts.index
            sizes = category_counts.values
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.colors[:len(labels)])
            ax.axis('equal')
            ax.set_title('Distribution of Vehicle Categories')
            
            fig_path = self.figure_dir / "vehicle_category_distribution.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Vehicle category distribution chart saved to {fig_path}")
            type_counts = vehicle_df['Type'].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            type_counts.plot(kind='bar', ax=ax, color=self.colors[:len(type_counts)])
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
            ax.set_title('Vehicle Type Distribution')
            
            fig_path = self.figure_dir / "vehicle_type_distribution.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Vehicle type distribution chart saved to {fig_path}")
            actual_vehicle_counts = vehicle_df['ActualVehicleID'].nunique()
            
            stats_file = self.figure_dir / "vehicle_statistics.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Total Trajectories: {len(vehicle_df)}\n")
                f.write(f"Total Physical Vehicles: {actual_vehicle_counts}\n")
                f.write(f"BEV Count: {category_counts.get('BEV', 0)}\n")
                f.write(f"HFCV Count: {category_counts.get('HFCV', 0)}\n")
                f.write("\nVehicle Type Distribution:\n")
                for vehicle_type, count in type_counts.items():
                    f.write(f"{vehicle_type}: {count}\n")
            
            self.logger.info(f"Vehicle statistics saved to {stats_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to visualize vehicle distribution: {e}")
    
    def visualize_objective_values(self, objective_file=None):
        """
        Visualize objective function values
        
        Parameters:
            objective_file (str, optional): Objective function values file path
        """
        if objective_file is None:
            objective_file = self.output_dir / "04Objective.txt"
        
        if not os.path.exists(objective_file):
            self.logger.error(f"Objective function values file not found: {objective_file}")
            return
        
        try:
            objective_df = pd.read_csv(objective_file, sep='\t')
            total_cost = objective_df['TotalCost'].iloc[0]
            total_travel_time = objective_df['TotalTravelTime'].iloc[0]
            total_ghg = objective_df['TotalGHG'].iloc[0]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = ['Total Cost (Yuan)', 'Total Travel Time (s)', 'Total GHG Emission (kg)']
            y = [total_cost, total_travel_time, total_ghg]
            bars = ax.bar(x, y, color=self.colors[:len(x)])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
            
            ax.set_title('Objective Function Values')
            
            fig_path = self.figure_dir / "objective_values.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Objective function values chart saved to {fig_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to visualize objective function values: {e}")
    
    def visualize_driving_statistics(self, driving_stats_file=None):
        """
        Visualize driving statistics
        
        Parameters:
            driving_stats_file (str, optional): Driving statistics file path
        """
        if driving_stats_file is None:
            driving_stats_file = self.output_dir / "07DrivingTimeStatistics.txt"
        
        if not os.path.exists(driving_stats_file):
            self.logger.error(f"Driving statistics file not found: {driving_stats_file}")
            return
        
        try:
            stats_df = pd.read_csv(driving_stats_file, sep='\t')
            
            if 'VehicleType' in stats_df.columns:
                stats_df['Category'] = stats_df['VehicleType'].apply(lambda x: 'BEV' if x.startswith('BEV_') else 'HFCV')
            else:
                self.logger.warning("VehicleType column missing, cannot categorize vehicles")
                stats_df['Category'] = 'Unknown'
            category_avg_driving = stats_df.groupby('Category')['TotalDrivingTime'].mean()
            category_avg_rest = stats_df.groupby('Category')['TotalRestTime'].mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(category_avg_driving.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, category_avg_driving.values / 3600, width, label='Avg. Driving Time (h)')
            bars2 = ax.bar(x + width/2, category_avg_rest.values / 3600, width, label='Avg. Rest Time (h)')
            
            ax.set_xlabel('Vehicle Category')
            ax.set_ylabel('Time (hours)')
            ax.set_title('Average Driving and Rest Time by Vehicle Category')
            ax.set_xticks(x)
            ax.set_xticklabels(category_avg_driving.index)
            ax.legend()
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom')
            
            fig_path = self.figure_dir / "driving_rest_time.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Driving and rest time chart saved to {fig_path}")
            category_avg_refill_count = stats_df.groupby('Category')['EnergyRefillCount'].mean()
            category_avg_refill_amount = stats_df.groupby('Category')['EnergyRefillAmount'].mean()
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Vehicle Category')
            ax1.set_ylabel('Average Refill Count', color=color)
            bars1 = ax1.bar(x - width/2, category_avg_refill_count.values, width, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Average Refill Amount', color=color)
            bars2 = ax2.bar(x + width/2, category_avg_refill_amount.values, width, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Average Energy Refill Count and Amount by Vehicle Category')
            ax1.set_xticks(x)
            ax1.set_xticklabels(category_avg_refill_count.index)
            
            fig.tight_layout()
            fig.legend([bars1, bars2], ['Avg. Refill Count', 'Avg. Refill Amount'], loc='upper right')
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', color='black')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', color='black')
            
            fig_path = self.figure_dir / "energy_refill.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"Energy refill chart saved to {fig_path}")
            category_avg_ghg = stats_df.groupby('Category')['GHGEmission'].mean()
            category_total_ghg = stats_df.groupby('Category')['GHGEmission'].sum()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.pie(category_avg_ghg.values, labels=category_avg_ghg.index, autopct='%1.1f%%',
                   startangle=90, colors=self.colors[:len(category_avg_ghg)])
            ax1.axis('equal')
            ax1.set_title('Average GHG Emission by Vehicle Category')
            
            ax2.pie(category_total_ghg.values, labels=category_total_ghg.index, autopct='%1.1f%%',
                   startangle=90, colors=self.colors[:len(category_total_ghg)])
            ax2.axis('equal')
            ax2.set_title('Total GHG Emission by Vehicle Category')
            
            fig_path = self.figure_dir / "ghg_emission.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"GHG emission chart saved to {fig_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to visualize driving statistics: {e}")