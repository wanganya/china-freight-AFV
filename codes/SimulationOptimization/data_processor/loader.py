"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Data loading module
Responsible for efficiently loading various types of data from the file system
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import time
import json
import pickle
import random
from datetime import datetime

from utils.logger import log_memory_usage, ProgressLogger


class DataLoader:
    """Data loading class, responsible for efficiently loading various types of data from the file system"""
    
    def __init__(self, config, logger, season='Winter', sample_ratio=1.0):
        """
        Initialize data loader
        
        Parameters:
            config (dict): Configuration dictionary
            logger (logging.Logger): Logger
            season (str): Season, options: 'Winter', 'Spring', 'Summer', 'Autumn'
        """
        self.config = config
        self.logger = logger
        self.season = season
        self.path_config = config['path_config']
        self.data_config = config['data_processing']
        self.chunk_size = self.data_config.get('chunk_size', 10000)
        self.memory_limit = self.data_config.get('memory_limit', 40) * 1024 * 1024 * 1024
        self.sample_ratio = sample_ratio
        if sample_ratio < 1.0:
            self.logger.warning(f"Using data sampling mode, processing only {sample_ratio*100:.1f}% of the data")
        
        self.start_date = config['simulation']['seasons'][season]['start_date']
        self.end_date = config['simulation']['seasons'][season]['end_date']
        
        self.output_dir = Path(self.path_config['output_directory'])
        self.cache_dir = self.output_dir / 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.trajectory_metadata = None
        
        self.data = {}
    
    def load_all_data(self):
        """
        Load all necessary data (except trajectory data)
        
        Returns:
            dict: Dictionary containing all loaded data
        """
        self.logger.info(f"Loading {self.season} season data...")
        log_memory_usage(self.logger, "Before data loading ")
        
        trajectory_file = self._get_trajectory_file()
        self.logger.info(f"Building trajectory index: {trajectory_file}")
        self.data['trajectory_metadata'] = self._build_trajectory_index(trajectory_file)
        log_memory_usage(self.logger, "After trajectory index building ")
        
        road_file = self.path_config['road_network_path']
        self.logger.info(f"Loading road network: {road_file}")
        self.data['road_network'] = self._load_road_network(road_file)
        log_memory_usage(self.logger, "After road network loading ")
        
        self.logger.info("Loading traffic volume data...")
        self.data['traffic_volume'] = self._load_traffic_volume_data()
        log_memory_usage(self.logger, "After traffic volume loading ")
        
        station_file = self.path_config['station_data_path']
        self.logger.info(f"Loading station data: {station_file}")
        self.data['stations'] = self._load_stations(station_file)
        log_memory_usage(self.logger, "After station data loading ")
        
        self._load_remaining_data()
        
        self.logger.info("All basic data loading completed")
        log_memory_usage(self.logger, "After data loading completed ")
        
        if 'trajectory_metadata' in self.data:
            self.trajectory_metadata = self.data['trajectory_metadata']
            self.logger.info(f"Trajectory metadata set as instance variable, {self.trajectory_metadata['total_count']} records")
        
        return self.data
    
    def _get_trajectory_file(self):
        """Get trajectory file path for the season"""
        season_file_map = {
            'Winter': 'Trajectory_Winter_20190114_20190120_sort.txt',
            'Spring': 'Trajectory_Spring_20190418_20190424_sort.txt',
            'Summer': 'Trajectory_Summer_20190716_20190722_sort.txt',
            'Autumn': 'Trajectory_Autumn_20191106_20191112_sort.txt'
        }
        
        return os.path.join(self.path_config['trajectory_path'], season_file_map[self.season])
    
    def _build_trajectory_index(self, file_path):
        """
        Build trajectory data index including file positions and basic metadata
        
        Parameters:
            file_path (str): Trajectory file path
            
        Returns:
            dict: Dictionary containing trajectory index information
        """
        index_file = self.cache_dir / f"trajectory_{self.season}_index.pkl"
        
        if os.path.exists(index_file):
            self.logger.info(f"Loading trajectory index from cache: {index_file}")
            with open(index_file, 'rb') as f:
                trajectory_index = pickle.load(f)
            self.logger.info(f"Trajectory index loading completed, {trajectory_index['total_count']} records")
            return trajectory_index
        
        self.logger.info(f"Building trajectory file index: {file_path}")
        
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1
        
        self.logger.info(f"Estimated trajectory file has {line_count} lines")
        
        trajectory_index = {
            'file_path': file_path,
            'header': None,
            'positions': [], 
            'vehicle_ids': [],  
            'dates': [],       
            'times': [],      
            'total_count': 0  
        }
        
        progress = ProgressLogger(self.logger, line_count, "Building trajectory index")
        progress.start()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            trajectory_index['header'] = header_line.split('\t')
            
            pos = f.tell()
            line = f.readline()
            line_count = 1
            
            while line:
                if line_count % 1000 == 0:
                    progress.update(1000)
                
                if self.sample_ratio < 1.0 and random.random() > self.sample_ratio:
                    pos = f.tell()
                    line = f.readline()
                    line_count += 1
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) >= 3:  
                    try:
                        vehicle_id = int(parts[0])
                        date = int(parts[1])
                        time_val = int(parts[2])
                        
                        trajectory_index['positions'].append(pos)
                        trajectory_index['vehicle_ids'].append(vehicle_id)
                        trajectory_index['dates'].append(date)
                        trajectory_index['times'].append(time_val)
                        trajectory_index['total_count'] += 1
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Skipping invalid line {line_count}: {e}")
                
                pos = f.tell()
                line = f.readline()
                line_count += 1
        
        progress.finish()
        
        with open(index_file, 'wb') as f:
            pickle.dump(trajectory_index, f)
        
        self.logger.info(f"Trajectory index built, containing {trajectory_index['total_count']} records, saved to {index_file}")
        return trajectory_index

    def load_trajectory_batch(self, start_idx, batch_size, filter_func=None):
        """
        Load a batch of trajectory data
        
        Parameters:
            start_idx (int): Starting index
            batch_size (int): Batch size
            filter_func (callable, optional): Filter function
            
        Returns:
            pandas.DataFrame: Loaded trajectory data
        """
        total_records = 0
        filtered_records = 0
        
        if not hasattr(self, 'trajectory_metadata') or self.trajectory_metadata is None:
            self.logger.info("Attempting to load trajectory index from cache")
            try:
                season = self.season
                cache_file = self.cache_dir / f"trajectory_{season}_index.pkl"
                
                if os.path.exists(cache_file):
                    self.logger.info(f"Loading trajectory index from cache: {cache_file}")
                    with open(cache_file, 'rb') as f:
                        self.trajectory_metadata = pickle.load(f)
                    self.logger.info(f"Trajectory metadata loaded from cache, records: {self.trajectory_metadata['total_count']}")
                else:
                    self.logger.error(f"Trajectory metadata cache file not found: {cache_file}")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Failed to load trajectory metadata: {e}")
                return pd.DataFrame()
        
        metadata = self.trajectory_metadata
        self.logger.debug(f"Loading trajectory batch, start_idx: {start_idx}, batch_size: {batch_size}")
        
        file_path = metadata['file_path']
        header = metadata['header']
        
        if start_idx >= metadata['total_count']:
            return pd.DataFrame(columns=header)
        
        end_idx = min(start_idx + batch_size, metadata['total_count'])
        
        positions = metadata['positions'][start_idx:end_idx]
        if not positions:
            return pd.DataFrame(columns=header)
        
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for pos in positions:
                f.seek(pos)
                line = f.readline().strip()
                parts = line.split('\t')
                
                if len(parts) == len(header):
                    record = dict(zip(header, parts))
                    total_records += 1
                    
                    if filter_func is None or filter_func(record):
                        records.append(record)
                    else:
                        filtered_records += 1
        
        df = pd.DataFrame(records)
        
        type_converters = {
            'VehicleID': lambda x: int(x) if x.isdigit() else 0,
            'Date': lambda x: int(x) if x.isdigit() else 0,
            'Time': lambda x: int(x) if x.isdigit() else 0,
            '核定载质量': lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0.0
        }
        
        for col, converter in type_converters.items():
            if col in df.columns:
                df[col] = df[col].apply(converter)
        
        if 'Path' in df.columns:
            df['PathList'] = df['Path'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            df['PathLength'] = df['PathList'].apply(len)
        
        self.logger.info(f"Batch reading completed: total {total_records}, filtered {filtered_records}, kept {len(records)}")
        
        return df

    def _load_road_network(self, file_path):
        """
        Load road network data
        
        Parameters:
            file_path (str): Road network file path (GeoPackage)
        
        Returns:
            geopandas.GeoDataFrame: Road network data
        """
        cache_file = self.cache_dir / "road_network_cache.parquet"
        
        if cache_file.exists() and self.data_config.get('cache_processed_data', True):
            try:
                self.logger.info(f"Loading road network data from cache: {cache_file}")
                gdf = gpd.read_parquet(cache_file)
                self.logger.info(f"Road network data loading completed, {len(gdf)} records")
                return gdf
            except Exception as e:
                self.logger.warning(f"Failed to load road network data from cache: {e}")
        
        self.logger.info(f"Loading road network data from GPKG: {file_path}")
        gdf = gpd.read_file(file_path)
        
        gdf['ID'] = gdf['ID'].astype(str)
        
        if self.data_config.get('cache_processed_data', True):
            try:
                self.logger.info(f"Caching road network data to: {cache_file}")
                gdf.to_parquet(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to cache road network data: {e}")
        
        self.logger.info(f"Road network data loading completed, {len(gdf)} records")
        return gdf
    
    def _load_traffic_volume_data(self):
        """
        Load traffic volume data
        
        Returns:
            dict: Dictionary with dates as keys and DataFrames as values
        """
        volume_dir = self.path_config['traffic_volume_path']
        
        start = int(self.start_date)
        end = int(self.end_date)
        dates = list(range(start, end + 1))
        
        traffic_data = {}
        
        for date in dates:
            file_path = os.path.join(volume_dir, f"{date}_update.xls")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Traffic volume data file not found: {file_path}")
                continue
            
            self.logger.info(f"Loading traffic volume data: {file_path}")
            df = pd.read_excel(file_path)
            
            if '路线简码' not in df.columns or '小时' not in df.columns or '拥挤度' not in df.columns:
                self.logger.warning(f"Traffic volume data file format incorrect: {file_path}")
                continue
            
            traffic_data[date] = df
            self.logger.info(f"Traffic volume data for date {date} loaded, {len(df)} records")
        
        if not traffic_data:
            self.logger.error("Failed to load any traffic volume data!")
        
        return traffic_data
    
    def _load_stations(self, file_path):
        """
        Load gas station data
        
        Parameters:
            file_path (str): Gas station Shapefile file path
        
        Returns:
            geopandas.GeoDataFrame: Gas station data
        """
        cache_file = self.cache_dir / "stations_cache.parquet"
        
        if cache_file.exists() and self.data_config.get('cache_processed_data', True):
            try:
                self.logger.info(f"Loading station data from cache: {cache_file}")
                gdf = gpd.read_parquet(cache_file)
                self.logger.info(f"Station data loading completed, {len(gdf)} records")
                return gdf
            except Exception as e:
                self.logger.warning(f"Failed to load station data from cache: {e}")
        
        self.logger.info(f"Loading station data from Shapefile: {file_path}")
        gdf = gpd.read_file(file_path)
        
        gdf['LinkRoad'] = gdf['LinkRoad'].astype(str)
        
        if self.data_config.get('cache_processed_data', True):
            try:
                self.logger.info(f"Caching station data to: {cache_file}")
                gdf.to_parquet(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to cache station data: {e}")
        
        self.logger.info(f"Station data loading completed, {len(gdf)} records")
        return gdf
    
    def _load_remaining_data(self):
        """Load other necessary data (temperature, vehicle types, electricity prices, etc.)"""
        temp_file = self.path_config['temperature_data_path']
        self.logger.info(f"Loading temperature data: {temp_file}")
        self.data['temperature'] = pd.read_csv(temp_file, sep='\t', encoding='utf-8')
        
        vehicle_file = self.path_config['vehicle_data_path']
        self.logger.info(f"Loading vehicle type data: {vehicle_file}")
        self.data['vehicles'] = pd.read_csv(vehicle_file, sep='\t', encoding='utf-8')
        
        cs_file = self.path_config['cs_data_path']
        self.logger.info(f"Loading charging station data: {cs_file}")
        self.data['charging_stations'] = pd.read_csv(cs_file, sep='\t', encoding='utf-8')
        
        bss_file = self.path_config['bss_data_path']
        self.logger.info(f"Loading battery swap station data: {bss_file}")
        self.data['battery_swap_stations'] = pd.read_csv(bss_file, sep='\t', encoding='utf-8')
        
        hrs_file = self.path_config['hrs_data_path']
        self.logger.info(f"Loading hydrogen station data: {hrs_file}")
        self.data['hydrogen_stations'] = pd.read_csv(hrs_file, sep='\t', encoding='utf-8')
        
        cp_file = self.path_config['charging_post_data_path']
        self.logger.info(f"Loading charging post data: {cp_file}")
        self.data['charging_posts'] = pd.read_csv(cp_file, sep='\t', encoding='utf-8')
        
        if self.season == 'Summer':
            price_file = self.path_config['electricity_price_summer_path']
            self.logger.info(f"Loading summer electricity price data: {price_file}")
            self.data['electricity_price'] = pd.read_csv(price_file, sep='\t', encoding='utf-8')
        else:
            price_file = self.path_config['electricity_price_unsummer_path']
            self.logger.info(f"Loading non-summer electricity price data: {price_file}")
            self.data['electricity_price'] = pd.read_csv(price_file, sep='\t', encoding='utf-8')
        
        h2_price_file = self.path_config['hydrogen_price_path']
        self.logger.info(f"Loading hydrogen price data: {h2_price_file}")
        self.data['hydrogen_price'] = pd.read_csv(h2_price_file, sep='\t', encoding='utf-8')
        
        params_file = self.path_config['parameters_path']
        self.logger.info(f"Loading parameters: {params_file}")
        self.data['parameters'] = pd.read_csv(params_file, sep='\t', encoding='utf-8')
        
        speed_file = self.path_config['free_speed_path']
        self.logger.info(f"Loading free flow speed data: {speed_file}")
        self.data['free_speed'] = pd.read_csv(speed_file, sep='\t', encoding='utf-8')
        
        vehicle_ghg_file = self.path_config['vehicle_ghg_path']
        self.logger.info(f"Loading vehicle GHG emission data: {vehicle_ghg_file}")
        self.data['vehicle_ghg'] = pd.read_csv(vehicle_ghg_file, sep='\t', encoding='utf-8')
        
        battery_ghg_file = self.path_config['battery_ghg_path']
        self.logger.info(f"Loading battery GHG emission data: {battery_ghg_file}")
        self.data['battery_ghg'] = pd.read_csv(battery_ghg_file, sep='\t', encoding='utf-8')
        
        province_city_file = self.path_config['province_city_code_path']
        self.logger.info(f"Loading province-city code mapping data: {province_city_file}")
        self.data['province_city_code'] = pd.read_csv(province_city_file, sep='\t', encoding='utf-8')
        
        if 'distance_correction_file' in self.path_config:
            distance_correction_file = self.path_config['distance_correction_file']
            self.logger.info(f"Loading distance correction factor data: {distance_correction_file}")
            try:
                import pickle
                with open(distance_correction_file, 'rb') as f:
                    correction_data = pickle.load(f)
                self.data['distance_correction'] = correction_data
                self.logger.info(f"Distance correction factor data loaded, {len(correction_data['city_factors'])} cities")
            except Exception as e:
                self.logger.warning(f"Failed to load distance correction factor data: {e}")
                self.data['distance_correction'] = None