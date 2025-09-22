"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Data preprocessing module
Responsible for data cleaning, conversion and preprocessing operations
"""

import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json
import multiprocessing as mp
from functools import partial

from utils.logger import log_memory_usage, ProgressLogger


class DataPreprocessor:
    
    def __init__(self, config, logger, n_workers=None):
        """
        Initialize data preprocessor
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger): Logger instance
            n_workers (int, optional): Number of parallel processing workers
        """
        self.config = config
        self.logger = logger
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        self.output_dir = Path(config['path_config']['output_directory'])
        self.cache_dir = self.output_dir / 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.data_config = config['data_processing']
        self.clean_nan = self.data_config.get('clean_nan_records', True)
    
    def process(self, data):
        """
        Process all input data (except trajectory data)
        
        Args:
            data (dict): Dictionary containing all input data
        
        Returns:
            dict: Processed data
        """
        start_time = time.time()
        self.logger.info("Starting data preprocessing...")
        log_memory_usage(self.logger, "Before preprocessing")
        
        processed = {}
        
        if 'trajectory_metadata' in data:
            processed['trajectory_metadata'] = data['trajectory_metadata']
        else:
            self.logger.warning("Trajectory metadata not found")
        
        processed['road_network'] = self._process_road_network(data['road_network'])
        log_memory_usage(self.logger, "After road network processing")
        
        processed['road_to_city'] = self._build_road_to_city_map(data['road_network'])
        
        processed['traffic_mapping'] = self._process_traffic_volume(data['traffic_volume'])
        
        processed['stations'] = self._process_stations(data['stations'])
        
        processed['temperature_mapping'] = self._process_temperature(data['temperature'])
        
        processed['vehicles'] = self._process_vehicles(data['vehicles'])
        
        processed['charging_stations'] = self._process_charging_stations(data['charging_stations'])
        
        processed['battery_swap_stations'] = self._process_battery_swap_stations(data['battery_swap_stations'])
        
        processed['hydrogen_stations'] = self._process_hydrogen_stations(data['hydrogen_stations'])
        
        processed['charging_posts'] = self._process_charging_posts(data['charging_posts'])
        
        processed['electricity_price'] = self._process_electricity_price(data['electricity_price'])
        
        processed['hydrogen_price'] = self._process_hydrogen_price(data['hydrogen_price'])
        
        processed['parameters'] = self._process_parameters(data['parameters'])
        
        processed['free_speed'] = self._process_free_speed(data['free_speed'])
        
        processed['vehicle_ghg'] = self._process_vehicle_ghg(data['vehicle_ghg'])
        
        processed['battery_ghg'] = self._process_battery_ghg(data['battery_ghg'])
        
        processed['province_city_map'] = self._process_province_city_code(data['province_city_code'])
        
        if 'distance_correction' in data and data['distance_correction'] is not None:
            processed['distance_correction'] = self._process_distance_correction(data['distance_correction'])
        else:
            self.logger.warning("Distance correction data not found")
            processed['distance_correction'] = None
        
        processed['raw_data'] = data
        
        self.logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
        log_memory_usage(self.logger, "After preprocessing")
        
        return processed
      
    def _process_road_network(self, road_network_gdf):
        """
        Process road network data
        
        Args:
            road_network_gdf (geopandas.GeoDataFrame): Original road network data
        
        Returns:
            geopandas.GeoDataFrame: Processed road network data
        """
        gdf = road_network_gdf.copy()
        
        required_columns = ['ID', 'Code', 'fclass', 'Long_meter', 'CityCode']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        
        if missing_columns:
            self.logger.error(f"Road network missing required columns: {missing_columns}")
            raise ValueError(f"Road network missing required columns: {missing_columns}")
        
        if self.clean_nan:
            nan_counts = gdf.isna().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Missing values in road network: {nan_counts}")
                
                critical_missing = gdf[gdf['ID'].isna() | gdf['Long_meter'].isna()]
                if len(critical_missing) > 0:
                    gdf = gdf.dropna(subset=['ID', 'Long_meter'])
                    self.logger.info(f"Removed {len(critical_missing)} records with missing ID or length")
                
                if gdf['Code'].isna().sum() > 0:
                    gdf['Code'] = gdf['Code'].fillna('')
                
                if gdf['CityCode'].isna().sum() > 0:
                    most_common_city = gdf['CityCode'].mode()[0]
                    gdf['CityCode'] = gdf['CityCode'].fillna(most_common_city)
        
        gdf.set_index('ID', inplace=True, drop=False)
        
        cache_file = self.cache_dir / "processed_road_network.parquet"
        try:
            gdf.to_parquet(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to save road network: {e}")
        
        return gdf
    
    def _build_road_to_city_map(self, road_network_gdf):
        """
        Build road ID to city code mapping
        
        Args:
            road_network_gdf (geopandas.GeoDataFrame): Road network data
        
        Returns:
            dict: Road ID to city code mapping
        """

        road_to_city = {str(row['ID']): row['CityCode'] for _, row in road_network_gdf.iterrows()}
        
        self.logger.info(f"Road to city mapping built: {len(road_to_city)} mappings")
        return road_to_city
    
    def _process_traffic_volume(self, traffic_volume_dict):
        """
        Process traffic volume data
        
        Args:
            traffic_volume_dict (dict): Mapping from date to traffic volume DataFrame
        
        Returns:
            dict: Processed traffic volume mapping
        """

        traffic_mapping = {}
        
        for date, df in traffic_volume_dict.items():
            if not all(col in df.columns for col in ['RouteCode', 'Hour', 'Congestion']):
                self.logger.warning(f"Traffic volume data missing required columns for date {date}")
                continue
            
            date_mapping = {}
            
            for _, row in df.iterrows():
                code = row['路线简码']
                hour = int(row['小时'])
                congestion = float(row['拥挤度'])
                
                if code not in date_mapping:
                    date_mapping[code] = {}
                
                date_mapping[code][hour] = congestion
            
            traffic_mapping[date] = date_mapping
        
        self.logger.info(f"Traffic volume processed: {len(traffic_mapping)} dates")
        return traffic_mapping
    
    def _process_stations(self, stations_gdf):
        """
        Process fuel station data
        
        Args:
            stations_gdf (geopandas.GeoDataFrame): Original fuel station data
        
        Returns:
            geopandas.GeoDataFrame: Processed fuel station data
        """

        gdf = stations_gdf.copy()
        
        required_columns = ['ID', 'LinkRoad', 'Lat', 'Lon', 'CityCode']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        
        if missing_columns:
            self.logger.error(f"Stations missing required columns: {missing_columns}")
            raise ValueError(f"Stations missing required columns: {missing_columns}")
        
        if self.clean_nan:
            nan_counts = gdf.isna().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Missing values in stations: {nan_counts}")
                
                critical_missing = gdf[gdf['ID'].isna() | gdf['Lat'].isna() | gdf['Lon'].isna()]
                if len(critical_missing) > 0:
                    gdf = gdf.dropna(subset=['ID', 'Lat', 'Lon'])
                    self.logger.info(f"Removed {len(critical_missing)} records with missing ID or coordinates")
                
                if gdf['LinkRoad'].isna().sum() > 0:
                    gdf['LinkRoad'] = gdf['LinkRoad'].fillna('0')
                
                if gdf['CityCode'].isna().sum() > 0:
                    most_common_city = gdf['CityCode'].mode()[0]
                    gdf['CityCode'] = gdf['CityCode'].fillna(most_common_city)
        
        gdf.set_index('ID', inplace=True, drop=False)
        gdf['StationType'] = 'FuelStation'
        
        cache_file = self.cache_dir / "processed_stations.parquet"
        try:
            gdf.to_parquet(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to save stations: {e}")
        
        return gdf
    
    def _process_temperature(self, temperature_df):
        """
        Process temperature data
        
        Args:
            temperature_df (pandas.DataFrame): Original temperature data
        
        Returns:
            dict: City-season-temperature mapping
        """
        temp_mapping = {}
        
        required_columns = ['CityCode', 'Season', 'Temperature']
        missing_columns = [col for col in required_columns if col not in temperature_df.columns]
        
        if missing_columns:
            self.logger.error(f"Temperature missing required columns: {missing_columns}")
            raise ValueError(f"Temperature missing required columns: {missing_columns}")
        
        for _, row in temperature_df.iterrows():
            city_code = row['CityCode']
            season = row['Season']
            temperature = row['Temperature']
            
            if city_code not in temp_mapping:
                temp_mapping[city_code] = {}
            
            temp_mapping[city_code][season] = temperature
        
        self.logger.info(f"Temperature processed: {len(temp_mapping)} cities")
        return temp_mapping
    
    def _process_vehicles(self, vehicles_df):
        """
        Process typical vehicle type data
        
        Args:
            vehicles_df (pandas.DataFrame): Original vehicle type data
        
        Returns:
            pandas.DataFrame: Processed vehicle type data
        """

        df = vehicles_df.copy()
        
        required_columns = ['Type', 'CargoWeight', 'VehicleCost', 'Capacity', 'BatteryCost', 
                           'LifeSpanVe', 'LifeSpanBat']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Vehicles missing required columns: {missing_columns}")
            raise ValueError(f"Vehicles missing required columns: {missing_columns}")
        
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in vehicles: {df.isna().sum()}")
            
            if 'BatteryCost' in df.columns and df['BatteryCost'].isna().any():
                df['BatteryCost'] = df['BatteryCost'].fillna(0)
                
            if 'Unit3' in df.columns and df['Unit3'].isna().any():
                df['Unit3'] = df['Unit3'].fillna('')
        
        df['VehicleCategory'] = df['Type'].apply(lambda x: 'BEV' if x.startswith('BEV') else 'HFCV')
        
        df['AnnualBaseCost'] = df['VehicleCost'] / df['LifeSpanVe']
        
        df['AnnualExtraBatteryCost'] = 0.0
        bev_mask = df['VehicleCategory'] == 'BEV'
        df.loc[bev_mask, 'AnnualExtraBatteryCost'] = 2 * df.loc[bev_mask, 'BatteryCost'] / df.loc[bev_mask, 'LifeSpanVe']
        
        df['AnnualTotalCost'] = df['AnnualBaseCost'] + df['AnnualExtraBatteryCost']
        
        vehicles_dict = df.set_index('Type').to_dict(orient='index')
        
        weight_groups = {}
        for _, row in df.iterrows():
            cargo_weight = row['CargoWeight']
            if cargo_weight not in weight_groups:
                weight_groups[cargo_weight] = {'BEV': [], 'HFCV': []}
            
            vehicle_type = row['VehicleCategory']
            weight_groups[cargo_weight][vehicle_type].append(row['Type'])
        
        result = {
            'df': df,
            'dict': vehicles_dict,
            'weight_groups': weight_groups
        }
        
        self.logger.info(f"Vehicles processed: {len(df)} types")
        return result
    
    def _process_charging_stations(self, cs_df):
        """
        Process charging station data
        
        Args:
            cs_df (pandas.DataFrame): Original charging station data
        
        Returns:
            dict: Processed charging station data
        """

        df = cs_df.copy()
        
        required_columns = ['Type', 'Cost', 'GHG', 'LifeSpanCS']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Charging stations missing required columns: {missing_columns}")
            raise ValueError(f"Charging stations missing required columns: {missing_columns}")
        
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in charging stations: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        df['AnnualCost'] = df['Cost'] / df['LifeSpanCS']
        df['AnnualGHG'] = df['GHG'] / df['LifeSpanCS']
        
        cs_dict = df.set_index('Type').to_dict(orient='index')
        
        self.logger.info(f"Charging stations processed: {len(df)} types")
        return {'df': df, 'dict': cs_dict}
    
    def _process_battery_swap_stations(self, bss_df):
        """
        Process battery swap station data
        
        Args:
            bss_df (pandas.DataFrame): Original battery swap station data
        
        Returns:
            dict: Processed battery swap station data
        """

        df = bss_df.copy()
        
        required_columns = ['Type', 'Cost', 'GHG', 'LifeSpanBSS', 'ServiceSpeed']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Battery swap stations missing required columns: {missing_columns}")
            raise ValueError(f"Battery swap stations missing required columns: {missing_columns}")
        
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in battery swap stations: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        df['AnnualCost'] = df['Cost'] / df['LifeSpanBSS']
        df['AnnualGHG'] = df['GHG'] / df['LifeSpanBSS']
        df['ServiceTimeSeconds'] = df['ServiceSpeed'] * 60
        
        bss_dict = df.set_index('Type').to_dict(orient='index')
        
        self.logger.info(f"Battery swap stations processed: {len(df)} types")
        return {'df': df, 'dict': bss_dict}
    
    def _process_hydrogen_stations(self, hrs_df):
        """
        Process hydrogen station data
        
        Args:
            hrs_df (pandas.DataFrame): Original hydrogen station data
        
        Returns:
            dict: Processed hydrogen station data
        """

        df = hrs_df.copy()
        
        required_columns = ['Type', 'Cost', 'GHG', 'Capacity', 'Efficiency', 
                           'LifeSpanHRS', 'ServiceSpeed']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Hydrogen stations missing required columns: {missing_columns}")
            raise ValueError(f"Hydrogen stations missing required columns: {missing_columns}")
        
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in hydrogen stations: {df.isna().sum()}")
        
        if 'Efficiency' in df.columns and df['Efficiency'].isna().any():
            df['Efficiency'] = df['Efficiency'].fillna(1)
        
        if 'Unit4' in df.columns and df['Unit4'].isna().any():
            df['Unit4'] = df['Unit4'].fillna('')
        
        df['AnnualCost'] = df['Cost'] / df['LifeSpanHRS']
        df['AnnualGHG'] = df['GHG'] / df['LifeSpanHRS']
        df['ServiceTimeSeconds'] = df['ServiceSpeed'] * 60
        
        hrs_dict = df.set_index('Type').to_dict(orient='index')
        
        self.logger.info(f"Hydrogen stations processed: {len(df)} types")
        return {'df': df, 'dict': hrs_dict}
    
    def _process_charging_posts(self, cp_df):
        """
        Process charging post data
        
        Args:
            cp_df (pandas.DataFrame): Original charging post data
        
        Returns:
            dict: Processed charging post data
        """

        df = cp_df.copy()
        
        required_columns = ['Type', 'Cost', 'Power', 'Efficiency', 'GHG', 'LifeSpanCP']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Charging posts missing required columns: {missing_columns}")
            raise ValueError(f"Charging posts missing required columns: {missing_columns}")
        
        if self.clean_nan:
            nan_counts = df.isna().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Missing values in charging posts: {nan_counts}")
                
                if 'Efficiency' in df.columns and df['Efficiency'].isna().any():
                    df['Efficiency'] = df['Efficiency'].fillna(0.9)
                
                if 'Unit3' in df.columns and df['Unit3'].isna().any():
                    df['Unit3'] = df['Unit3'].fillna('')
        
        df['AnnualCostPerPost'] = df['Cost'] * df['Power'] / df['LifeSpanCP']
        df['AnnualGHG'] = df['GHG'] / df['LifeSpanCP']
        
        cp_dict = df.set_index('Type').to_dict(orient='index')
        
        self.logger.info(f"Charging posts processed: {len(df)} types")
        return {'df': df, 'dict': cp_dict}
    
    def _process_electricity_price(self, electricity_price_df):
        """
        Process electricity price data
        
        Args:
            electricity_price_df (pandas.DataFrame): Original electricity price data
        
        Returns:
            dict: City-hour-electricity price mapping
        """

        price_mapping = {}
        
        required_columns = ['CityCode', 'Time', 'Cost']
        missing_columns = [col for col in required_columns if col not in electricity_price_df.columns]
        
        if missing_columns:
            self.logger.error(f"Electricity price missing required columns: {missing_columns}")
            raise ValueError(f"Electricity price missing required columns: {missing_columns}")
        
        df = electricity_price_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in electricity price: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            city_code = row['CityCode']
            time_hour = int(row['Time'])
            cost = float(row['Cost'])
            
            if city_code not in price_mapping:
                price_mapping[city_code] = {}
            
            price_mapping[city_code][time_hour] = cost
        
        self.logger.info(f"Electricity price processed: {len(price_mapping)} cities")
        return price_mapping
    
    def _process_hydrogen_price(self, hydrogen_price_df):
        """
        Process hydrogen price data
        
        Args:
            hydrogen_price_df (pandas.DataFrame): Original hydrogen price data
        
        Returns:
            dict: City-hour-hydrogen price mapping
        """

        price_mapping = {}
        
        required_columns = ['CityCode', 'Time', 'Cost']
        missing_columns = [col for col in required_columns if col not in hydrogen_price_df.columns]
        
        if missing_columns:
            self.logger.error(f"Hydrogen price missing required columns: {missing_columns}")
            raise ValueError(f"Hydrogen price missing required columns: {missing_columns}")
        
        df = hydrogen_price_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in hydrogen price: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            city_code = row['CityCode']
            time_hour = int(row['Time'])
            cost = float(row['Cost'])
            
            if city_code not in price_mapping:
                price_mapping[city_code] = {}
            
            price_mapping[city_code][time_hour] = cost
        
        self.logger.info(f"Hydrogen price processed: {len(price_mapping)} cities")
        return price_mapping
    
    def _process_parameters(self, parameters_df):
        """
        Process related parameters
        
        Args:
            parameters_df (pandas.DataFrame): Original parameter data
        
        Returns:
            dict: Parameter name-value mapping
        """

        param_mapping = {}
        
        required_columns = ['Parameter', 'Value']
        missing_columns = [col for col in required_columns if col not in parameters_df.columns]
        
        if missing_columns:
            self.logger.error(f"Parameters missing required columns: {missing_columns}")
            raise ValueError(f"Parameters missing required columns: {missing_columns}")
        
        df = parameters_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in parameters: {df.isna().sum()}")
        
        if 'Unit' in df.columns and df['Unit'].isna().any():
            df['Unit'] = df['Unit'].fillna('')
        
        for _, row in df.iterrows():
            param_name = row['Parameter']
            param_value = float(row['Value'])
            param_mapping[param_name] = param_value
        
        self.logger.info(f"Parameters processed: {len(param_mapping)} parameters")
        return param_mapping
    
    def _process_free_speed(self, free_speed_df):
        """
        Process free flow speed data
        
        Args:
            free_speed_df (pandas.DataFrame): Original free flow speed data
        
        Returns:
            dict: Road category-speed mapping
        """

        speed_mapping = {}
        
        required_columns = ['Category', 'FreeSpeed']
        missing_columns = [col for col in required_columns if col not in free_speed_df.columns]
        
        if missing_columns:
            self.logger.error(f"Free speed missing required columns: {missing_columns}")
            raise ValueError(f"Free speed missing required columns: {missing_columns}")
        
        df = free_speed_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in free speed: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            category = row['Category']
            speed = row['FreeSpeed']
            speed_mapping[category] = speed
        
        self.logger.info(f"Free speed processed: {len(speed_mapping)} categories")
        return speed_mapping
    
    def _process_vehicle_ghg(self, vehicle_ghg_df):
        """
        Process vehicle GHG emission data
        
        Args:
            vehicle_ghg_df (pandas.DataFrame): Original vehicle GHG emission data
        
        Returns:
            dict: Province-vehicle type-GHG emission mapping
        """

        ghg_mapping = {}
        
        required_columns = ['ProvinceCode', 'Type', 'GHGs']
        missing_columns = [col for col in required_columns if col not in vehicle_ghg_df.columns]
        
        if missing_columns:
            self.logger.error(f"Vehicle GHG missing required columns: {missing_columns}")
            raise ValueError(f"Vehicle GHG missing required columns: {missing_columns}")
        
        df = vehicle_ghg_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in vehicle GHG: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            province_code = row['ProvinceCode']
            vehicle_type = row['Type']
            ghg = float(row['GHGs'])
            
            if province_code not in ghg_mapping:
                ghg_mapping[province_code] = {}
            
            ghg_mapping[province_code][vehicle_type] = ghg
        
        self.logger.info(f"Vehicle GHG processed: {len(ghg_mapping)} provinces")
        return ghg_mapping
    
    def _process_battery_ghg(self, battery_ghg_df):
        """
        Process battery GHG emission data
        
        Args:
            battery_ghg_df (pandas.DataFrame): Original battery GHG emission data
        
        Returns:
            dict: Province-battery type-GHG emission mapping
        """

        ghg_mapping = {}
        
        required_columns = ['ProvinceCode', 'Type', 'GHGs']
        missing_columns = [col for col in required_columns if col not in battery_ghg_df.columns]
        
        if missing_columns:
            self.logger.error(f"Battery GHG missing required columns: {missing_columns}")
            raise ValueError(f"Battery GHG missing required columns: {missing_columns}")
        
        df = battery_ghg_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in battery GHG: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            province_code = row['ProvinceCode']
            battery_type = row['Type']
            ghg = float(row['GHGs'])
            
            if province_code not in ghg_mapping:
                ghg_mapping[province_code] = {}
            
            ghg_mapping[province_code][battery_type] = ghg
        
        self.logger.info(f"Battery GHG processed: {len(ghg_mapping)} provinces")
        return ghg_mapping
    
    def _process_province_city_code(self, province_city_df):
        """
        Process provincial and municipal code-related data
        
        Args:
            province_city_df (pandas.DataFrame): Original province and city code related data
        
        Returns:
            dict: Mapping of city codes to province codes
        """

        city_to_province = {}
        
        required_columns = ['ProvinceCode', 'CityCode']
        missing_columns = [col for col in required_columns if col not in province_city_df.columns]
        
        if missing_columns:
            self.logger.error(f"Province city code missing required columns: {missing_columns}")
            raise ValueError(f"Province city code missing required columns: {missing_columns}")
        
        df = province_city_df.copy()
        if self.clean_nan and df.isna().sum().sum() > 0:
            self.logger.warning(f"Missing values in province city code: {df.isna().sum()}")
            df = df.dropna(subset=required_columns)
        
        for _, row in df.iterrows():
            province_code = row['ProvinceCode']
            city_code = row['CityCode']
            city_to_province[city_code] = province_code
        
        self.logger.info(f"Province city code processed: {len(city_to_province)} cities")
        return city_to_province
        
    def _process_distance_correction(self, distance_correction_data):
        """
        Processing distance correction factor data
        
        Args:
            distance_correction_data (dict): Original distance correction data
            
        Returns:
            dict: Processed distance correction data for quick query
        """
        try:
            city_factors = distance_correction_data['city_factors']
            
            correction_lookup = {}
            
            for city_code, city_data in city_factors.items():
                city_code_str = str(city_code)
                correction_lookup[city_code_str] = {
                    'city_name': city_data['city_name'],
                    'base_factor': city_data['base_factor'],
                    'time_factors': {}
                }
                
                for date, hourly_factors in city_data['time_factors'].items():
                    date_str = str(date)
                    correction_lookup[city_code_str]['time_factors'][date_str] = hourly_factors
            
            self.logger.info(f"Distance correction processed: {len(correction_lookup)} cities")
            
            return {
                'metadata': distance_correction_data.get('metadata', {}),
                'city_factors': correction_lookup
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process distance correction: {e}")
            return None