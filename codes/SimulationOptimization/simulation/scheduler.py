"""
Author: Shiqi (Anya) WANG
Date: 2025/5/21
Description: Scheduling system module
Responsible for coordinating interactions between vehicles, stations and road networks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import multiprocessing as mp
from pathlib import Path
import os
import pickle
from collections import defaultdict
import random
import uuid
import shutil
import gc

from simulation.vehicle import BEV, HFCV
from simulation.station import StationManager
from utils.logger import ProgressLogger, log_memory_usage

from functools import lru_cache

import json

from numba import jit


class SimulationScheduler:
    """
    Simulation scheduler responsible for coordinating the entire simulation process
    """
    
    def __init__(self, config, logger, processed_data, cache_dir, n_workers=None, memmap_suffix=None):
        """
        Initialize simulation scheduler
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger): Logger
            processed_data (dict): Preprocessed data
            cache_dir (Path): Cache directory
            n_workers (int, optional): Number of parallel processing workers
            memmap_suffix (str, optional): Memory mapping file suffix for distinguishing different instances
        """
        self.config = config
        self.logger = logger
        self.data = processed_data
        self.cache_dir = cache_dir
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        
        self.season = config.get('simulation', {}).get('season', 'Winter')
        self.logger.info(f"Simulation season: {self.season}")
        
        self.parameters = processed_data['parameters']
        self.road_network = processed_data['road_network']
        self.road_to_city = processed_data['road_to_city']
        self.traffic_mapping = processed_data['traffic_mapping']
        self.temperature_mapping = processed_data['temperature_mapping']
        self.free_speed = processed_data['free_speed']
        self.electricity_price = processed_data['electricity_price']
        self.hydrogen_price = processed_data['hydrogen_price']
        self.stations = processed_data['stations']
        self.province_city_map = processed_data['province_city_map']
        
        self.distance_correction = processed_data.get('distance_correction', None)
        if self.distance_correction:
            self.logger.info("Distance correction data loaded")
        else:
            self.logger.warning("Distance correction data not found, using default factor 1.0")
        
        self.memmap_suffix = memmap_suffix or ""
        
        self.station_manager = StationManager(config, logger)
        
        self.simulation_results = {
            'vehicle_records': [],
            'path_execution': [],
            'parking_records': [],
            'driving_statistics': [],
            'service_records': [],
            'station_service_records': [],
            'cost_records': {
                'capital_cost': 0.0,
                'energy_cost': 0.0
            },
            'ghg_records': {
                'infrastructure_ghg': 0.0,
                'vehicle_ghg': 0.0
            }
        }
        
        self.actual_vehicle_map = {}
        self.used_vehicles = set()
        
        random.seed(config['optimization']['simulated_annealing']['random_seed'])
        
        self.memmap_dir = self.cache_dir / f'memmap{self.memmap_suffix}'
        os.makedirs(self.memmap_dir, exist_ok=True)
        
        if os.path.exists(self.memmap_dir):
            try:
                shutil.rmtree(self.memmap_dir)
                os.makedirs(self.memmap_dir, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to clean memmap directory: {e}")
        
        self.infeasible_solutions = []
        
        self.memmap_files = {}
        
        if 'trajectory_metadata' in processed_data:
            self.trajectory_metadata = processed_data['trajectory_metadata']
            logger.info(f"Trajectory metadata set with {self.trajectory_metadata['total_count']} records")
        else:
            logger.warning("Trajectory metadata not found")
            self.trajectory_metadata = None
        
        self.battery_demand_tracker = defaultdict(lambda: defaultdict(list))
        self.daily_max_battery_demand = defaultdict(lambda: defaultdict(int))
        
        self.hydrogen_demand_tracker = defaultdict(lambda: defaultdict(float))
        self.daily_max_hydrogen_demand = defaultdict(float)
    
    def _create_memmap(self, name, shape, dtype=np.float32):
        """
        Create or get memory mapping file
        
        Args:
            name (str): Mapping file name
            shape (tuple): Array shape
            dtype (np.dtype): Data type
            
        Returns:
            numpy.memmap: Memory mapped array
        """
        filename = self.memmap_dir / f"{name}.dat"
    
        try:
            memmap = np.memmap(
                filename,
                dtype=dtype,
                mode='w+',
                shape=shape
            )
            
            self.memmap_files[name] = {
                'filename': str(filename),
                'shape': shape,
                'dtype': str(dtype)
            }
            
            return memmap
        except Exception as e:
            self.logger.error(f"Failed to create memmap file: {e}")
            return None
    
    def _get_memmap(self, name):
        """
        Get existing memory mapping file
        
        Args:
            name (str): Mapping file name
            
        Returns:
            numpy.memmap: Memory mapped array, None if not exists
        """
        if name not in self.memmap_files:
            self.logger.warning(f"Memmap file {name} does not exist")
            return None
        
        metadata = self.memmap_files[name]
        
        try:
            memmap = np.memmap(
                metadata['filename'],
                dtype=eval(metadata['dtype']),
                mode='r+',
                shape=metadata['shape']
            )
            return memmap
        except Exception as e:
            self.logger.error(f"Failed to get memmap file: {e}")
            return None
    
    def _cleanup_memmaps(self):
        """
        Clean up all memory mapping files
        """
        for name, metadata in list(self.memmap_files.items()):
            try:
                filename = metadata['filename']
                if os.path.exists(filename):
                    os.remove(filename)
                    pass
                
                metadata_file = f"{os.path.splitext(filename)[0]}_metadata.json"
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    
            except Exception as e:
                self.logger.warning(f"Failed to remove memmap file: {e}")
        
        self.memmap_files = {}
    
    @jit(nopython=True)
    def _calculate_distance_numba(lon1, lat1, lon2, lat2):
        """Numba accelerated distance calculation"""
        R = 6371.0
        
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance

    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371.0
        
        lon1, lat1 = point1
        lon2, lat2 = point2
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _get_distance_correction_factor(self, city_code, date, hour, station_id=None):
        """
        Get distance correction factor
        
        Args:
            city_code (int/str): City code
            date (int): Date (YYYYMMDD)
            hour (int): Hour (0-23)
            station_id (str, optional): Station ID
            
        Returns:
            float: Distance correction factor
        """
        if not self.distance_correction or not self.distance_correction.get('city_factors'):
            return 1.0
        
        try:
            city_code_str = str(city_code)
            date_str = str(date)
            
            city_factors = self.distance_correction['city_factors']
            if city_code_str not in city_factors:
                return 1.0
            
            city_data = city_factors[city_code_str]
            
            if date_str in city_data['time_factors']:
                hourly_factors = city_data['time_factors'][date_str]
                if 0 <= hour < len(hourly_factors):
                    return hourly_factors[hour]
            
            return city_data.get('base_factor', 1.0)
            
        except Exception as e:
            self.logger.debug(f"Failed to get distance correction factor: {e}")
            return 1.0
    
    def configure_infrastructure(self, station_config):
        """
        Configure infrastructure (charging stations, battery swap stations, hydrogen stations)
        
        Args:
            station_config (dict): Infrastructure configuration
            
        Returns:
            dict: Infrastructure statistics
        """
        self.logger.info("Configuring infrastructure")
        
        self.station_manager = StationManager(self.config, self.logger)
        
        cs_count = 0
        bss_count = 0
        hrs_count = 0
        fcp_count = 0
        scp_count = 0
        bev_batteries = defaultdict(int)
        
        for station_id, config in station_config.items():
            station_info = self.stations.loc[self.stations['ID'] == int(station_id)].iloc[0].to_dict()
            
            if config.get('CS', 0) > 0:
                cs_type = 'CS_01'
                fcp = config.get('FCP', 0)
                scp = config.get('SCP', 0)
                
                self.station_manager.add_charging_station(
                    station_id=station_id,
                    station_info=station_info,
                    cs_type=cs_type,
                    fcp_count=fcp,
                    scp_count=scp,
                    cs_data=self.data['charging_stations'],
                    cp_data=self.data['charging_posts']
                )
                
                cs_count += 1
                fcp_count += fcp
                scp_count += scp
            
            if config.get('BSS', 0) > 0:
                bss_type = 'BSS_01'
                
                spare_batteries = {}
                
                self.station_manager.add_battery_swap_station(
                    station_id=station_id,
                    station_info=station_info,
                    bss_type=bss_type,
                    spare_batteries=spare_batteries,
                    bss_data=self.data['battery_swap_stations']
                )
                
                bss_count += 1
            
            if config.get('HRS_01', 0) > 0:
                hrs_type = 'HRS_01'
                
                self.station_manager.add_hydrogen_station(
                    station_id=station_id,
                    station_info=station_info,
                    hrs_type=hrs_type,
                    hrs_data=self.data['hydrogen_stations']
                )
                
                hrs_count += 1
            
        infrastructure_stats = {
            'charging_stations': cs_count,
            'battery_swap_stations': bss_count,
            'hydrogen_stations': hrs_count,
            'fast_charging_posts': fcp_count,
            'slow_charging_posts': scp_count,
            'spare_batteries': dict(bev_batteries)
        }
        
        self.logger.info(f"Infrastructure configured: {infrastructure_stats}")
        return infrastructure_stats

    def assign_vehicle_types(self, vehicle_assignments):
        """
        Assign vehicle types
        
        Args:
            vehicle_assignments (dict): Vehicle type assignments
            
        Returns:
            dict: Vehicle type statistics
        """
        self.logger.info("Assigning vehicle types")
        
        vehicle_type_stats = defaultdict(int)
        bev_count = 0
        hfcv_count = 0
        
        self.actual_vehicle_map = {}
        self.used_vehicles = set()
        
        trajectory_count = 0
        
        for vehicle_id, assignment in vehicle_assignments.items():
            trajectory_count += 1
            if isinstance(assignment, dict):
                vehicle_type = assignment['Type']
                actual_vehicle_id = assignment['ActualVehicleID']
                
                self.actual_vehicle_map[vehicle_id] = assignment
                self.used_vehicles.add(actual_vehicle_id)
                
                vehicle_type_stats[vehicle_type] += 1
                if vehicle_type.startswith('BEV_'):
                    bev_count += 1
                elif vehicle_type.startswith('HFCV_'):
                    hfcv_count += 1
            else:
                self.logger.warning(f"Incorrect assignment format for vehicle {vehicle_id}: {assignment}")
        
        vehicle_stats = {
            'total_trajectories': trajectory_count,
            'total_vehicles': len(self.used_vehicles),
            'total_bev': bev_count,
            'total_hfcv': hfcv_count,
            'type_distribution': dict(vehicle_type_stats)
        }
        
        self.logger.info(f"Vehicle type assignment completed: {len(self.actual_vehicle_map)} vehicle IDs assigned")
        return vehicle_stats



    def simulate(self, trajectories=None, batch_size=1000, filter_func=None, max_trajectories=None):
        """
        Execute simulation with streaming trajectory data processing
        
        Args:
            trajectories (pandas.DataFrame, optional): Trajectory data, use streaming loading if not specified
            batch_size (int): Batch processing size
            filter_func (callable, optional): Trajectory filter function
            max_trajectories (int, optional): Maximum number of trajectories to process
        """
        start_time = time.time()
        self.logger.info("Starting simulation")
        log_memory_usage(self.logger, "Before simulation")
        
        self._initialize_simulation_results()
        
        def flexible_vehicle_filter(record):
            vehicle_id = str(record['VehicleID'])
            if vehicle_id in self.actual_vehicle_map:
                return True
            
            cargo_weight = float(record.get('核定载质量', 0))
            
            vehicle_type = self._determine_vehicle_type(cargo_weight)
            
            actual_vehicle_id = f"auto_generated_{len(self.actual_vehicle_map)}"
            
            self.actual_vehicle_map[vehicle_id] = {
                'Type': vehicle_type,
                'ActualVehicleID': actual_vehicle_id
            }
            

            
            return True
        
        if filter_func is None:
            filter_func = flexible_vehicle_filter
        else:
            original_filter = filter_func
            filter_func = lambda record: original_filter(record) and flexible_vehicle_filter(record)
        
        
        if trajectories is not None:
            total_trajectories = len(trajectories)
            use_streaming = False
            self.logger.info(f"Using provided trajectory data: {total_trajectories} records")
        elif hasattr(self, 'trajectory_metadata') and self.trajectory_metadata is not None:
            total_trajectories = self.trajectory_metadata['total_count']
            use_streaming = True
            self.logger.info(f"Using streaming processing: {total_trajectories} trajectories")
        else:
            try:
                season = getattr(self, 'season', 'Winter')
                cache_file = self.cache_dir / f"trajectory_{season}_index.pkl"
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        self.trajectory_metadata = pickle.load(f)
                    total_trajectories = self.trajectory_metadata['total_count']
                    use_streaming = True
                else:
                    self.logger.error("Trajectory metadata cache file not found")
                    return self.simulation_results
            except Exception as e:
                self.logger.error(f"Failed to load trajectory metadata: {e}")
                return self.simulation_results
        
        if max_trajectories and max_trajectories < total_trajectories:
            original_total = total_trajectories
            total_trajectories = max_trajectories
            self.logger.info(f"Limiting trajectory processing: from {original_total} to {total_trajectories}")
        else:
            self.logger.info(f"Processing all {total_trajectories} trajectories")
        
        progress = ProgressLogger(self.logger, total_trajectories, "Simulation")
        progress.start()
        

        self.infeasible_solutions = []
        

        processed_count = 0
        batch_count = 0
        empty_batches_count = 0
        max_empty_batches = 10
        
        while processed_count < total_trajectories:
            if use_streaming:
                from data_processor.loader import DataLoader
                loader = DataLoader(self.config, self.logger, self.season)
                batch_trajectories = loader.load_trajectory_batch(
                    start_idx=processed_count,
                    batch_size=batch_size,
                    filter_func=filter_func
                )
            else:
                batch_end = min(processed_count + batch_size, total_trajectories)
                batch_trajectories = trajectories.iloc[processed_count:batch_end]
            
            if batch_trajectories.empty:
                self.logger.warning(f"Empty batch starting from index {processed_count}")
                processed_count += batch_size
                empty_batches_count += 1
                
                if empty_batches_count >= max_empty_batches:
                    self.logger.warning("Consecutive empty batches, stopping processing")
                    break
                
                continue
            else:
                empty_batches_count = 0
            
            batch_count += 1
            
            self._process_trajectory_batch(batch_trajectories, progress, batch_count, processed_count)
            
            processed_count += batch_size
            
            import gc
            gc.collect()
            log_memory_usage(self.logger, f"Batch {batch_count} completed")
            
            if max_trajectories and processed_count >= max_trajectories:
                break
            
            if self.config.get('data_processing', {}).get('save_intermediate_results', False):
                self._save_intermediate_results(batch_count, processed_count)
        
        progress.finish()
        
        self._finalize_battery_requirements()
        
        self._finalize_hydrogen_requirements()
        
        self._calculate_total_cost()
        self._calculate_total_ghg()
        
        self._collect_station_service_records()
        
        self._log_simulation_statistics()
        
        self.logger.info("Simulation completed")
        log_memory_usage(self.logger, "After simulation")
        
        return self.simulation_results

    def _determine_vehicle_type(self, cargo_weight):
        """Determine vehicle type based on cargo weight"""
        if cargo_weight <= 10:
            return "BEV_01"
        elif cargo_weight <= 20:
            return "BEV_02"
        else:
            return "HFCV_01"

    def _initialize_simulation_results(self):
        """Initialize simulation results"""
        self.simulation_results = {
            'vehicle_records': [],
            'path_execution': [],
            'path_execution_memmaps': [],
            'parking_records': [],
            'parking_records_memmaps': [],
            'driving_statistics': [],
            'service_records': [],
            'service_records_memmaps': [],
            'station_service_records': [],
            'cost_records': {
                'capital_cost': 0.0,
                'energy_cost': 0.0
            },
            'ghg_records': {
                'infrastructure_ghg': 0.0,
                'vehicle_ghg': 0.0
            }
        }
        
        self.results_batch_count = 0
        self.max_results_per_batch = 1000
        
        self._cleanup_memmaps()

    def _process_trajectory_batch(self, batch_trajectories, progress, batch_count=0, processed_count=0):
        """Process a batch of trajectory data"""
        if self.config.get('simulation', {}).get('parallel_vehicles', False) and self.n_workers > 1:
            with mp.Pool(self.n_workers) as pool:
                vehicle_info_list = []
                for i, trajectory in batch_trajectories.iterrows():
                    vehicle_id = str(trajectory['VehicleID'])
                    info = {
                        'trajectory': trajectory,
                        'vehicle_id': vehicle_id,
                        'vehicle_assignment': self.actual_vehicle_map[vehicle_id]
                    }
                    vehicle_info_list.append(info)
                
                results = pool.map(self._simulate_vehicle_worker, vehicle_info_list)
                
                for result in results:
                    if result:
                        self._merge_vehicle_simulation_result(result)
                        
                        if result.get('infeasible', False):
                            self.infeasible_solutions.append({
                                'vehicle_id': result.get('vehicle_id'),
                                'vehicle_type': result.get('vehicle_type'),
                                'reason': result.get('infeasible_reason', 'Unknown reason')
                            })
                
                progress.update(len(vehicle_info_list))
                
                self.results_batch_count += 1
                if self.results_batch_count >= 5:
                    self._save_intermediate_results(batch_count, processed_count)
                    
                    if len(self.simulation_results['path_execution']) > self.max_results_per_batch:
                        self.simulation_results['path_execution'] = self.simulation_results['path_execution'][-100:]
                    if len(self.simulation_results['parking_records']) > self.max_results_per_batch:
                        self.simulation_results['parking_records'] = self.simulation_results['parking_records'][-100:]
                    
                    gc.collect()
                    self.results_batch_count = 0
                
        else:
            for i, trajectory in batch_trajectories.iterrows():
                vehicle_id = str(trajectory['VehicleID'])
                
                assignment = self.actual_vehicle_map[vehicle_id]
                
                if isinstance(assignment, dict):
                    vehicle_type = assignment['Type']
                    actual_vehicle_id = assignment['ActualVehicleID']
                else:
                    vehicle_type = self._get_vehicle_type(vehicle_id, trajectory['核定载质量'])
                    actual_vehicle_id = assignment
                    
                vehicle = self._create_vehicle(
                    vehicle_id=vehicle_id,
                    trajectory_info=trajectory,
                    vehicle_type=vehicle_type,
                    actual_vehicle_id=actual_vehicle_id
                )
                
                if vehicle is None:
                    self.logger.warning(f"Failed to create vehicle {vehicle_id}")
                    progress.update()
                    continue
                
                result = self._simulate_single_vehicle(vehicle)
                
                if result.get("infeasible", False):
                    self.infeasible_solutions.append({
                        'vehicle_id': vehicle_id,
                        'vehicle_type': vehicle_type,
                        'reason': result.get("infeasible_reason", "Unknown reason")
                    })
                    
                    if self.config.get('optimization', {}).get('early_stop_all_infeasible', True):
                        self.logger.info("Batch processing terminated early due to infeasibility")
                        break
                
                gc.collect()
                self._cleanup_memmaps()
                
                del vehicle
                
                progress.update()

    def _simulate_vehicle_worker(self, vehicle_info):
        """Worker function for parallel vehicle simulation"""
        try:
            trajectory = vehicle_info['trajectory']
            vehicle_id = vehicle_info['vehicle_id']
            assignment = vehicle_info['vehicle_assignment']
            
            vehicle_type = assignment['Type'] if isinstance(assignment, dict) else self._get_vehicle_type(vehicle_id, trajectory['核定载质量'])
            actual_vehicle_id = assignment['ActualVehicleID'] if isinstance(assignment, dict) else assignment
            
            if not vehicle_type:
                return None
            vehicle = self._create_vehicle(
                vehicle_id=vehicle_id,
                trajectory_info=trajectory,
                vehicle_type=vehicle_type,
                actual_vehicle_id=actual_vehicle_id
            )
            
            if vehicle is None:
                return None
            
            result = self._simulate_single_vehicle(vehicle)
            result['vehicle_id'] = vehicle_id
            result['vehicle_type'] = vehicle_type
            
            return result
        except Exception as e:
            self.logger.exception(f"Parallel vehicle simulation failed: {e}")
            return None

    def _merge_vehicle_simulation_result(self, result):
        """Merge single vehicle simulation results into total results"""
        if 'vehicle_record' in result:
            self.simulation_results['vehicle_records'].append(result['vehicle_record'])
        
        if 'path_execution' in result:
            self.simulation_results['path_execution'].extend(result['path_execution'])
        
        if 'path_execution_memmap' in result:
            self.simulation_results['path_execution_memmaps'].append(result['path_execution_memmap'])
        
        if 'parking_records' in result:
            self.simulation_results['parking_records'].extend(result['parking_records'])
        
        if 'parking_records_memmap' in result:
            self.simulation_results['parking_records_memmaps'].append(result['parking_records_memmap'])
        
        if 'driving_stats' in result:
            self.simulation_results['driving_statistics'].append(result['driving_stats'])

    def _save_intermediate_results(self, batch_count, processed_count):
        """Save intermediate results"""
        temp_results = {
            'vehicle_records': self.simulation_results['vehicle_records'][:],
            'driving_statistics': self.simulation_results['driving_statistics'][:],
            'infeasible_solutions': self.infeasible_solutions[:] if hasattr(self, 'infeasible_solutions') else [],
            'processed_count': processed_count
        }
        
        temp_file = self.cache_dir / f"batch_results_{batch_count}.pkl"
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(temp_results, f)
        except Exception as e:
            self.logger.warning(f"Failed to save batch results: {e}")

    def _create_vehicle(self, vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id):
        """
        Create vehicle object
        
        Args:
            vehicle_id (int): Vehicle trip ID
            trajectory_info (pandas.Series): Trajectory information
            vehicle_type (str): Vehicle type
            actual_vehicle_id (str): Actual vehicle ID
            
        Returns:
            Vehicle: BEV or HFCV vehicle object
        """
        try:
            if vehicle_type.startswith('BEV_'):
                return BEV(
                    vehicle_id=vehicle_id,
                    trajectory_info=trajectory_info,
                    vehicle_type=vehicle_type,
                    actual_vehicle_id=actual_vehicle_id,
                    config=self.config,
                    parameters=self.parameters,
                    vehicle_data=self.data['vehicles'],
                    logger=self.logger
                )
            elif vehicle_type.startswith('HFCV_'):
                return HFCV(
                    vehicle_id=vehicle_id,
                    trajectory_info=trajectory_info,
                    vehicle_type=vehicle_type,
                    actual_vehicle_id=actual_vehicle_id,
                    config=self.config,
                    parameters=self.parameters,
                    vehicle_data=self.data['vehicles'],
                    logger=self.logger
                )
            else:
                self.logger.error(f"Unknown vehicle type: {vehicle_type}")
                return None
        
        except Exception as e:
            self.logger.exception(f"Error creating vehicle {vehicle_id}: {e}")
            return None
    
    def _get_vehicle_type(self, vehicle_id, cargo_weight):
        """
        Get vehicle type
        
        Args:
            vehicle_id (int): Vehicle trip ID
            cargo_weight (float): Approved load weight
            
        Returns:
            str: Vehicle type
        """
        if vehicle_id not in self.actual_vehicle_map:
            self.logger.warning(f"Vehicle ID {vehicle_id} missing type assignment")
            return None
        
        if isinstance(self.actual_vehicle_map[vehicle_id], dict):
            return self.actual_vehicle_map[vehicle_id]['Type']
        
        return None
    
    def _simulate_single_vehicle(self, vehicle):
        """
        Execute single vehicle simulation
        
        Args:
            vehicle (Vehicle): Vehicle object
        """
        current_date = vehicle.date
        
        infeasible = False
        infeasible_reason = ""
        
        early_stop_infeasible = self.config.get('optimization', {}).get('early_stop_infeasible', True)
               
        if (hasattr(vehicle, 'need_charge') and vehicle.need_charge) or \
           (hasattr(vehicle, 'need_refuel') and vehicle.need_refuel):
            is_feasible, reason = self._handle_energy_refill(vehicle)
            if not is_feasible:
                infeasible = True
                infeasible_reason = reason
                self.logger.warning(reason)
                
                if not hasattr(self, 'infeasible_solutions'):
                    self.infeasible_solutions = []
                self.infeasible_solutions.append({
                    'vehicle_id': vehicle.vehicle_id,
                    'vehicle_type': vehicle.vehicle_type,
                    'reason': reason
                })
                
                if self.config.get('optimization', {}).get('early_stop_all_infeasible', False):
                    infeasible_count = len(self.infeasible_solutions)
                    processed_count = len(self.simulation_results.get('vehicle_records', [])) + 1
                    if processed_count > 0:
                        infeasible_ratio = infeasible_count / processed_count
                        early_stop_threshold = self.config.get('optimization', {}).get('early_stop_threshold', 0.3)
                        
                        if infeasible_ratio > early_stop_threshold:
                            self.logger.info(f"Infeasibility ratio ({infeasible_ratio:.2%}) exceeds threshold ({early_stop_threshold:.2%}), stopping batch processing")
                            return {
                                "infeasible": True,
                                "infeasible_reason": reason
                            }
                

                return {
                    "infeasible": True,
                    "infeasible_reason": reason,
                    "vehicle_record": {
                        'VehicleID': vehicle.vehicle_id,
                        'Type': vehicle.vehicle_type,
                        'Path': ','.join(vehicle.path_list),
                        'ActualVehicleID': vehicle.actual_vehicle_id,
                        'Date': vehicle.date,
                        'Time': vehicle.start_time,
                        'Status': 'Infeasible'
                    },
                    "driving_stats": {
                        'VehicleID': vehicle.vehicle_id,
                        'ActualVehicleID': vehicle.actual_vehicle_id,
                        'VehicleType': vehicle.vehicle_type,
                        'TotalDrivingTime': 0,
                        'TotalRestTime': 0,
                        'RestCount': 0,
                        'AverageRestDuration': 0,
                        'MaxContinuousDrivingTime': 0,
                        'TotalDistance': 0,
                        'EnergyRefillCount': 0,
                        'EnergyRefillTime': 0,
                        'EnergyRefillAmount': 0,
                        'EnergyRefillCost': 0,
                        'DetourCount': 0,
                        'DetourDistance': 0,
                        'DetourTime': 0,
                        'GHGEmission': 0,
                        'Status': 'Infeasible'
                    }
                }
        

        self.simulation_results['vehicle_records'].append({
            'VehicleID': vehicle.vehicle_id,
            'Type': vehicle.vehicle_type,
            'Path': ','.join(vehicle.path_list),
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'Date': vehicle.date,
            'Time': vehicle.start_time
        })
            

        last_update_time = vehicle.current_time
        for i in range(len(vehicle.path_list) - 1):
            current_road_id = vehicle.path_list[i]
            next_road_id = vehicle.path_list[i + 1]
            

            current_time = vehicle.current_time
            
            time_since_last_update = (current_time - last_update_time).total_seconds()
            if time_since_last_update >= 1800:
                self._update_station_status(current_time)
                last_update_time = current_time
            
            current_road = self._get_road_info(current_road_id)

            if current_road is None:
                self.logger.warning(f"Road ID {current_road_id} not found, skipping")
                continue

            road_category, road_code, road_length, city_code = self._get_road_info_cached(current_road_id)
            congestion, speed = self._get_road_congestion_speed(
                road_id=current_road_id,
                road_category=road_category,
                road_code=road_code,
                date=vehicle.date,
                time=vehicle.current_time.hour
            )

            travel_time = self._calculate_travel_time(road_length, speed)
            temperature = self._get_temperature(
                city_code=current_road['CityCode'],
                date=vehicle.date
            )
            
            energy_consumption = vehicle.calculate_energy_consumption(
                distance=road_length,
                time=travel_time,
                temperature=temperature
            )
            
            vehicle.update_soc(energy_consumption)
            
            city_code = current_road['CityCode']
            province_code = self._get_province_code(city_code)
            if hasattr(vehicle, 'vehicle_type'):
                ghg_per_km = self._get_vehicle_ghg(
                    province_code=province_code,
                    vehicle_type=vehicle.vehicle_type
                )
                
                road_length_km = road_length / 1000
                road_ghg = ghg_per_km * road_length_km
                vehicle.ghg_emission += road_ghg
            if (hasattr(vehicle, 'need_charge') and vehicle.need_charge) or \
               (hasattr(vehicle, 'need_refuel') and vehicle.need_refuel):
                is_feasible, reason = self._handle_energy_refill(vehicle)
                if not is_feasible:
                    infeasible = True
                    infeasible_reason = reason
                    self.logger.warning(reason)
                    
                    if not hasattr(self, 'infeasible_solutions'):
                        self.infeasible_solutions = []
                    self.infeasible_solutions.append({
                        'vehicle_id': vehicle.vehicle_id,
                        'vehicle_type': vehicle.vehicle_type,
                        'reason': reason
                    })
                    
                    if self.config.get('optimization', {}).get('early_stop_all_infeasible', False):
                        infeasible_count = len(self.infeasible_solutions)
                        processed_count = len(self.simulation_results.get('vehicle_records', [])) + 1
                        if processed_count > 0:
                            infeasible_ratio = infeasible_count / processed_count
                            early_stop_threshold = self.config.get('optimization', {}).get('early_stop_threshold', 0.3)
                            
                            if infeasible_ratio > early_stop_threshold:
                                self.logger.info(f"Infeasibility ratio ({infeasible_ratio:.2%}) exceeds threshold ({early_stop_threshold:.2%})")
                                return {
                                    "infeasible": True,
                                    "infeasible_reason": reason
                                }
                    driving_stats = vehicle.get_driving_statistics()
                    driving_stats['VehicleType'] = vehicle.vehicle_type
                    driving_stats['Status'] = 'Infeasible'
                        
                    return {
                        "infeasible": True,
                        "infeasible_reason": reason,
                        "path_execution": vehicle.path_execution,
                        "parking_records": vehicle.parking_records,
                        "driving_stats": driving_stats
                    }
            
            if vehicle.check_rest_needed():
                self._handle_rest(vehicle)
            start_time = vehicle.current_time
            end_time = start_time + timedelta(seconds=travel_time)
            vehicle.current_time = end_time
            vehicle.total_distance += road_length
            vehicle.total_travel_time += travel_time
            vehicle.continuous_driving_time += travel_time
            vehicle.update_path_execution(
                road_id=current_road_id,
                start_time=start_time,
                end_time=end_time,
                distance=road_length,
                travel_time=travel_time,
                is_detour=False
            )
        
        driving_stats = vehicle.get_driving_statistics()
        driving_stats['VehicleType'] = vehicle.vehicle_type
        self.simulation_results['driving_statistics'].append(driving_stats)
        
        path_execution_count = len(vehicle.path_execution)
        if path_execution_count > 5000:
            try:
                path_execution_data = np.zeros((path_execution_count, 7), dtype=np.float32)
                path_execution_metadata = []
                
                for i, record in enumerate(vehicle.path_execution):
                    path_execution_data[i, 0] = float(record['VehicleID'])
                    path_execution_data[i, 1] = record['Distance']
                    path_execution_data[i, 2] = record['TravelTime']
                    path_execution_data[i, 3] = 1.0 if record['IsDetour'] else 0.0
                    path_execution_metadata.append({
                        'RoadID': record['RoadID'],
                        'StartTime': record['StartTime'],
                        'EndTime': record['EndTime']
                    })
                memmap_name = f"path_exec_{vehicle.vehicle_id}"
                path_memmap = self._create_memmap(
                    memmap_name,
                    shape=path_execution_data.shape,
                    dtype=np.float32
                )
                
                if path_memmap is not None:
                    path_memmap[:] = path_execution_data[:]
                    
                    metadata_file = self.memmap_dir / f"{memmap_name}_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(path_execution_metadata, f)
                    
                    self.simulation_results['path_execution_memmaps'].append(memmap_name)
                else:
                    self.simulation_results['path_execution'].extend(vehicle.path_execution)
            except Exception as e:
                self.logger.warning(f"Failed to store path execution records in memmap: {e}")
                self.simulation_results['path_execution'].extend(vehicle.path_execution)
        else:
            self.simulation_results['path_execution'].extend(vehicle.path_execution)
        
        parking_records_count = len(vehicle.parking_records)
        if parking_records_count > 500:
            try:
                parking_data = np.zeros((parking_records_count, 5), dtype=np.float32)
                parking_metadata = []
                
                for i, record in enumerate(vehicle.parking_records):
                    parking_data[i, 0] = float(record['VehicleID'])
                    parking_data[i, 1] = float(record['ParkingDuration'])
                    parking_data[i, 2] = float(record['ContinuousDrivingTime'])
                    parking_data[i, 3] = 1.0 if record['IsEnergyRefill'] else 0.0
                    parking_data[i, 4] = float(record['EnergyRefillAmount'])
                    
                    parking_metadata.append({
                        'ParkingLocationID': record['ParkingLocationID'],
                        'ParkingLocationType': record['ParkingLocationType'],
                        'ArrivalTime': record['ArrivalTime'],
                        'DepartureTime': record['DepartureTime']
                    })
                memmap_name = f"parking_{vehicle.vehicle_id}"
                parking_memmap = self._create_memmap(
                    memmap_name,
                    shape=parking_data.shape,
                    dtype=np.float32
                )
                
                if parking_memmap is not None:
                    parking_memmap[:] = parking_data[:]
                    
                    metadata_file = self.memmap_dir / f"{memmap_name}_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(parking_metadata, f)
                    
                    self.simulation_results['parking_records_memmaps'].append(memmap_name)
                else:
                    self.simulation_results['parking_records'].extend(vehicle.parking_records)
            except Exception as e:
                self.logger.warning(f"Failed to store parking records in memmap: {e}")
                self.simulation_results['parking_records'].extend(vehicle.parking_records)
        else:
            self.simulation_results['parking_records'].extend(vehicle.parking_records)
        return {
            "infeasible": infeasible,
            "infeasible_reason": infeasible_reason,
            "driving_stats": driving_stats
        }
    
    def get_station_config(self):
        """
        Get current infrastructure configuration
        
        Returns:
            dict: Infrastructure configuration
        """
        station_config = {}
        for station_id, station in self.station_manager.charging_stations.items():
            station_config[station_id] = {
                'CS': 1,
                'FCP': station.fcp_count,
                'SCP': station.scp_count,
                'BSS': 0
            }
        
        for station_id, station in self.station_manager.battery_swap_stations.items():
            if station_id not in station_config:
                station_config[station_id] = {
                    'CS': 0,
                    'FCP': 0,
                    'SCP': 0,
                    'BSS': 1
                }
            else:
                station_config[station_id]['BSS'] = 1
            
            for bev_type, count in station.spare_batteries.items():
                station_config[station_id][bev_type] = count
        
        for station_id, station in self.station_manager.hydrogen_stations.items():
            if station_id not in station_config:
                station_config[station_id] = {
                    'CS': 0,
                    'FCP': 0,
                    'SCP': 0,
                    'BSS': 0
                }
            
            station_config[station_id][station.hrs_type] = 1
        
        return station_config
    
    def _update_station_status(self, current_time):
        """Update all station status centrally"""
        hydrogen_refresh_interval = self.parameters.get('HydrogenFuelTime', 24.0)
        battery_refresh_interval = self.parameters.get('SwapBatteryFullTime', 24.0)
        
        for station_id, station in self.station_manager.hydrogen_stations.items():
            station.update_hydrogen_status(current_time, hydrogen_refresh_interval)
        
        for station_id, station in self.station_manager.battery_swap_stations.items():
            station.update_battery_status(current_time)
    
    @lru_cache(maxsize=2048)
    def _get_road_info_cached(self, road_id):
        """
        Get road information
        
        Args:
            road_id (str): Road ID
            
        Returns:
            tuple: (road category, road code, road length, city code)
        """
        try:
            road = self.road_network.loc[road_id]
            return (
                road.get('fclass', ''),
                road.get('Code', ''),
                road.get('Long_meter', 0.0),
                road.get('CityCode', 0)
            )
        except (KeyError, IndexError):
            self.logger.warning(f"Road ID not found: {road_id}")
            return ('', '', 0.0, 0)
    
    def _get_road_info(self, road_id):
        """
        Get road information
        
        Args:
            road_id (str): Road ID
            
        Returns:
            dict: Road information
        """
        try:
            road = self.road_network.loc[road_id]
            return road.to_dict()
        except (KeyError, IndexError):
            self.logger.warning(f"Road ID not found: {road_id}")
            return {}
    
    @lru_cache(maxsize=1024)
    def _get_road_congestion_speed(self, road_id, road_category, road_code, date, time):
        """
        Get road congestion and speed
        
        Args:
            road_id (str): Road ID
            road_category (str): Road category
            road_code (str): Road code
            date (int): Date (YYYYMMDD)
            time (int): Hour (0-23)
            
        Returns:
            tuple: (congestion, speed in m/s)
        """
        hour = time + 1 if time < 23 else 24
        
        congestion = 0.0
        
        try:
            if date in self.traffic_mapping and road_code in self.traffic_mapping[date]:
                if hour in self.traffic_mapping[date][road_code]:
                    congestion = self.traffic_mapping[date][road_code][hour]
        except Exception as e:
            pass
        
        free_speed = 55.0
        if road_category in self.free_speed:
            free_speed = self.free_speed[road_category]
        
        speed_range = self.parameters.get('SpeedRange', 0.8)
        
        if congestion < speed_range:
            speed = free_speed * (1 - congestion)
        else:
            speed = 55.0
        
        speed_mps = speed * 1000 / 3600
        
        return congestion, speed_mps
    
    def _precompute_road_data(self):
        """Precompute and cache commonly used road data"""
        self.road_data_cache = {}
        
        all_road_ids = set()
        for trajectory in self.data['trajectory'].iterrows():
            for road_id in trajectory['PathList']:
                all_road_ids.add(road_id)
        for road_id in all_road_ids:
            try:
                road = self.road_network.loc[road_id]
                self.road_data_cache[road_id] = {
                    'category': road.get('fclass', ''),
                    'code': road.get('Code', ''),
                    'length': road.get('Long_meter', 0.0),
                    'city_code': road.get('CityCode', 0)
                }
            except (KeyError, IndexError):
                pass

    def _get_road_info_fast(self, road_id):
        """Get road information quickly"""
        if road_id in self.road_data_cache:
            return self.road_data_cache[road_id]
        try:
            road = self.road_network.loc[road_id]
            data = {
                'category': road.get('fclass', ''),
                'code': road.get('Code', ''),
                'length': road.get('Long_meter', 0.0),
                'city_code': road.get('CityCode', 0)
            }
            self.road_data_cache[road_id] = data
            return data
        except (KeyError, IndexError):
            return {
                'category': '',
                'code': '',
                'length': 0.0,
                'city_code': 0
            }
    
    def _calculate_travel_time(self, distance, speed):
        """
        Calculate travel time
        
        Args:
            distance (float): Distance in meters
            speed (float): Speed in m/s
            
        Returns:
            int: Travel time in seconds
        """
        if speed <= 0:
            speed = 0.1
        
        travel_time = int(distance / speed)
        return max(1, travel_time)
    
    @lru_cache(maxsize=1024)
    def _get_temperature(self, city_code, date):
        """
        Get ambient temperature
        
        Args:
            city_code (int): City code
            date (int): Date (YYYYMMDD)
            
        Returns:
            float: Temperature in Celsius
        """
        season = self._get_season_from_date(date)
        
        default_temp = 25.0
        
        try:
            if city_code in self.temperature_mapping and season in self.temperature_mapping[city_code]:
                return self.temperature_mapping[city_code][season]
        except Exception as e:
            pass
        
        return default_temp
    
    @lru_cache(maxsize=1024)
    def _get_season_from_date(self, date):
        """
        Determine season from date
        
        Args:
            date (int): Date (YYYYMMDD)
            
        Returns:
            str: Season ('Spring', 'Summer', 'Autumn', 'Winter')
        """
        date_str = str(date)
        month = int(date_str[4:6])
        
        if 3 <= month <= 5:
            return 'Spring'
        elif 6 <= month <= 8:
            return 'Summer'
        elif 9 <= month <= 11:
            return 'Autumn'
        else:
            return 'Winter'

    def _handle_energy_refill(self, vehicle):
        """
        Handle energy refill
        
        Args:
            vehicle (Vehicle): Vehicle object
        
        Returns:
            tuple: (is_feasible, infeasible_reason)
        """
        current_position = self._estimate_vehicle_position(vehicle)
        
        if hasattr(vehicle, 'need_charge') and vehicle.need_charge:
            if random.random() < 0.5:
                station_type = 'CS'
            else:
                station_type = 'BSS'
        elif hasattr(vehicle, 'need_refuel') and vehicle.need_refuel:
            station_type = 'HRS'
        else:
            return True, ""
        station_id, station, distance = self.station_manager.find_nearest_station(
            vehicle_location=current_position,
            vehicle_type=vehicle.vehicle_type,
            need_type=station_type
        )
        
        if not station:
            return False, f"Vehicle {vehicle.vehicle_id} ({vehicle.vehicle_type}) cannot find suitable energy station {station_type}"
        
        station_city_code = station.city_code
        current_date = vehicle.date
        current_hour = vehicle.current_time.hour
        correction_factor = self._get_distance_correction_factor(
            city_code=station_city_code,
            date=current_date,
            hour=current_hour,
            station_id=station_id
        )
        
        corrected_distance = distance * correction_factor
        detour_distance = corrected_distance * 2 * 1000
        detour_time = int(detour_distance / 60)
        vehicle.detour_count += 1
        vehicle.detour_distance += detour_distance
        vehicle.detour_time += detour_time
                
        start_time = vehicle.current_time
        arrival_time = start_time + timedelta(seconds=detour_time // 2)
        vehicle.update_path_execution(
            road_id=f"DETOUR_TO_{station_type}_{station_id}",
            start_time=start_time,
            end_time=arrival_time,
            distance=detour_distance / 2,
            travel_time=detour_time // 2,
            is_detour=True
        )
        
        city_code = station.city_code
        hour = arrival_time.hour + 1 if arrival_time.hour < 23 else 24
        if station_type == 'HRS' and hasattr(vehicle, 'refuel'):
            needed_hydrogen = (1.0 - vehicle.current_soc) * vehicle.capacity
            
            self._track_hydrogen_demand(station_id, needed_hydrogen, vehicle.date)
        
        if station_type == 'CS' and hasattr(vehicle, 'charge'):
            electricity_price = self._get_electricity_price(city_code, hour, vehicle.date)
            service_start, service_end, amount, cost = station.provide_charging_service(
                vehicle=vehicle,
                current_time=arrival_time,
                electricity_price=electricity_price
            )
            
            vehicle.current_time = service_end
            
        elif station_type == 'BSS' and hasattr(vehicle, 'swap_battery'):
            electricity_price = self._get_electricity_price(city_code, hour, vehicle.date)
            
            battery_refresh_interval = self.parameters.get('SwapBatteryFullTime', 2.0)
            
            self._track_battery_demand(station_id, vehicle.vehicle_type, vehicle.date)
            service_start, service_end, amount, cost = station.provide_swap_service(
                vehicle=vehicle,
                current_time=arrival_time,
                electricity_price=electricity_price,
                battery_refresh_interval=battery_refresh_interval
            )
            
            vehicle.current_time = service_end
            
        elif station_type == 'HRS' and hasattr(vehicle, 'refuel'):
            hydrogen_price = self._get_hydrogen_price(city_code, hour)
            hydrogen_refresh_interval = self.parameters.get('HydrogenFuelTime', 4.0)
            
            service_start, service_end, amount, cost = station.provide_refueling_service(
                vehicle=vehicle,
                current_time=arrival_time,
                hydrogen_price=hydrogen_price,
                hydrogen_refresh_interval=hydrogen_refresh_interval
            )
            
            vehicle.current_time = service_end
        
        service_duration = (service_end - service_start).total_seconds()
        if service_duration >= 1200:
            vehicle.continuous_driving_time = 0
        return_start_time = vehicle.current_time
        return_end_time = return_start_time + timedelta(seconds=detour_time // 2)
        
        vehicle.update_path_execution(
            road_id=f"DETOUR_FROM_{station_type}_{station_id}",
            start_time=return_start_time,
            end_time=return_end_time,
            distance=detour_distance / 2,
            travel_time=detour_time // 2,
            is_detour=True
        )
        
        vehicle.current_time = return_end_time
        parking_record = {
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'ParkingLocationID': station_id,
            'ParkingLocationType': station_type,
            'ArrivalTime': arrival_time.strftime('%H:%M:%S'),
            'DepartureTime': service_end.strftime('%H:%M:%S'),
            'ParkingDuration': service_duration,
            'ContinuousDrivingTime': vehicle.continuous_driving_time,
            'IsEnergyRefill': 1,
            'EnergyRefillAmount': amount
        }
        
        vehicle.parking_records.append(parking_record)
        
        return True, ""

    def _handle_rest(self, vehicle):
        """
        Handle vehicle rest
        
        Args:
            vehicle (Vehicle): Vehicle object
        """
        current_position = self._estimate_vehicle_position(vehicle)
        
        rest_area_id, rest_area, distance = self.station_manager.find_nearest_rest_area(
            vehicle_location=current_position,
            rest_areas=self.stations.to_dict('index')
        )
        
        if not rest_area:
            self.logger.warning(f"Vehicle {vehicle.vehicle_id} cannot find rest area, continuing")
            return
        
        rest_area_city_code = rest_area.get('CityCode', 0)
        current_date = vehicle.date
        current_hour = vehicle.current_time.hour
        
        correction_factor = self._get_distance_correction_factor(
            city_code=rest_area_city_code,
            date=current_date,
            hour=current_hour,
            station_id=rest_area_id
        )
        
        corrected_distance = distance * correction_factor
        detour_distance = corrected_distance * 2 * 1000
        detour_time = int(detour_distance / 60)
        vehicle.detour_count += 1
        vehicle.detour_distance += detour_distance
        vehicle.detour_time += detour_time
                
        start_time = vehicle.current_time
        arrival_time = start_time + timedelta(seconds=detour_time // 2)
        
        vehicle.update_path_execution(
            road_id=f"DETOUR_TO_REST_{rest_area_id}",
            start_time=start_time,
            end_time=arrival_time,
            distance=detour_distance / 2,
            travel_time=detour_time // 2,
            is_detour=True
        )
        
        rest_duration = 20 * 60
        vehicle.take_rest(
            rest_location={'ID': rest_area_id, 'StationType': 'Else'},
            rest_start_time=arrival_time,
            rest_duration=rest_duration
        )
        
        return_start_time = arrival_time + timedelta(seconds=rest_duration)
        return_end_time = return_start_time + timedelta(seconds=detour_time // 2)
        
        vehicle.update_path_execution(
            road_id=f"DETOUR_FROM_REST_{rest_area_id}",
            start_time=return_start_time,
            end_time=return_end_time,
            distance=detour_distance / 2,
            travel_time=detour_time // 2,
            is_detour=True
        )
        
        vehicle.current_time = return_end_time
    
    def _estimate_vehicle_position(self, vehicle):
        """
        Estimate vehicle current position
        
        Args:
            vehicle (Vehicle): Vehicle object
            
        Returns:
            tuple: Vehicle position (longitude, latitude)
        """
        if len(vehicle.path_list) > 0:
            road_id = vehicle.path_list[vehicle.current_position]
            road_info = self._get_road_info(road_id)
            
            if road_info and 'geometry' in road_info:
                try:
                    if hasattr(road_info['geometry'], 'geoms'):
                        coords = list(road_info['geometry'].geoms[0].coords)
                        return coords[0]
                    else:
                        coords = list(road_info['geometry'].coords)
                        return coords[0]
                except (AttributeError, IndexError) as e:
                    self.logger.warning(f"Cannot get road {road_id} coordinates: {e}")
                    pass
        
        return (116.397, 39.909)
    
    @lru_cache(maxsize=1024)
    def _get_electricity_price(self, city_code, hour, date):
        """
        Get electricity price
        
        Args:
            city_code (int): City code
            hour (int): Hour (1-24)
            date (int): Date (YYYYMMDD)
            
        Returns:
            float: Electricity price (yuan/kWh)
        """
        season = self._get_season_from_date(date)
        default_price = 0.8
        
        try:
            if city_code in self.electricity_price and hour in self.electricity_price[city_code]:
                return self.electricity_price[city_code][hour]
        except Exception as e:
            pass
        
        return default_price
    
    @lru_cache(maxsize=1024)
    def _get_hydrogen_price(self, city_code, hour):
        """
        Get hydrogen price
        
        Args:
            city_code (int): City code
            hour (int): Hour (1-24)
            
        Returns:
            float: Hydrogen price (yuan/kg)
        """
        default_price = 60.0
        
        try:
            if city_code in self.hydrogen_price and hour in self.hydrogen_price[city_code]:
                return self.hydrogen_price[city_code][hour]
        except Exception as e:
            pass
        
        return default_price
    
    def _get_province_code(self, city_code):
        """
        Get province code
        
        Args:
            city_code (int): City code
            
        Returns:
            int: Province code
        """
        default_province = 110000
        
        try:
            if city_code in self.province_city_map:
                return self.province_city_map[city_code]
        except Exception as e:
            pass
        
        return default_province
    
    @lru_cache(maxsize=1024)
    def _get_vehicle_ghg(self, province_code, vehicle_type):
        """
        Get vehicle GHG emission
        
        Args:
            province_code (int): Province code
            vehicle_type (str): Vehicle type
            
        Returns:
            float: GHG emission (kg/km)
        """
        default_ghg = 0.5
        
        try:
            vehicle_ghg = self.data['vehicle_ghg']
            if province_code in vehicle_ghg and vehicle_type in vehicle_ghg[province_code]:
                return vehicle_ghg[province_code][vehicle_type]
        except Exception as e:
            pass
        
        return default_ghg
    
    def _calculate_total_cost(self):
        """Calculate total cost (normalized to 7-day timespan)"""
        capital_cost = 0.0
        annual_to_weekly = 7.0 / 365.0
        
        for station_id, cs in self.station_manager.charging_stations.items():
            capital_cost += cs.annual_cost * annual_to_weekly
            capital_cost += cs.fcp_count * self.data['charging_posts']['dict']['FCP']['AnnualCostPerPost'] * annual_to_weekly
            capital_cost += cs.scp_count * self.data['charging_posts']['dict']['SCP']['AnnualCostPerPost'] * annual_to_weekly
        
        for station_id, bss in self.station_manager.battery_swap_stations.items():
            capital_cost += bss.annual_cost * annual_to_weekly
            
            for bev_type, count in bss.spare_batteries.items():
                if bev_type in self.data['vehicles']['dict'] and count > 0:
                    battery_annual_cost = self.data['vehicles']['dict'][bev_type]['BatteryCost'] / self.data['vehicles']['dict'][bev_type]['LifeSpanBat']
                    capital_cost += count * battery_annual_cost * annual_to_weekly
        
        for station_id, hrs in self.station_manager.hydrogen_stations.items():
            capital_cost += hrs.annual_cost * annual_to_weekly
        
        for vehicle_id, assignment in self.actual_vehicle_map.items():
            if isinstance(assignment, dict):
                actual_vehicle_id = assignment['ActualVehicleID']
                vehicle_type = assignment['Type']
            else:
                actual_vehicle_id = assignment
                vehicle_type = self._get_vehicle_type(vehicle_id, None)
            
            if actual_vehicle_id not in self.used_vehicles:
                continue
            
            if vehicle_type and vehicle_type in self.data['vehicles']['dict']:
                vehicle_data = self.data['vehicles']['dict'][vehicle_type]
                
                annual_base_cost = vehicle_data['VehicleCost'] / vehicle_data['LifeSpanVe']
                
                annual_extra_battery_cost = 0.0
                if vehicle_type.startswith('BEV_'):
                    annual_extra_battery_cost = 2 * vehicle_data['BatteryCost'] / vehicle_data['LifeSpanVe']
                
                vehicle_weekly_cost = (annual_base_cost + annual_extra_battery_cost) * annual_to_weekly
                capital_cost += vehicle_weekly_cost
                self.used_vehicles.remove(actual_vehicle_id)
        
        energy_cost = sum(d['EnergyRefillCost'] for d in self.simulation_results['driving_statistics'])
        
        self.simulation_results['cost_records'] = {
            'capital_cost': capital_cost,
            'energy_cost': energy_cost,
            'total_cost': capital_cost + energy_cost
        }
    
    def _calculate_total_ghg(self):
        """Calculate total GHG emission"""
        infrastructure_ghg = 0.0
        
        for station_id, cs in self.station_manager.charging_stations.items():
            infrastructure_ghg += cs.annual_ghg * 7 / 365
            infrastructure_ghg += cs.fcp_count * self.data['charging_posts']['dict']['FCP']['AnnualGHG'] * 7 / 365
            infrastructure_ghg += cs.scp_count * self.data['charging_posts']['dict']['SCP']['AnnualGHG'] * 7 / 365
        
        for station_id, bss in self.station_manager.battery_swap_stations.items():
            infrastructure_ghg += bss.annual_ghg * 7 / 365
            
            for bev_type, count in bss.spare_batteries.items():
                city_code = bss.city_code
                province_code = self._get_province_code(city_code)
                
                try:
                    if province_code in self.data['battery_ghg'] and bev_type in self.data['battery_ghg'][province_code]:
                        ghgs = self.data['battery_ghg'][province_code][bev_type]
                        capacity = self.data['vehicles']['dict'][bev_type]['Capacity']
                        lifespan = self.data['vehicles']['dict'][bev_type]['LifeSpanBat']
                        
                        battery_ghg = ghgs * capacity / lifespan * (7 / 365)
                        infrastructure_ghg += count * battery_ghg
                except Exception as e:
                    self.logger.warning(f"Battery GHG calculation failed: {e}")
        
        for station_id, hrs in self.station_manager.hydrogen_stations.items():
            infrastructure_ghg += hrs.annual_ghg * 7 / 365
        
        vehicle_ghg = sum(d['GHGEmission'] for d in self.simulation_results['driving_statistics'])
        
        self.simulation_results['ghg_records'] = {
            'infrastructure_ghg': infrastructure_ghg,
            'vehicle_ghg': vehicle_ghg,
            'total_ghg': infrastructure_ghg + vehicle_ghg
        }
    
    def _log_simulation_statistics(self):
        """Output simulation statistics"""
        if not hasattr(self, 'simulation_results'):
            self.logger.warning("Cannot output statistics: simulation_results not found")
            return
    
        vehicle_count = len(self.simulation_results.get('vehicle_records', []))
        
        path_execution_count = len(self.simulation_results.get('path_execution', []))
        for memmap_name in self.simulation_results.get('path_execution_memmaps', []):
            if memmap_name in self.memmap_files:
                path_execution_count += self.memmap_files[memmap_name]['shape'][0]
        
        parking_count = len(self.simulation_results.get('parking_records', []))
        for memmap_name in self.simulation_results.get('parking_records_memmaps', []):
            if memmap_name in self.memmap_files:
                parking_count += self.memmap_files[memmap_name]['shape'][0]
        
        cost_records = self.simulation_results.get('cost_records', {})
        ghg_records = self.simulation_results.get('ghg_records', {})
        
        self.logger.info(f"Simulation statistics:")
        self.logger.info(f"  - Vehicles: {vehicle_count}")
        self.logger.info(f"  - Path executions: {path_execution_count}")
        self.logger.info(f"  - Parking records: {parking_count}")
        self.logger.info(f"  - Total cost: {cost_records.get('total_cost', 0):.2f} yuan")
        self.logger.info(f"  - Total GHG: {ghg_records.get('total_ghg', 0):.2f} kg")
    
    def export_results(self, output_dir=None):
        """
        Export simulation results
        
        Args:
            output_dir (str, optional): Output directory, use config default if not specified
        """
        if output_dir is None:
            output_dir = self.config['path_config']['output_directory']
        
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        vehicle_records_df = pd.DataFrame(self.simulation_results['vehicle_records'])
        if not vehicle_records_df.empty:
            vehicle_records_df.to_csv(output_path / '01ElectrifiedVehicle.txt', sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Vehicle assignment results saved to {output_path / '01ElectrifiedVehicle.txt'}")
        station_records = []
        for station_id, station_info in self.stations.iterrows():
            record = {
                'ID': station_info['ID'],
                'LinkRoad': station_info['LinkRoad'],
                'Lon': station_info['Lon'],
                'Lat': station_info['Lat'],
                'CS': 0,
                'FCP': 0,
                'SCP': 0,
                'BSS': 0,
                'HRS_01': 0,
                'HRS_02': 0,
                'HRS_03': 0
            }
            
            for vehicle_type in self.data['vehicles']['df']['Type']:
                if vehicle_type.startswith('BEV_'):
                    record[vehicle_type] = 0
            
            if station_id in self.station_manager.charging_stations:
                cs = self.station_manager.charging_stations[station_id]
                record['CS'] = 1
                record['FCP'] = cs.fcp_count
                record['SCP'] = cs.scp_count
            
            if station_id in self.station_manager.battery_swap_stations:
                bss = self.station_manager.battery_swap_stations[station_id]
                record['BSS'] = 1
                
                for bev_type, count in bss.spare_batteries.items():
                    if bev_type in record:
                        record[bev_type] = count
            
            if station_id in self.station_manager.hydrogen_stations:
                hrs = self.station_manager.hydrogen_stations[station_id]
                hrs_type = hrs.hrs_type
                if hrs_type in record:
                    record[hrs_type] = 1
            
            station_records.append(record)
        
        if station_records:
            station_df = pd.DataFrame(station_records)
            station_df.to_csv(output_path / '02Station.txt', sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Station configuration saved to {output_path / '02Station.txt'}")
        
        path_execution_records = self.simulation_results['path_execution'].copy()
        
        for memmap_name in self.simulation_results['path_execution_memmaps']:
            try:
                path_memmap = self._get_memmap(memmap_name)
                if path_memmap is None:
                    continue
                
                metadata_file = self.memmap_dir / f"{memmap_name}_metadata.json"
                if not os.path.exists(metadata_file):
                    self.logger.warning(f"Path execution metadata file not found: {metadata_file}")
                    continue
                
                with open(metadata_file, 'r') as f:
                    path_metadata = json.load(f)
                
                for i in range(path_memmap.shape[0]):
                    record = {
                        'VehicleID': int(path_memmap[i, 0]),
                        'Distance': float(path_memmap[i, 1]),
                        'TravelTime': float(path_memmap[i, 2]),
                        'IsDetour': bool(path_memmap[i, 3]),
                        'RoadID': path_metadata[i]['RoadID'],
                        'StartTime': path_metadata[i]['StartTime'],
                        'EndTime': path_metadata[i]['EndTime']
                    }
                    path_execution_records.append(record)
            except Exception as e:
                self.logger.warning(f"Failed to process memmap path execution records: {e}")
        
        path_execution_df = pd.DataFrame(path_execution_records)
        if not path_execution_df.empty:
            path_execution_df.to_csv(output_path / '03PathExecution.txt', sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Path execution records saved to {output_path / '03PathExecution.txt'}")

        objective_records = [{
            'TotalCost': self.simulation_results['cost_records']['total_cost'],
            'TotalTravelTime': sum(d['TotalDrivingTime'] for d in self.simulation_results['driving_statistics']),
            'TotalGHG': self.simulation_results['ghg_records']['total_ghg']
        }]
        
        objective_df = pd.DataFrame(objective_records)
        objective_df.to_csv(output_path / '04Objective.txt', sep='\t', index=False, encoding='utf-8')
        self.logger.info(f"Objective values saved to {output_path / '04Objective.txt'}")
        algorithm_records = [{
            'Parameter': 'SimulatedAnnealing_InitialTemperature',
            'Value': self.config['optimization']['simulated_annealing']['initial_temperature']
        }, {
            'Parameter': 'SimulatedAnnealing_CoolingRate',
            'Value': self.config['optimization']['simulated_annealing']['cooling_rate']
        }, {
            'Parameter': 'SimulatedAnnealing_MinTemperature',
            'Value': self.config['optimization']['simulated_annealing']['min_temperature']
        }, {
            'Parameter': 'ObjectiveWeight_Cost',
            'Value': self.config['optimization']['objective_weights']['cost']
        }, {
            'Parameter': 'ObjectiveWeight_TravelTime',
            'Value': self.config['optimization']['objective_weights']['travel_time']
        }, {
            'Parameter': 'ObjectiveWeight_GHG',
            'Value': self.config['optimization']['objective_weights']['ghg']
        }]
        
        algorithm_df = pd.DataFrame(algorithm_records)
        algorithm_df.to_csv(output_path / '05AlgorithmRelated.txt', sep='\t', index=False, encoding='utf-8')
        self.logger.info(f"Algorithm parameters saved to {output_path / '05AlgorithmRelated.txt'}")
        
        parking_records = self.simulation_results['parking_records'].copy()
    
        for memmap_name in self.simulation_results['parking_records_memmaps']:
            try:
                parking_memmap = self._get_memmap(memmap_name)
                if parking_memmap is None:
                    continue
                
                metadata_file = self.memmap_dir / f"{memmap_name}_metadata.json"
                if not os.path.exists(metadata_file):
                    self.logger.warning(f"Parking metadata file not found: {metadata_file}")
                    continue
                
                with open(metadata_file, 'r') as f:
                    parking_metadata = json.load(f)
                
                for i in range(parking_memmap.shape[0]):
                    record = {
                        'VehicleID': int(parking_memmap[i, 0]),
                        'ParkingDuration': float(parking_memmap[i, 1]),
                        'ContinuousDrivingTime': float(parking_memmap[i, 2]),
                        'IsEnergyRefill': bool(parking_memmap[i, 3]),
                        'EnergyRefillAmount': float(parking_memmap[i, 4]),
                        'ParkingLocationID': parking_metadata[i]['ParkingLocationID'],
                        'ParkingLocationType': parking_metadata[i]['ParkingLocationType'],
                        'ArrivalTime': parking_metadata[i]['ArrivalTime'],
                        'DepartureTime': parking_metadata[i]['DepartureTime']
                    }
                    parking_records.append(record)
            except Exception as e:
                self.logger.warning(f"Failed to process memmap parking records: {e}")
        
        parking_records_df = pd.DataFrame(parking_records)
        if not parking_records_df.empty:
            parking_records_df.to_csv(output_path / '06ParkingRecord.txt', sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Parking records saved to {output_path / '06ParkingRecord.txt'}")
        
        driving_stats_df = pd.DataFrame(self.simulation_results['driving_statistics'])
        if not driving_stats_df.empty:
            driving_stats_df.to_csv(output_path / '07DrivingTimeStatistics.txt', sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Driving statistics saved to {output_path / '07DrivingTimeStatistics.txt'}")

        station_service_records = self.simulation_results.get('station_service_records', [])
        if station_service_records:
            station_service_df = pd.DataFrame(station_service_records)
            
            columns_order = [
                'Timestamp', 'EventType', 'StationID', 'StationType', 'VehicleID', 
                'ActualVehicleID', 'VehicleType', 'QueueLength', 'WaitingTime', 
                'ServiceTime', 'EnergyAmount', 'ServiceCost', 'ChargerType', 
                'BatteryType', 'HydrogenType', 'StationStatus'
            ]
            
            for col in columns_order:
                if col not in station_service_df.columns:
                    station_service_df[col] = ''
            
            station_service_df = station_service_df[columns_order]
            
            station_service_df.to_csv(output_path / '08StationServiceRecords.txt', 
                                    sep='\t', index=False, encoding='utf-8')
            self.logger.info(f"Station service records saved to {output_path / '08StationServiceRecords.txt'}, {len(station_service_records)} records")
        else:
            self.logger.warning("No station service records to export")

        self._cleanup_memmaps()
        
    def __del__(self):
        """Destructor to ensure memmap cleanup"""
        try:
            if hasattr(self, '_cleanup_memmaps'):
                self._cleanup_memmaps()
        except Exception:
            pass

    def _track_battery_demand(self, station_id, vehicle_type, date):
        """
        Track battery demand
        
        Args:
            station_id (str): Battery swap station ID
            vehicle_type (str): Vehicle type
            date (int): Date
        """
        if station_id not in self.battery_demand_tracker[date]:
            self.battery_demand_tracker[date][station_id] = defaultdict(int)
        
        self.battery_demand_tracker[date][station_id][vehicle_type] += 1

    def _finalize_battery_requirements(self):
        """
        Finalize battery requirements based on 7-day maximum demand
        """
        self.logger.info("Calculating battery requirements...")
        
        station_battery_requirements = defaultdict(lambda: defaultdict(int))
        
        for date, stations in self.battery_demand_tracker.items():
            for station_id, battery_types in stations.items():
                for battery_type, demand in battery_types.items():
                    current_max = station_battery_requirements[station_id][battery_type]
                    station_battery_requirements[station_id][battery_type] = max(current_max, demand)
        
        for station_id, battery_requirements in station_battery_requirements.items():
            if station_id in self.station_manager.battery_swap_stations:
                bss = self.station_manager.battery_swap_stations[station_id]
                bss.spare_batteries = dict(battery_requirements)
                self.logger.info(f"Station {station_id} battery requirements: {dict(battery_requirements)}")
        
        total_battery_requirements = defaultdict(int)
        for station_id, bss in self.station_manager.battery_swap_stations.items():
            for battery_type, count in bss.spare_batteries.items():
                total_battery_requirements[battery_type] += count
        
        self.logger.info(f"Total battery requirements: {dict(total_battery_requirements)}")

    def _track_hydrogen_demand(self, station_id, hydrogen_amount, date):
        """
        Track hydrogen demand
        
        Args:
            station_id (str): Hydrogen station ID
            hydrogen_amount (float): Hydrogen consumption (kg)
            date (int): Date
        """
        self.hydrogen_demand_tracker[date][station_id] += hydrogen_amount

    def _finalize_hydrogen_requirements(self):
        """
        Analyze hydrogen demand statistics after simulation (for analysis only, does not affect cost calculation)
        """
        self.logger.info("Analyzing hydrogen demand statistics...")
        
        station_hydrogen_requirements = defaultdict(float)
        
        for date, stations in self.hydrogen_demand_tracker.items():
            for station_id, hydrogen_used in stations.items():
                current_max = station_hydrogen_requirements[station_id]
                station_hydrogen_requirements[station_id] = max(current_max, hydrogen_used)
        
        for station_id, hydrogen_requirement in station_hydrogen_requirements.items():
            if station_id in self.station_manager.hydrogen_stations:
                self.logger.info(f"Station {station_id} max daily hydrogen demand: {hydrogen_requirement:.2f} kg")
        
        total_hydrogen_requirement = sum(station_hydrogen_requirements.values())
        
        self.logger.info(f"Total max daily hydrogen demand: {total_hydrogen_requirement:.2f} kg")

    def _collect_station_service_records(self):
        """
        Collect service records from all stations
        """
        self.logger.info("Collecting station service records...")
        
        all_service_records = []
        
        for station_id, station in self.station_manager.charging_stations.items():
            all_service_records.extend(station.service_records)
        
        for station_id, station in self.station_manager.battery_swap_stations.items():
            all_service_records.extend(station.service_records)
        
        for station_id, station in self.station_manager.hydrogen_stations.items():
            all_service_records.extend(station.service_records)
        
        all_service_records.sort(key=lambda x: x['Timestamp'])
        
        self.simulation_results['station_service_records'] = all_service_records
        
        self.logger.info(f"Collected {len(all_service_records)} station service records")