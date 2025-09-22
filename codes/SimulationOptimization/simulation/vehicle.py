"""
Author: Shiqi (Anya) WANG
Date: 2025/5/21
Description: Vehicle behavior simulation module
Responsible for simulating driving, charging/refueling and resting behaviors of Battery Electric Vehicles (BEV) and Hydrogen Fuel Cell Vehicles (HFCV)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import math
from pathlib import Path


class Vehicle:
    """Base vehicle class that defines basic attributes and methods for vehicles"""
    
    def __init__(self, vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id, config, parameters, logger=None):
        """
        Initialize vehicle
        
        Args:
            vehicle_id (int): Vehicle trip ID
            trajectory_info (dict): Vehicle trajectory information
            vehicle_type (str): Vehicle type (e.g., 'BEV_01', 'HFCV_03')
            actual_vehicle_id (str): Actual vehicle ID (for tracking multiple trips of the same physical vehicle)
            config (dict): Configuration dictionary
            parameters (dict): Model parameters dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.vehicle_id = vehicle_id
        self.trajectory_info = trajectory_info
        self.vehicle_type = vehicle_type
        self.actual_vehicle_id = actual_vehicle_id
        self.config = config
        self.parameters = parameters
        self.logger = logger or logging.getLogger(__name__)
        
        self.date = trajectory_info['Date']
        self.start_time = trajectory_info['Time']
        self.path_list = trajectory_info['PathList']
        self.cargo_weight = trajectory_info['核定载质量']
        
        self.current_position = 0
        self.current_soc = 1.0
        self.current_time = self._parse_datetime(self.date, self.start_time)
        self.total_distance = 0.0
        self.total_travel_time = 0
        self.continuous_driving_time = 0
        self.rest_count = 0
        self.total_rest_time = 0
        
        self.energy_refill_count = 0
        self.energy_refill_time = 0
        self.energy_refill_amount = 0.0
        self.energy_refill_cost = 0.0
        
        self.detour_count = 0
        self.detour_distance = 0.0
        self.detour_time = 0
        
        self.ghg_emission = 0.0
        
        self.path_execution = []
        
        self.parking_records = []
    
    def _parse_datetime(self, date, time_value):
        """
        Parse date and time values into datetime object
        
        Args:
            date (int): Date in format YYYYMMDD
            time_value (int): Time in format HHMMSS
        
        Returns:
            datetime: Parsed datetime object
        """
        date_str = str(date)
        time_str = str(time_value).zfill(6)
        
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        return datetime(year, month, day, hour, minute, second)
    
    def update_soc(self, energy_consumption):
        """
        Update State of Charge (SOC) for battery/hydrogen
        
        Args:
            energy_consumption (float): Energy consumption (kWh/kg)
        """
        pass
    
    def check_rest_needed(self):
        """
        Check if rest is needed
        
        Returns:
            bool: Whether rest is needed
        """
        max_driving_hours = self.config['simulation']['rest']['max_driving_hours']
        max_driving_seconds = max_driving_hours * 3600
        
        return self.continuous_driving_time >= max_driving_seconds
    
    def take_rest(self, rest_location, rest_start_time, rest_duration=None):
        """
        Execute rest behavior
        
        Args:
            rest_location (dict): Rest location information
            rest_start_time (datetime): Rest start time
            rest_duration (int, optional): Rest duration in seconds, uses minimum rest time if not specified
        """
        if rest_duration is None:
            min_rest_minutes = self.config['simulation']['rest']['min_rest_time']
            rest_duration = min_rest_minutes * 60
        
        self.rest_count += 1
        self.total_rest_time += rest_duration
        
        self.continuous_driving_time = 0
        
        rest_end_time = rest_start_time + timedelta(seconds=rest_duration)
        self.current_time = rest_end_time
        parking_record = {
            'VehicleID': self.vehicle_id,
            'ActualVehicleID': self.actual_vehicle_id,
            'ParkingLocationID': rest_location.get('ID', 0),
            'ParkingLocationType': rest_location.get('StationType', 'Else'),
            'ArrivalTime': rest_start_time.strftime('%H:%M:%S'),
            'DepartureTime': rest_end_time.strftime('%H:%M:%S'),
            'ParkingDuration': rest_duration,
            'ContinuousDrivingTime': self.continuous_driving_time,
            'IsEnergyRefill': 0,
            'EnergyRefillAmount': 0.0
        }
        
        self.parking_records.append(parking_record)
    
    def update_path_execution(self, road_id, start_time, end_time, distance, travel_time, is_detour=False):
        """
        Record path execution information
        
        Args:
            road_id (str): Road ID
            start_time (datetime): Start time
            end_time (datetime): End time
            distance (float): Distance in meters
            travel_time (int): Travel time in seconds
            is_detour (bool): Whether this is a detour path
        """
        self.path_execution.append({
            'VehicleID': self.vehicle_id,
            'ActualVehicleID': self.actual_vehicle_id,
            'RoadID': road_id,
            'StartTime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EndTime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Distance': distance,
            'TravelTime': travel_time,
            'IsDetour': is_detour
        })
    
    def get_driving_statistics(self):
        """
        Get driving statistics
        
        Returns:
            dict: Driving statistics information
        """
        avg_rest_duration = 0 if self.rest_count == 0 else self.total_rest_time / self.rest_count
        
        return {
            'VehicleID': self.vehicle_id,
            'ActualVehicleID': self.actual_vehicle_id,
            'TotalDrivingTime': self.total_travel_time,
            'TotalRestTime': self.total_rest_time,
            'RestCount': self.rest_count,
            'AverageRestDuration': avg_rest_duration,
            'MaxContinuousDrivingTime': self.continuous_driving_time,
            'TotalDistance': self.total_distance,
            'EnergyRefillCount': self.energy_refill_count,
            'EnergyRefillTime': self.energy_refill_time,
            'EnergyRefillAmount': self.energy_refill_amount,
            'EnergyRefillCost': self.energy_refill_cost,
            'DetourCount': self.detour_count,
            'DetourDistance': self.detour_distance,
            'DetourTime': self.detour_time,
            'GHGEmission': self.ghg_emission
        }


class BEV(Vehicle):
    """
    Battery Electric Vehicle (BEV) class for simulating electric vehicle behavior
    """
    
    def __init__(self, vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id, 
                 config, parameters, vehicle_data, logger=None):
        """
        Initialize Battery Electric Vehicle
        
        Args:
            vehicle_id (int): Vehicle trip ID
            trajectory_info (dict): Vehicle trajectory information
            vehicle_type (str): Vehicle type (e.g., 'BEV_01')
            actual_vehicle_id (str): Actual vehicle ID
            config (dict): Configuration dictionary
            parameters (dict): Model parameters dictionary
            vehicle_data (dict): Vehicle type data
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id, 
                        config, parameters, logger)
        
        self.vehicle_data = vehicle_data['dict'][vehicle_type]
        self.capacity = self.vehicle_data['Capacity']
        
        self.min_soc = config['simulation']['bev']['min_soc']
        self.need_charge = False
    
    def update_soc(self, energy_consumption_kwh):
        """
        Update battery State of Charge (SOC)
        
        Args:
            energy_consumption_kwh (float): Energy consumption in kWh
        
        Returns:
            float: Updated SOC
        """
        soc_decrease = energy_consumption_kwh / self.capacity
        
        self.current_soc -= soc_decrease
        
        if self.current_soc <= self.min_soc:
            self.need_charge = True
        
        return self.current_soc
    
    def charge(self, charging_station, charging_type, electricity_price):
        """
        Electric vehicle charging
        
        Args:
            charging_station (dict): Charging station information
            charging_type (str): Charging type ('FCP' or 'SCP')
            electricity_price (float): Electricity price (yuan/kWh)
        
        Returns:
            tuple: (charging_time (seconds), charge_amount (kWh), charging_cost (yuan))
        """
        power = charging_station[charging_type]['Power']
        efficiency = charging_station[charging_type]['Efficiency']
        
        needed_charge = (1.0 - self.current_soc) * self.capacity
        
        charging_time = (needed_charge / (power * efficiency)) * 3600
        
        charging_cost = needed_charge * electricity_price
        
        self.current_soc = 1.0
        self.need_charge = False
        
        self.energy_refill_count += 1
        self.energy_refill_time += charging_time
        self.energy_refill_amount += needed_charge
        self.energy_refill_cost += charging_cost
        
        return charging_time, needed_charge, charging_cost
    
    def swap_battery(self, battery_swap_station, spare_battery, electricity_price):
        """
        Electric vehicle battery swapping
        
        Args:
            battery_swap_station (dict): Battery swap station information
            spare_battery (dict): Spare battery information
            electricity_price (float): Electricity price (yuan/kWh)
        
        Returns:
            tuple: (swap_time (seconds), gained_charge (kWh), swap_cost (yuan))
        """
        swap_time = battery_swap_station['ServiceTimeSeconds']
        
        gained_charge = (1.0 - self.current_soc) * self.capacity
        
        swap_cost = gained_charge * electricity_price
        
        self.current_soc = 1.0
        self.need_charge = False
        
        self.energy_refill_count += 1
        self.energy_refill_time += swap_time
        self.energy_refill_amount += gained_charge
        self.energy_refill_cost += swap_cost
        
        return swap_time, gained_charge, swap_cost
    
    def calculate_energy_consumption(self, distance, time, temperature):
        """
        Calculate energy consumption
        
        Args:
            distance (float): Travel distance in meters
            time (int): Travel time in seconds
            temperature (float): Ambient temperature in Celsius
        
        Returns:
            float: Energy consumption in kWh
        """
        if time <= 0:
            time = 1
        distance_km = distance / 1000
        time_hour = time / 3600
        v = distance_km / time_hour
        
        if v <= 0:
            v = 1.0
        
        AC = 1 if temperature > 25.0 else 0
        
        H = 1 if temperature < 5.0 else 0
        
        current_hour = self.current_time.hour
        N = 1 if (current_hour >= 19 or current_hour < 7) else 0
        
        try:
            base_consumption = 0.372 - 0.076 * math.log(v) + 0.044 * AC + 0.114 * H + 0.007 * N
        except ValueError:
            base_consumption = 0.372 - 0.076 * math.log(1.0) + 0.044 * AC + 0.114 * H + 0.007 * N
        
        temp_factor = self.get_temp_factor_bev(temperature)
        
        energy_consumption = base_consumption * temp_factor * distance_km
        
        return max(0.0, energy_consumption)
    
    def get_temp_factor_bev(self, temperature):
        """
        Get BEV temperature influence factor
        
        Args:
            temperature (float): Ambient temperature in Celsius
        
        Returns:
            float: Temperature influence factor
        """
        if temperature < -10:
            return 1.066
        elif temperature < 0:
            return 1.034
        elif temperature < 10:
            return 1.017
        elif temperature <= 35:
            return 1.0
        else:
            return 1.003


class HFCV(Vehicle):
    """
    Hydrogen Fuel Cell Vehicle (HFCV) class for simulating hydrogen fuel cell vehicle behavior
    """
    
    def __init__(self, vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id, 
                 config, parameters, vehicle_data, logger=None):
        """
        Initialize Hydrogen Fuel Cell Vehicle
        
        Args:
            vehicle_id (int): Vehicle trip ID
            trajectory_info (dict): Vehicle trajectory information
            vehicle_type (str): Vehicle type (e.g., 'HFCV_01')
            actual_vehicle_id (str): Actual vehicle ID
            config (dict): Configuration dictionary
            parameters (dict): Model parameters dictionary
            vehicle_data (dict): Vehicle type data
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(vehicle_id, trajectory_info, vehicle_type, actual_vehicle_id, 
                        config, parameters, logger)
        
        self.vehicle_data = vehicle_data['dict'][vehicle_type]
        self.capacity = self.vehicle_data['Capacity']
        
        self.min_soc = config['simulation']['hfcv']['min_soc']
        self.need_refuel = False
        
        self.hy_to_elec = parameters['HytoElec']
    
    def update_soc(self, energy_consumption_kg):
        """
        Update hydrogen State of Charge (SOC)
        
        Args:
            energy_consumption_kg (float): Hydrogen consumption in kg
        
        Returns:
            float: Updated SOC
        """
        soc_decrease = energy_consumption_kg / self.capacity
        
        self.current_soc -= soc_decrease
        
        if self.current_soc <= self.min_soc:
            self.need_refuel = True
        
        return self.current_soc
    
    def refuel(self, hydrogen_station, hydrogen_price):
        """
        Hydrogen fuel cell vehicle refueling
        
        Args:
            hydrogen_station (dict): Hydrogen station information
            hydrogen_price (float): Hydrogen price (yuan/kg)
        
        Returns:
            tuple: (refuel_time (seconds), hydrogen_amount (kg), refuel_cost (yuan))
        """
        refuel_time = hydrogen_station['ServiceTimeSeconds']
        
        needed_hydrogen = (1.0 - self.current_soc) * self.capacity
        
        new_soc = 1.0
        
        refuel_cost = needed_hydrogen * hydrogen_price
        
        self.current_soc = new_soc
        if self.current_soc > self.min_soc * 1.5:
            self.need_refuel = False
        
        self.energy_refill_count += 1
        self.energy_refill_time += refuel_time
        self.energy_refill_amount += needed_hydrogen
        self.energy_refill_cost += refuel_cost
        
        return refuel_time, needed_hydrogen, refuel_cost
    
    def calculate_energy_consumption(self, distance, time, temperature):
        """
        Calculate energy consumption
        
        Args:
            distance (float): Travel distance in meters
            time (int): Travel time in seconds
            temperature (float): Ambient temperature in Celsius
        
        Returns:
            float: Hydrogen consumption in kg
        """
        if time <= 0:
            time = 1
        distance_km = distance / 1000
        time_hour = time / 3600
        v = distance_km / time_hour
        
        if v <= 0:
            v = 1.0
        
        AC = 1 if temperature > 25.0 else 0
        
        H = 1 if temperature < 5.0 else 0
        
        current_hour = self.current_time.hour
        N = 1 if (current_hour >= 19 or current_hour < 7) else 0
        
        try:
            base_consumption = 0.372 - 0.076 * math.log(v) + 0.044 * AC + 0.114 * H + 0.007 * N
        except ValueError:
            base_consumption = 0.372 - 0.076 * math.log(1.0) + 0.044 * AC + 0.114 * H + 0.007 * N
        
        temp_factor = self.get_temp_factor_hfcv(temperature)
        
        electric_consumption = base_consumption * temp_factor * distance_km
        
        hydrogen_consumption = electric_consumption / self.hy_to_elec
        
        return max(0.0, hydrogen_consumption)
    
    def get_temp_factor_hfcv(self, temperature):
        """
        Get HFCV temperature influence factor
        
        Args:
            temperature (float): Ambient temperature in Celsius
        
        Returns:
            float: Temperature influence factor
        """
        if temperature < -10:
            return 1.042
        elif temperature < 0:
            return 1.024
        elif temperature < 10: 
            return 1.012
        elif temperature <= 35:
            return 1.0
        else:
            return 1.002