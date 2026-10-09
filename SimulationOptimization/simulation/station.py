"""
Author: Shiqi (Anya) WANG
Date: 2025/5/21
Description: Station service simulation module
Responsible for simulating the service processes of charging stations, battery swap stations, and hydrogen stations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import heapq
from collections import deque, defaultdict
from rtree import index


class StationManager:
    """Station manager that manages all types of stations"""
    
    def __init__(self, config, logger=None):
        """
        Initialize station manager
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.charging_stations = {}
        self.battery_swap_stations = {}
        self.hydrogen_stations = {}
        
        self.slow_charging_posts = 0
        self.fast_charging_posts = 0
        
        self.spare_batteries = defaultdict(int)
        
        self.service_records = []
        
        self.spatial_index = index.Index()
        self.station_locations = {}
    
    def add_charging_station(self, station_id, station_info, cs_type, fcp_count, scp_count, cs_data, cp_data):
        """
        Add charging station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            cs_type (str): Charging station type (e.g., 'CS_01')
            fcp_count (int): Fast charging post count
            scp_count (int): Slow charging post count
            cs_data (dict): Charging station type data
            cp_data (dict): Charging post type data
        """
        if station_id in self.charging_stations:
            self.logger.warning(f"Charging station {station_id} already exists and will be overwritten")
        
        cs = ChargingStation(
            station_id=station_id,
            station_info=station_info,
            cs_type=cs_type,
            fcp_count=fcp_count,
            scp_count=scp_count,
            cs_data=cs_data,
            cp_data=cp_data,
            logger=self.logger
        )
        
        self.charging_stations[station_id] = cs
        
        self.fast_charging_posts += fcp_count
        self.slow_charging_posts += scp_count
        
        lon, lat = station_info['Lon'], station_info['Lat']
        self.station_locations[station_id] = (lon, lat)
        self.spatial_index.insert(int(station_id), (lon, lat, lon, lat), obj={'type': 'CS'})
        
        self.logger.debug(f"Added charging station {station_id}, type: {cs_type}, FCP: {fcp_count}, SCP: {scp_count}")
    
    def add_battery_swap_station(self, station_id, station_info, bss_type, spare_batteries, bss_data, logger=None):
        """
        Add battery swap station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            bss_type (str): Battery swap station type (e.g., 'BSS_01')
            spare_batteries (dict): Spare battery count by type
            bss_data (dict): Battery swap station type data
            logger (logging.Logger, optional): Logger instance
        """
        if station_id in self.battery_swap_stations:
            self.logger.warning(f"Battery swap station {station_id} already exists and will be overwritten")
        
        bss = BatterySwapStation(
            station_id=station_id,
            station_info=station_info,
            bss_type=bss_type,
            spare_batteries=spare_batteries,
            bss_data=bss_data,
            logger=self.logger
        )
        
        self.battery_swap_stations[station_id] = bss
        
        for bev_type, count in spare_batteries.items():
            self.spare_batteries[bev_type] += count
        
        battery_count = sum(spare_batteries.values())
        
        lon, lat = station_info['Lon'], station_info['Lat']
        self.station_locations[station_id] = (lon, lat)
        self.spatial_index.insert(int(station_id), (lon, lat, lon, lat), obj={'type': 'BSS'})
        
        self.logger.debug(f"Added battery swap station {station_id}, type: {bss_type}, batteries: {battery_count}")
    
    def add_hydrogen_station(self, station_id, station_info, hrs_type, hrs_data, logger=None):
        """
        Add hydrogen station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            hrs_type (str): Hydrogen station type (e.g., 'HRS_01')
            hrs_data (dict): Hydrogen station type data
            logger (logging.Logger, optional): Logger instance
        """
        if station_id in self.hydrogen_stations:
            self.logger.warning(f"Hydrogen station {station_id} already exists and will be overwritten")
        
        hrs = HydrogenStation(
            station_id=station_id,
            station_info=station_info,
            hrs_type=hrs_type,
            hrs_data=hrs_data,
            logger=self.logger
        )
        
        self.hydrogen_stations[station_id] = hrs
        
        lon, lat = station_info['Lon'], station_info['Lat']
        self.station_locations[station_id] = (lon, lat)
        self.spatial_index.insert(int(station_id), (lon, lat, lon, lat), obj={'type': 'HRS'})
        
        self.logger.debug(f"Added hydrogen station {station_id}, type: {hrs_type}")

    def find_nearest_station(self, vehicle_location, vehicle_type, need_type='CS'):
        """
        Find nearest service station using spatial index
        
        Args:
            vehicle_location (tuple): Vehicle location (longitude, latitude)
            vehicle_type (str): Vehicle type (e.g., 'BEV_01', 'HFCV_01')
            need_type (str): Required station type, 'CS'=charging, 'BSS'=battery swap, 'HRS'=hydrogen
        
        Returns:
            tuple: (nearest station ID, station object, distance)
        """
        search_radius = 1.0
        max_radius = 10.0
        
        lon, lat = vehicle_location
        found_stations = []
        
        while search_radius <= max_radius:
            bbox = (lon - search_radius/100, lat - search_radius/100, 
                    lon + search_radius/100, lat + search_radius/100)
            
            nearest_ids = list(self.spatial_index.intersection(bbox))
            
            for station_id in nearest_ids:
                station_id = str(station_id)
                
                if need_type == 'CS' and station_id in self.charging_stations:
                    station = self.charging_stations[station_id]
                    distance = self._calculate_distance(vehicle_location, (station.lon, station.lat))
                    found_stations.append((station_id, station, distance))
                    
                elif need_type == 'BSS' and station_id in self.battery_swap_stations:
                    station = self.battery_swap_stations[station_id]
                    distance = self._calculate_distance(vehicle_location, (station.lon, station.lat))
                    found_stations.append((station_id, station, distance))
                        
                elif need_type == 'HRS' and station_id in self.hydrogen_stations:
                    station = self.hydrogen_stations[station_id]
                    distance = self._calculate_distance(vehicle_location, (station.lon, station.lat))
                    found_stations.append((station_id, station, distance))
            
            if found_stations:
                found_stations.sort(key=lambda x: x[2])
                return found_stations[0]
            
            search_radius *= 2
        
        return None, None, float('inf')
    
    def find_nearest_rest_area(self, vehicle_location, rest_areas):
        """
        Find nearest rest area
        
        Args:
            vehicle_location (tuple): Vehicle location (longitude, latitude)
            rest_areas (dict): Rest area dictionary
        
        Returns:
            tuple: (nearest rest area ID, rest area info, distance)
        """
        closest_id = None
        closest_area = None
        closest_distance = float('inf')
        
        for area_id, area_info in rest_areas.items():
            area_location = (area_info['Lon'], area_info['Lat'])
            distance = self._calculate_distance(vehicle_location, area_location)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_id = area_id
                closest_area = area_info
        
        return closest_id, closest_area, closest_distance
    
    def _calculate_distance(self, point1, point2):
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            point1 (tuple): First point coordinates (longitude, latitude)
            point2 (tuple): Second point coordinates (longitude, latitude)
        
        Returns:
            float: Distance between points (kilometers)
        """
        R = 6371.0
        
        lon1, lat1 = np.radians(point1)
        lon2, lat2 = np.radians(point2)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def get_station_stats(self):
        """
        Get statistics for all stations
        
        Returns:
            dict: Station statistics
        """
        stats = {
            'charging_stations': len(self.charging_stations),
            'fast_charging_posts': self.fast_charging_posts,
            'slow_charging_posts': self.slow_charging_posts,
            'battery_swap_stations': len(self.battery_swap_stations),
            'spare_batteries': dict(self.spare_batteries),
            'hydrogen_stations': len(self.hydrogen_stations),
            'service_records': len(self.service_records)
        }
        
        return stats


class ChargingStation:
    """Charging station class that simulates charging service processes"""
    
    def __init__(self, station_id, station_info, cs_type, fcp_count, scp_count, cs_data, cp_data, logger=None):
        """
        Initialize charging station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            cs_type (str): Charging station type (e.g., 'CS_01')
            fcp_count (int): Fast charging post count
            scp_count (int): Slow charging post count
            cs_data (dict): Charging station type data
            cp_data (dict): Charging post type data
            logger (logging.Logger, optional): Logger instance
        """
        self.station_id = station_id
        self.station_info = station_info
        self.cs_type = cs_type
        self.logger = logger or logging.getLogger(__name__)
        
        self.lon = station_info['Lon']
        self.lat = station_info['Lat']
        self.city_code = station_info['CityCode']
        
        self.cs_data = cs_data['dict'][cs_type]
        self.annual_cost = self.cs_data['AnnualCost']
        self.annual_ghg = self.cs_data['AnnualGHG']
        
        self.fcp_count = fcp_count
        self.scp_count = scp_count
        
        self.cp_data = cp_data['dict']
        self.fcp_power = self.cp_data['FCP']['Power']
        self.fcp_efficiency = self.cp_data['FCP']['Efficiency']
        self.scp_power = self.cp_data['SCP']['Power']
        self.scp_efficiency = self.cp_data['SCP']['Efficiency']
        
        self.fcp_end_times = [0] * fcp_count
        self.scp_end_times = [0] * scp_count
        
        self.total_services = 0
        self.total_charging_time = 0
        self.total_charging_amount = 0
        self.total_charging_cost = 0
        self.total_waiting_time = 0
        self.max_queue_length = 0
        self.current_queue_length = 0
        self.service_records = []
    
    def provide_charging_service(self, vehicle, current_time, electricity_price):
        """
        Provide charging service
        
        Args:
            vehicle (BEV): Electric vehicle requiring charge
            current_time (datetime): Current time
            electricity_price (float): Current electricity price (yuan/kWh)
        
        Returns:
            tuple: (service start time, service end time, charging amount, charging cost)
        """
        current_timestamp = current_time.timestamp()
        
        earliest_available_time = float('inf')
        selected_charger_type = None
        selected_charger_index = -1
        
        for i, end_time in enumerate(self.fcp_end_times):
            if end_time <= current_timestamp and end_time < earliest_available_time:
                earliest_available_time = end_time
                selected_charger_type = 'FCP'
                selected_charger_index = i
        
        if selected_charger_type is None:
            for i, end_time in enumerate(self.scp_end_times):
                if end_time <= current_timestamp and end_time < earliest_available_time:
                    earliest_available_time = end_time
                    selected_charger_type = 'SCP'
                    selected_charger_index = i
        
        if selected_charger_type is None:
            for i, end_time in enumerate(self.fcp_end_times):
                if end_time < earliest_available_time:
                    earliest_available_time = end_time
                    selected_charger_type = 'FCP'
                    selected_charger_index = i
            
            if selected_charger_type is None:
                for i, end_time in enumerate(self.scp_end_times):
                    if end_time < earliest_available_time:
                        earliest_available_time = end_time
                        selected_charger_type = 'SCP'
                        selected_charger_index = i
        
        if selected_charger_type is None:
            self.logger.error(f"Station {self.station_id} cannot find available charging post")
            return None, None, 0, 0
        
        waiting_time = 0
        if earliest_available_time > current_timestamp:
            waiting_time = earliest_available_time - current_timestamp
            self.current_queue_length += 1
            self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
        
        service_start_timestamp = max(current_timestamp, earliest_available_time)
        service_start_time = datetime.fromtimestamp(service_start_timestamp)
        
        charging_time, charging_amount, charging_cost = vehicle.charge(
            charging_station=self.cp_data,
            charging_type=selected_charger_type,
            electricity_price=electricity_price
        )
        
        service_end_timestamp = service_start_timestamp + charging_time
        service_end_time = datetime.fromtimestamp(service_end_timestamp)
        
        if selected_charger_type == 'FCP':
            self.fcp_end_times[selected_charger_index] = service_end_timestamp
        else:
            self.scp_end_times[selected_charger_index] = service_end_timestamp
        
        if waiting_time > 0:
            self.current_queue_length -= 1
        
        self.total_services += 1
        self.total_charging_time += charging_time
        self.total_charging_amount += charging_amount
        self.total_charging_cost += charging_cost
        self.total_waiting_time += waiting_time
        
        service_record = {
            'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'ARRIVAL',
            'StationID': self.station_id,
            'StationType': 'CS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(charging_time),
            'EnergyAmount': charging_amount,
            'ServiceCost': charging_cost,
            'ChargerType': selected_charger_type,
            'BatteryType': '',
            'HydrogenType': '',
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_record)
        
        service_start_record = {
            'Timestamp': service_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_START',
            'StationID': self.station_id,
            'StationType': 'CS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(charging_time),
            'EnergyAmount': charging_amount,
            'ServiceCost': charging_cost,
            'ChargerType': selected_charger_type,
            'BatteryType': '',
            'HydrogenType': '',
            'StationStatus': 'BUSY'
        }
        
        self.service_records.append(service_start_record)
        
        service_end_record = {
            'Timestamp': service_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_END',
            'StationID': self.station_id,
            'StationType': 'CS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': max(0, self.current_queue_length - 1),
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(charging_time),
            'EnergyAmount': charging_amount,
            'ServiceCost': charging_cost,
            'ChargerType': selected_charger_type,
            'BatteryType': '',
            'HydrogenType': '',
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_end_record)
        
        return service_start_time, service_end_time, charging_amount, charging_cost


class BatterySwapStation:
    """Battery swap station class that simulates battery swap service processes"""
    
    def __init__(self, station_id, station_info, bss_type, spare_batteries, bss_data, logger=None):
        """
        Initialize battery swap station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            bss_type (str): Battery swap station type (e.g., 'BSS_01')
            spare_batteries (dict): Spare battery count by type (initially may be empty)
            bss_data (dict): Battery swap station type data
            logger (logging.Logger, optional): Logger instance
        """
        self.station_id = station_id
        self.station_info = station_info
        self.bss_type = bss_type
        self.logger = logger or logging.getLogger(__name__)
        
        self.lon = station_info['Lon']
        self.lat = station_info['Lat']
        self.city_code = station_info['CityCode']
        
        self.bss_data = bss_data['dict'][bss_type]
        self.annual_cost = self.bss_data['AnnualCost']
        self.annual_ghg = self.bss_data['AnnualGHG']
        self.service_time = self.bss_data['ServiceTimeSeconds']
        
        self.spare_batteries = spare_batteries.copy() if spare_batteries else {}
        self.battery_refresh_schedule = {}
        
        self.daily_battery_usage = defaultdict(lambda: defaultdict(int))
        self.current_date = None
        
        self.service_end_time = 0
        
        self.total_services = 0
        self.total_service_time = 0
        self.total_swap_amount = 0
        self.total_swap_cost = 0
        self.total_waiting_time = 0
        self.max_queue_length = 0
        self.current_queue_length = 0
        self.service_records = []
    
    def provide_swap_service(self, vehicle, current_time, electricity_price, battery_refresh_interval):
        """
        Provide battery swap service
        
        Args:
            vehicle (BEV): Electric vehicle requiring battery swap
            current_time (datetime): Current time
            electricity_price (float): Current electricity price (yuan/kWh)
            battery_refresh_interval (float): Battery refresh interval (hours)
        
        Returns:
            tuple: (service start time, service end time, swap amount, swap cost)
        """
        current_timestamp = current_time.timestamp()
        
        current_date = current_time.date()
        
        if self.current_date != current_date:
            self.current_date = current_date
        
        vehicle_type = vehicle.vehicle_type
        
        self.daily_battery_usage[current_date][vehicle_type] += 1
        
        waiting_time = 0
        if self.service_end_time > current_timestamp:
            waiting_time = self.service_end_time - current_timestamp
            self.current_queue_length += 1
            self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
        
        service_start_timestamp = max(current_timestamp, self.service_end_time)
        service_start_time = datetime.fromtimestamp(service_start_timestamp)
        
        swap_time, swap_amount, swap_cost = vehicle.swap_battery(
            battery_swap_station=self.bss_data,
            spare_battery=None,
            electricity_price=electricity_price
        )
        
        service_end_timestamp = service_start_timestamp + swap_time
        service_end_time = datetime.fromtimestamp(service_end_timestamp)
        
        self.service_end_time = service_end_timestamp
        
        if waiting_time > 0:
            self.current_queue_length -= 1
        
        self.total_services += 1
        self.total_service_time += swap_time
        self.total_swap_amount += swap_amount
        self.total_swap_cost += swap_cost
        self.total_waiting_time += waiting_time
        
        service_record = {
            'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'ARRIVAL',
            'StationID': self.station_id,
            'StationType': 'BSS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(swap_time),
            'EnergyAmount': swap_amount,
            'ServiceCost': swap_cost,
            'ChargerType': '',
            'BatteryType': vehicle.vehicle_type,
            'HydrogenType': '',
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_record)
        
        service_start_record = {
            'Timestamp': service_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_START',
            'StationID': self.station_id,
            'StationType': 'BSS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(swap_time),
            'EnergyAmount': swap_amount,
            'ServiceCost': swap_cost,
            'ChargerType': '',
            'BatteryType': vehicle.vehicle_type,
            'HydrogenType': '',
            'StationStatus': 'BUSY'
        }
        
        self.service_records.append(service_start_record)
        
        service_end_record = {
            'Timestamp': service_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_END',
            'StationID': self.station_id,
            'StationType': 'BSS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': max(0, self.current_queue_length - 1),
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(swap_time),
            'EnergyAmount': swap_amount,
            'ServiceCost': swap_cost,
            'ChargerType': '',
            'BatteryType': vehicle.vehicle_type,
            'HydrogenType': '',
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_end_record)
        
        return service_start_time, service_end_time, swap_amount, swap_cost
    
    def update_battery_status(self, current_time):
        """
        Update battery status (method may no longer be needed in new logic)
        
        Args:
            current_time (datetime): Current time
        
        Returns:
            dict: Empty dictionary (maintains interface compatibility)
        """
        return {}

    def get_daily_max_demand(self):
        """
        Get maximum daily demand for each battery type
        
        Returns:
            dict: {battery_type: max_daily_demand}
        """
        max_demand = defaultdict(int)
        
        for date, battery_usage in self.daily_battery_usage.items():
            for battery_type, usage in battery_usage.items():
                max_demand[battery_type] = max(max_demand[battery_type], usage)
        
        return dict(max_demand)


class HydrogenStation:
    """Hydrogen station class that simulates hydrogen refueling service processes"""
    
    def __init__(self, station_id, station_info, hrs_type, hrs_data, logger=None):
        """
        Initialize hydrogen station
        
        Args:
            station_id (int): Station ID
            station_info (dict): Basic station information
            hrs_type (str): Hydrogen station type (e.g., 'HRS_01')
            hrs_data (dict): Hydrogen station type data
            logger (logging.Logger, optional): Logger instance
        """
        self.station_id = station_id
        self.station_info = station_info
        self.hrs_type = hrs_type
        self.logger = logger or logging.getLogger(__name__)
        
        self.lon = station_info['Lon']
        self.lat = station_info['Lat']
        self.city_code = station_info['CityCode']
        
        self.hrs_data = hrs_data['dict'][hrs_type]
        self.annual_cost = self.hrs_data['AnnualCost']
        self.annual_ghg = self.hrs_data['AnnualGHG']
        self.service_time = self.hrs_data['ServiceTimeSeconds']
        self.capacity = self.hrs_data['Capacity']
        self.efficiency = self.hrs_data['Efficiency']
        
        self.daily_hydrogen_usage = defaultdict(float)
        self.current_date = None
        
        self.service_end_time = 0
        
        self.total_services = 0
        self.total_service_time = 0
        self.total_refuel_amount = 0
        self.total_refuel_cost = 0
        self.total_waiting_time = 0
        self.max_queue_length = 0
        self.current_queue_length = 0
        self.service_records = []
    
    def provide_refueling_service(self, vehicle, current_time, hydrogen_price, hydrogen_refresh_interval):
        """
        Provide hydrogen refueling service
        
        Args:
            vehicle (HFCV): Hydrogen fuel cell vehicle requiring refueling
            current_time (datetime): Current time
            hydrogen_price (float): Current hydrogen price (yuan/kg)
            hydrogen_refresh_interval (float): Hydrogen refresh interval (hours) - parameter retained for interface compatibility
        
        Returns:
            tuple: (service start time, service end time, refuel amount, refuel cost)
        """
        current_timestamp = current_time.timestamp()
        
        current_date = current_time.date()
        
        if self.current_date != current_date:
            self.current_date = current_date
        
        needed_hydrogen = (1.0 - vehicle.current_soc) * vehicle.capacity
        
        self.daily_hydrogen_usage[current_date] += needed_hydrogen
        
        waiting_time = 0
        if self.service_end_time > current_timestamp:
            waiting_time = self.service_end_time - current_timestamp
            self.current_queue_length += 1
            self.max_queue_length = max(self.max_queue_length, self.current_queue_length)
        
        service_start_timestamp = max(current_timestamp, self.service_end_time)
        service_start_time = datetime.fromtimestamp(service_start_timestamp)
        
        refuel_time, refuel_amount, refuel_cost = vehicle.refuel(
            hydrogen_station=self.hrs_data,
            hydrogen_price=hydrogen_price
        )
        
        service_end_timestamp = service_start_timestamp + refuel_time
        service_end_time = datetime.fromtimestamp(service_end_timestamp)
        
        self.service_end_time = service_end_timestamp
        
        if waiting_time > 0:
            self.current_queue_length -= 1
        
        self.total_services += 1
        self.total_service_time += refuel_time
        self.total_refuel_amount += refuel_amount
        self.total_refuel_cost += refuel_cost
        self.total_waiting_time += waiting_time
        
        service_record = {
            'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'ARRIVAL',
            'StationID': self.station_id,
            'StationType': 'HRS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(refuel_time),
            'EnergyAmount': refuel_amount,
            'ServiceCost': refuel_cost,
            'ChargerType': '',
            'BatteryType': '',
            'HydrogenType': self.hrs_type,
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_record)
        
        service_start_record = {
            'Timestamp': service_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_START',
            'StationID': self.station_id,
            'StationType': 'HRS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': self.current_queue_length,
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(refuel_time),
            'EnergyAmount': refuel_amount,
            'ServiceCost': refuel_cost,
            'ChargerType': '',
            'BatteryType': '',
            'HydrogenType': self.hrs_type,
            'StationStatus': 'BUSY'
        }
        
        self.service_records.append(service_start_record)
        
        service_end_record = {
            'Timestamp': service_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'EventType': 'SERVICE_END',
            'StationID': self.station_id,
            'StationType': 'HRS',
            'VehicleID': vehicle.vehicle_id,
            'ActualVehicleID': vehicle.actual_vehicle_id,
            'VehicleType': vehicle.vehicle_type,
            'QueueLength': max(0, self.current_queue_length - 1),
            'WaitingTime': int(waiting_time),
            'ServiceTime': int(refuel_time),
            'EnergyAmount': refuel_amount,
            'ServiceCost': refuel_cost,
            'ChargerType': '',
            'BatteryType': '',
            'HydrogenType': self.hrs_type,
            'StationStatus': 'AVAILABLE'
        }
        
        self.service_records.append(service_end_record)
        
        return service_start_time, service_end_time, refuel_amount, refuel_cost
    
    def update_hydrogen_status(self, current_time, hydrogen_refresh_interval):
        """
        Update hydrogen status (method no longer needed in new logic, retained for interface compatibility)
        
        Args:
            current_time (datetime): Current time
            hydrogen_refresh_interval (float): Hydrogen refresh interval (hours)
        
        Returns:
            float: Returns 0.0 for compatibility
        """
        return 0.0

    def get_daily_max_demand(self):
        """
        Get maximum daily hydrogen demand (for demand analysis)
        
        Returns:
            float: Maximum daily hydrogen demand (kg)
        """
        if not self.daily_hydrogen_usage:
            return 0.0
        
        return max(self.daily_hydrogen_usage.values())