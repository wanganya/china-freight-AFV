"""
Author: Shiqi (Anya) WANG
Date: 2025/3/8
Description: Objective function implementation module
Defines calculation methods for three optimization objectives: total cost, total travel time, and total GHG emissions
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path


class ObjectiveCalculator:
    """Optimization objective calculator for computing multi-objective function values"""
    
    def __init__(self, simulation_scheduler, config, logger=None):
        """
        Initialize objective calculator
        
        Args:
            simulation_scheduler (SimulationScheduler): Simulation scheduler
            config (dict): Configuration dictionary
            logger (logging.Logger, optional): Logger instance
        """
        self.scheduler = simulation_scheduler
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.weights = config['optimization']['objective_weights']
        self.cost_weight = self.weights['cost']
        self.travel_time_weight = self.weights['travel_time']
        self.ghg_weight = self.weights['ghg']
        
        self.cost_normalization = 1.0
        self.travel_time_normalization = 1.0
        self.ghg_normalization = 1.0
        
        self.is_normalized = False
    
    def calculate_total_cost(self, simulation_results):
        """
        Calculate total cost
        
        Args:
            simulation_results (dict): Simulation results
        
        Returns:
            float: Total cost (yuan)
        """
        if 'cost_records' not in simulation_results:
            self.logger.error("Missing cost records in simulation results")
            return 0.0
        
        cost_records = simulation_results['cost_records']
        
        capital_cost = cost_records.get('capital_cost', 0.0)
        
        energy_cost = cost_records.get('energy_cost', 0.0)
        
        total_cost = capital_cost + energy_cost
        
        return total_cost
    
    def calculate_total_travel_time(self, simulation_results):
        """
        Calculate total travel time
        
        Args:
            simulation_results (dict): Simulation results
        
        Returns:
            float: Total travel time (seconds)
        """
        if 'driving_statistics' not in simulation_results:
            self.logger.error("Missing driving statistics in simulation results")
            return 0.0
        
        driving_statistics = simulation_results['driving_statistics']
        
        total_travel_time = sum(d.get('TotalDrivingTime', 0) for d in driving_statistics)
        
        return total_travel_time
    
    def calculate_total_ghg(self, simulation_results):
        """
        Calculate total GHG emissions
        
        Args:
            simulation_results (dict): Simulation results
        
        Returns:
            float: Total GHG emissions (kg)
        """
        if 'ghg_records' not in simulation_results:
            self.logger.error("Missing GHG records in simulation results")
            return 0.0
        
        ghg_records = simulation_results['ghg_records']
        
        infrastructure_ghg = ghg_records.get('infrastructure_ghg', 0.0)
        
        vehicle_ghg = ghg_records.get('vehicle_ghg', 0.0)
        
        total_ghg = infrastructure_ghg + vehicle_ghg
        
        return total_ghg
    
    def calculate_weighted_objective(self, simulation_results):
        """
        Calculate weighted objective function value
        
        Args:
            simulation_results (dict): Simulation results
        
        Returns:
            tuple: (weighted objective function value, dictionary of individual objective values)
        """
        total_cost = self.calculate_total_cost(simulation_results)
        total_travel_time = self.calculate_total_travel_time(simulation_results)
        total_ghg = self.calculate_total_ghg(simulation_results)
        
        if not self.is_normalized:
            self.cost_normalization = max(1.0, total_cost)
            self.travel_time_normalization = max(1.0, total_travel_time)
            self.ghg_normalization = max(1.0, total_ghg)
            self.is_normalized = True
        
        normalized_cost = total_cost / self.cost_normalization
        normalized_travel_time = total_travel_time / self.travel_time_normalization
        normalized_ghg = total_ghg / self.ghg_normalization
        
        weighted_sum = (
            self.cost_weight * normalized_cost +
            self.travel_time_weight * normalized_travel_time +
            self.ghg_weight * normalized_ghg
        )
        
        objectives = {
            'total_cost': total_cost,
            'total_travel_time': total_travel_time,
            'total_ghg': total_ghg,
            'normalized_cost': normalized_cost,
            'normalized_travel_time': normalized_travel_time,
            'normalized_ghg': normalized_ghg
        }
        
        return weighted_sum, objectives
    
    def update_normalization_factors(self, cost_factor=None, travel_time_factor=None, ghg_factor=None):
        """
        Update normalization factors
        
        Args:
            cost_factor (float, optional): Cost normalization factor
            travel_time_factor (float, optional): Travel time normalization factor
            ghg_factor (float, optional): GHG emission normalization factor
        """
        if cost_factor is not None:
            self.cost_normalization = max(1.0, cost_factor)
        
        if travel_time_factor is not None:
            self.travel_time_normalization = max(1.0, travel_time_factor)
        
        if ghg_factor is not None:
            self.ghg_normalization = max(1.0, ghg_factor)
        
        self.is_normalized = True
    
    def get_objective_weights(self):
        """
        Get objective weights
        
        Returns:
            dict: Dictionary of objective weights
        """
        return {
            'cost': self.cost_weight,
            'travel_time': self.travel_time_weight,
            'ghg': self.ghg_weight
        }
    
    def set_objective_weights(self, cost_weight=None, travel_time_weight=None, ghg_weight=None):
        """
        Set objective weights
        
        Args:
            cost_weight (float, optional): Cost weight
            travel_time_weight (float, optional): Travel time weight
            ghg_weight (float, optional): GHG emission weight
        """
        if cost_weight is not None:
            self.cost_weight = cost_weight
        
        if travel_time_weight is not None:
            self.travel_time_weight = travel_time_weight
        
        if ghg_weight is not None:
            self.ghg_weight = ghg_weight
        
        total_weight = self.cost_weight + self.travel_time_weight + self.ghg_weight
        
        if total_weight > 0:
            self.cost_weight /= total_weight
            self.travel_time_weight /= total_weight
            self.ghg_weight /= total_weight