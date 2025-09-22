"""
Author: Shiqi (Anya) WANG
Date: 2025/5/2
Description: This script is used to generate the freight trip Origin-Destination (OD) pairs for each city.
"""

import logging
import sys
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import psutil


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TripRecord:
    """Truck OD data record"""
    index: int
    rated_capacity: float
    origin_lon: float
    origin_lat: float
    dest_lon: float
    dest_lat: float
    origin_city_code: str
    dest_city_code: str
    coefficient: Optional[float] = None

@dataclass
class CityFreightRecord:
    """City freight volume data record"""
    city: str
    city_code: str
    freight_volume: float
    origin_volume: float
    dest_volume: float

@dataclass
class InputPaths:
    """Input and output file path configuration"""
    BASE_DIR = Path("D:\ChinaTrip\Output")
    TRIP_FILE = BASE_DIR / "EFVTrip2019CoorReviseCityDiffODCargo.txt"
    FREIGHT_FILE = BASE_DIR / "2019CityFreVoluaddCodeRevisedDiffODFinalFinal.txt"
    OUTPUT_FILE = BASE_DIR / "EFVTrip2019CoorReviseCityDiffODExpand20250522.txt"


class DataLoader:
    """Data loading and preprocessing class"""
    
    @staticmethod
    def load_trip_data() -> pd.DataFrame:
        """Load truck OD data"""
        try:
            logging.info("Loading trip data")
            df = pd.read_csv(
                InputPaths.TRIP_FILE,
                sep='\t',
                encoding='utf-8',
                dtype={
                    'Index': int,
                    'OCityCode': str,
                    'DCityCode': str
                }
            )
            logging.info(f"Trip data loaded: {len(df)} records")
            return df
        except Exception as e:
            logging.error(f"Failed to load trip data: {str(e)}")
            raise

    @staticmethod
    def load_freight_data() -> pd.DataFrame:
        """Load city freight volume data"""
        try:
            logging.info("Loading freight data")
            df = pd.read_csv(
                InputPaths.FREIGHT_FILE,
                sep='\t',
                encoding='utf-8',
                dtype={
                    'CityCode': str,
                    'City': str,
                    '2019FreightVolumeRevisedDiffOD(t)': float,
                    'InterCityOVolume': float,
                    'InterCityDVolume': float
                }
            )
            logging.info(f"Freight data loaded: {len(df)} records")
            return df
        except Exception as e:
            logging.error(f"Failed to load freight data: {str(e)}")
            raise

    @staticmethod
    def preprocess_trip_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess truck OD data"""
        logging.info("Preprocessing trip data")
        
        initial_count = len(df)
        
        columns_to_check = ['核定载质量', 'OLon', 'OLat', 'DLon', 'DLat', 'OCityCode', 'DCityCode']
        mask = ~(
            df[columns_to_check].isna().any(axis=1) | 
            (df[columns_to_check] == 0).any(axis=1)
        )
        df_filtered = df[mask]
        
        filtered_count = len(df_filtered)
        logging.info(f"Filtered {initial_count - filtered_count} records, remaining: {filtered_count}")
        
        return df_filtered

    @staticmethod
    def preprocess_freight_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess city freight volume data"""
        logging.info("Preprocessing freight data")
        
        initial_count = len(df)
        
        invalid_records = df[df['CityCode'].isna() | (df['CityCode'] == '0')]
        if not invalid_records.empty:
            logging.warning("Invalid CityCode records found")
        
        df_filtered = df[~df['CityCode'].isna() & (df['CityCode'] != '0')]
        
        filtered_count = len(df_filtered)
        logging.info(f"Filtered {initial_count - filtered_count} records, remaining: {filtered_count}")
        
        return df_filtered
@dataclass
class GroupKey:
    """Key class for data grouping"""
    rated_capacity: float
    origin_city_code: str
    dest_city_code: str
    
    def __hash__(self):
        return hash((self.rated_capacity, self.origin_city_code, self.dest_city_code))


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, calculator: 'CoefficientCalculator') -> Dict[GroupKey, float]:
        """
        Execute optimization algorithm
        
        Args:
            calculator: CoefficientCalculator instance for fitness calculation
            
        Returns:
            Dict[GroupKey, float]: Optimized coefficient solution
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return algorithm name"""
        pass



class CoefficientCalculator:
    """Core class for calculating expansion coefficients"""
    
    def __init__(self, 
                 trip_data: pd.DataFrame, 
                 freight_data: pd.DataFrame,
                 optimizer: OptimizationAlgorithm,
                 relative_error_threshold: float = 0.05,
                 average_error_threshold: float = 0.03,
                 max_error_threshold: float = 0.15):
        """
        Initialize calculator
        
        Args:
            trip_data: Preprocessed truck OD data
            freight_data: Preprocessed city freight volume data
            optimizer: Optimization algorithm instance
            relative_error_threshold: Relative error threshold
            average_error_threshold: Average relative error threshold  
            max_error_threshold: Maximum relative error threshold
        """
        self.trip_data = trip_data
        self.freight_data = freight_data
        self.optimizer = optimizer  
        self.relative_error_threshold = relative_error_threshold
        self.average_error_threshold = average_error_threshold
        self.max_error_threshold = max_error_threshold
        
        self.capacity_sums = {}
        
        self.city_origin_freight_dict = dict(
            zip(freight_data['CityCode'], 
                freight_data['InterCityOVolume'])
        )
        
        self.city_dest_freight_dict = dict(
            zip(freight_data['CityCode'], 
                freight_data['InterCityDVolume'])
        )
        
        self.city_freight_dict = dict(
            zip(freight_data['CityCode'], 
                freight_data['2019FreightVolumeRevisedDiffOD(t)'])
        )
        
        self._init_city_tiers()
        
        self._init_groups()
       
        
        logging.info(f"CoefficientCalculator initialized: {len(self.groups)} groups")

    def _init_city_tiers(self):
        """Classify cities into tiers based on freight volume"""
        city_volumes = {
            city_code: volume 
            for city_code, volume in self.city_freight_dict.items()
        }
        
        sorted_cities = sorted(
            city_volumes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        total_cities = len(sorted_cities)
        tier1_count = int(total_cities * 0.2)
        tier2_count = int(total_cities * 0.5)
        
        self.city_tiers = {
            1: [city[0] for city in sorted_cities[:tier1_count]],
            2: [city[0] for city in sorted_cities[tier1_count:tier2_count]],
            3: [city[0] for city in sorted_cities[tier2_count:]]
        }
        
        total_volume = sum(city_volumes.values())
        self.city_weights = {
            city_code: volume/total_volume 
            for city_code, volume in city_volumes.items()
        }
        
        logging.info(f"City tiers initialized - Tier1: {len(self.city_tiers[1])}, "
                    f"Tier2: {len(self.city_tiers[2])}, "
                    f"Tier3: {len(self.city_tiers[3])}")
                
    def _init_groups(self):
        """Initialize data grouping"""
        logging.info("Initializing data groups")
        start_time = time.time()
        
        self.trip_data['OCityCode'] = self.trip_data['OCityCode'].astype(float).astype(int).astype(str)
        self.trip_data['DCityCode'] = self.trip_data['DCityCode'].astype(float).astype(int).astype(str)
        
        self.groups = {}
        group_counts = {'origin': {}, 'dest': {}}
        
        for idx, row in self.trip_data.iterrows():
            try:
                key = GroupKey(
                    rated_capacity=float(row['核定载质量']),
                    origin_city_code=str(row['OCityCode']),
                    dest_city_code=str(row['DCityCode'])
                )
                
                if key.origin_city_code not in group_counts['origin']:
                    group_counts['origin'][key.origin_city_code] = 0
                group_counts['origin'][key.origin_city_code] += 1
                
                if key.dest_city_code not in group_counts['dest']:
                    group_counts['dest'][key.dest_city_code] = 0
                group_counts['dest'][key.dest_city_code] += 1
                
                if key not in self.groups:
                    self.groups[key] = []
                self.groups[key].append(idx)
                
            except Exception as e:
                logging.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        logging.info(f"Total groups: {len(self.groups)}")
        
        for key, indices in self.groups.items():
            self.capacity_sums[key] = self.trip_data.loc[indices, '核定载质量'].sum()   
        
        logging.info(f"Data grouping completed in {time.time() - start_time:.2f}s")
        
    
    def analyze_target_data(self):
        """Analyze target data scale"""
        for city_code in list(self.city_origin_freight_dict.keys())[:5]:
            origin_target = self.city_origin_freight_dict.get(city_code, 0)
            dest_target = self.city_dest_freight_dict.get(city_code, 0)
            logging.info(f"City {city_code} - Origin: {origin_target:,.2f}, Dest: {dest_target:,.2f}")

    
    def calculate_city_freight_origin(self, city_code: str, coefficients: Dict[GroupKey, float]) -> float:
        """Calculate freight volume with city as origin"""
        return sum(
            coefficients[key] * self.capacity_sums[key]
            for key in self.groups.keys()
            if key.origin_city_code == city_code
        )
    
    def calculate_city_freight_destination(self, city_code: str, coefficients: Dict[GroupKey, float]) -> float:
        """Calculate freight volume with city as destination"""
        return sum(
            coefficients[key] * self.capacity_sums[key]
            for key in self.groups.keys()
            if key.dest_city_code == city_code
        )
    
        
    def validate_calculations(self, coefficients: Dict[GroupKey, float]) -> bool:
        """Validate calculation results"""
        sample_cities = list(self.city_origin_freight_dict.keys())[:5]
        
        for city_code in sample_cities:
            self.test_single_city(city_code, coefficients)
            
            origin_target = self.city_origin_freight_dict.get(city_code, 0)
            dest_target = self.city_dest_freight_dict.get(city_code, 0)
            origin_current = self.calculate_city_freight_origin(city_code, coefficients)
            dest_current = self.calculate_city_freight_destination(city_code, coefficients)
            
            logging.info(f"City {city_code} validation:")
            logging.info(f"Origin - Target: {origin_target:.2f}, Current: {origin_current:.2f}")
            logging.info(f"Dest - Target: {dest_target:.2f}, Current: {dest_current:.2f}")
            
            if origin_target > 0:
                origin_error = abs(origin_current - origin_target) / origin_target
                logging.info(f"Origin error: {origin_error:.4%}")
            
            if dest_target > 0:
                dest_error = abs(dest_current - dest_target) / dest_target
                logging.info(f"Dest error: {dest_error:.4%}")
        
        return True
    
    
    def test_single_city(self, city_code: str, coefficients: Dict[GroupKey, float]) -> None:
        """Test detailed calculation process for single city"""
        origin_target = self.city_origin_freight_dict.get(city_code, 0)
        dest_target = self.city_dest_freight_dict.get(city_code, 0)
        logging.info(f"Testing City {city_code} (Origin: {origin_target:,.2f}, Dest: {dest_target:,.2f})")
        
        origin_groups = [(k, v) for k, v in self.groups.items() if k.origin_city_code == city_code]
        if origin_groups:
            for group_key, indices in origin_groups[:3]:
                coef = coefficients[group_key]
                capacity_sum = sum(self.trip_data.loc[i, '核定载质量'] for i in indices)
                contribution = coef * capacity_sum
        
        dest_groups = [(k, v) for k, v in self.groups.items() if k.dest_city_code == city_code]
        if dest_groups:
            for group_key, indices in dest_groups[:3]:
                coef = coefficients[group_key]
                capacity_sum = sum(self.trip_data.loc[i, '核定载质量'] for i in indices)
                contribution = coef * capacity_sum    
    
    
    def calculate_fitness(self, coefficients: Dict[GroupKey, float]) -> Tuple[float, float, float, float]:
        """Calculate fitness and error metrics"""
        try:
            city_errors = {}
            total_freight = sum(v for v in self.city_freight_dict.values() if v > 0)
            
            for city_code in self.city_origin_freight_dict.keys():
                origin_target = self.city_origin_freight_dict.get(city_code, 0)
                dest_target = self.city_dest_freight_dict.get(city_code, 0)
                
                if origin_target <= 0 and dest_target <= 0:
                    continue
                
                origin_current = self.calculate_city_freight_origin(city_code, coefficients)
                dest_current = self.calculate_city_freight_destination(city_code, coefficients)
                
                origin_error = 0
                dest_error = 0
                
                if origin_target > 0 and origin_current > 0:
                    origin_error = abs(origin_current - origin_target) / origin_target
                elif origin_target > 0:
                    origin_error = 1.0
                
                if dest_target > 0 and dest_current > 0:
                    dest_error = abs(dest_current - dest_target) / dest_target  
                elif dest_target > 0:
                    dest_error = 1.0
                
                if origin_target > 0 and dest_target > 0:
                    combined_error = (origin_error + dest_error) / 2
                elif origin_target > 0:
                    combined_error = origin_error
                elif dest_target > 0:
                    combined_error = dest_error
                else:
                    continue
                
                if combined_error <= 0.2:
                    weighted_error = combined_error
                elif combined_error <= 0.5:
                    weighted_error = combined_error * 2
                else:
                    weighted_error = combined_error * 4
                
                tier = next((t for t, cities in self.city_tiers.items() if city_code in cities), 3)
                tier_weight = {1: 4.0, 2: 2.0, 3: 1.0}[tier]
                total_target = origin_target + dest_target
                volume_weight = (total_target / total_freight) * 5
                
                weight = tier_weight + volume_weight
                city_errors[city_code] = (weighted_error, weight)
            
            if not city_errors:
                return 0.0, 1.0, 1.0, 0.0
                
            total_weight = sum(weight for _, weight in city_errors.values())
            avg_error = sum(error * weight for error, weight in city_errors.values()) / total_weight
            max_error = max(error for error, _ in city_errors.values())
            
            fitness = 1.0 / (1.0 + avg_error ** 2)
            
            fitness = max(0.001, min(0.999, fitness))
            
            return fitness, avg_error, max_error, fitness
            
        except Exception as e:
            logging.error(f"Error calculating fitness: {str(e)}")
            return 0.0, 1.0, 1.0, 0.0
    
    def calculate_coefficients(self) -> Dict[GroupKey, float]:
        """Calculate coefficients"""
        logging.info(f"Starting coefficient calculation with {self.optimizer.get_algorithm_name()}...")
        return self.optimizer.optimize(self)

class ImprovedGeneticAlgorithm(OptimizationAlgorithm):
    """Improved genetic algorithm implementation"""
    
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 1000,
                 mutation_rate: float = 0.15,
                 elite_size: int = 20,
                 early_stop_patience: int = 50):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.early_stop_patience = early_stop_patience

    def get_algorithm_name(self) -> str:
        return "ImprovedGeneticAlgorithm"

    def initialize_population(self, calculator: 'CoefficientCalculator') -> List[Dict[GroupKey, float]]:
        """Improved population initialization"""
        logging.info("Initializing improved population...")
        population = []
        
        city_stats = self._analyze_city_targets(calculator)
        
        group_base_coefficients = {}
        for group_key in calculator.groups.keys():
            capacity = calculator.capacity_sums[group_key]
            if capacity <= 0:
                capacity = 1.0
            
            origin_target = calculator.city_origin_freight_dict.get(group_key.origin_city_code, 0)
            dest_target = calculator.city_dest_freight_dict.get(group_key.dest_city_code, 0)
            
            if origin_target > 0 or dest_target > 0:
                avg_target_per_capacity = max(origin_target, dest_target) / capacity if capacity > 0 else 1
                base_coef = max(1, min(1000, int(avg_target_per_capacity)))
            else:
                base_coef = 1
                
            group_base_coefficients[group_key] = base_coef
        
        for i in range(self.population_size):
            coefficients = {}
            
            if i == 0:
                coefficients = group_base_coefficients.copy()
            elif i < 10:
                for group_key in calculator.groups.keys():
                    base_coef = group_base_coefficients[group_key]
                    variation = np.random.uniform(0.8, 1.2)
                    coefficients[group_key] = max(1, int(base_coef * variation))
            else:
                for group_key in calculator.groups.keys():
                    base_coef = group_base_coefficients[group_key]
                    coef = max(1, min(1000, int(np.random.lognormal(np.log(base_coef), 0.5))))
                    coefficients[group_key] = coef
            
            population.append(coefficients)
        

        
        return population
    
    def _analyze_city_targets(self, calculator: 'CoefficientCalculator') -> dict:
        """Analyze statistical characteristics of city target data"""
        origin_targets = list(calculator.city_origin_freight_dict.values())
        dest_targets = list(calculator.city_dest_freight_dict.values())
        
        stats = {
            'origin_mean': np.mean([x for x in origin_targets if x > 0]),
            'origin_std': np.std([x for x in origin_targets if x > 0]),
            'dest_mean': np.mean([x for x in dest_targets if x > 0]),
            'dest_std': np.std([x for x in dest_targets if x > 0]),
        }
        

        return stats

    def select_parents(self, 
                      population: List[Dict[GroupKey, float]], 
                      fitness_scores: List[float]) -> List[Dict[GroupKey, float]]:
        """Improved selection strategy"""
        fitness_scores = np.array(fitness_scores)
        fitness_scores = np.nan_to_num(fitness_scores, nan=0.0, neginf=0.0)
        
        sorted_indices = np.argsort(fitness_scores)
        
        elite_indices = sorted_indices[-self.elite_size:]
        parents = [population[i] for i in elite_indices]
        
        remaining_count = self.population_size - self.elite_size
        for _ in range(remaining_count):
            tournament_indices = np.random.choice(len(population), size=3, replace=False)
            best_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
            parents.append(population[best_idx])
        
        return parents

    def crossover(self, parent1: Dict[GroupKey, float], parent2: Dict[GroupKey, float]) -> Dict[GroupKey, float]:
        """Improved crossover operation"""
        child = {}
        
        for group_key in parent1.keys():
            if np.random.random() < 0.5:
                child[group_key] = parent1[group_key]
            else:
                weight = np.random.beta(2, 2)
                value = weight * parent1[group_key] + (1 - weight) * parent2[group_key]
                child[group_key] = max(1, int(value))
        
        return child
    
    def mutate(self, individual: Dict[GroupKey, float], generation: int, max_generations: int):
        """Improved mutation operation"""
        progress = generation / max_generations
        local_mutation_rate = self.mutation_rate * (1 - 0.3 * progress)
        
        for group_key in individual.keys():
            if np.random.random() < local_mutation_rate:
                current_value = individual[group_key]
                
                mutation_type = np.random.choice(['gaussian', 'uniform', 'reset'])
                
                if mutation_type == 'gaussian':
                    sigma = max(1, current_value * 0.2)
                    delta = int(np.random.normal(0, sigma))
                    new_value = max(1, current_value + delta)
                elif mutation_type == 'uniform':
                    range_size = max(1, int(current_value * 0.5))
                    new_value = np.random.randint(max(1, current_value - range_size),
                                                current_value + range_size + 1)
                else:
                    new_value = np.random.randint(1, min(1000, current_value * 3))
                
                individual[group_key] = max(1, min(1000, new_value))

    def optimize(self, calculator: 'CoefficientCalculator') -> Dict[GroupKey, float]:
        """Improved optimization algorithm"""
        logging.info(f"{self.get_algorithm_name()} started...")
        start_time = time.time()
        
        population = self.initialize_population(calculator)
        best_fitness = float('-inf')
        best_coefficients = None
        best_avg_error = float('inf')
        best_max_error = float('inf')
        
        no_improvement_count = 0
        best_fitness_history = []
        
        for generation in range(self.generations):
            gen_start_time = time.time()
            
            population_metrics = [calculator.calculate_fitness(ind) for ind in population]
            fitness_scores = [metrics[0] for metrics in population_metrics]
            avg_errors = [metrics[1] for metrics in population_metrics]
            max_errors = [metrics[2] for metrics in population_metrics]
            
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_coefficients = population[current_best_idx].copy()
                best_avg_error = avg_errors[current_best_idx]
                best_max_error = max_errors[current_best_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            best_fitness_history.append(best_fitness)
            
            if (generation + 1) % 100 == 0:
                logging.info(f"Gen {generation + 1}/{self.generations} - "
                           f"Fitness: {best_fitness:.4f} - "
                           f"Avg Error: {best_avg_error:.2%}")
            
            if no_improvement_count >= self.early_stop_patience:
                logging.info(f"Early stopping: no improvement for {self.early_stop_patience} generations")
                break
            
            parents = self.select_parents(population, fitness_scores)
            
            new_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    child1 = self.crossover(parents[i], parents[i + 1])
                    child2 = self.crossover(parents[i + 1], parents[i])
                    self.mutate(child1, generation, self.generations)
                    self.mutate(child2, generation, self.generations)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(parents[i])
            
            if best_coefficients is not None:
                new_population[0] = best_coefficients.copy()
                    
            population = new_population
        
        logging.info(f"Improved GA completed in {time.time() - start_time:.2f}s, avg error: {best_avg_error:.2%}")
        
        return best_coefficients


class ImprovedCoefficientCalculator(CoefficientCalculator):
    """Improved coefficient calculator"""
    
    def calculate_fitness(self, coefficients: Dict[GroupKey, float]) -> Tuple[float, float, float, float]:
        """Improved fitness calculation"""
        try:
            city_errors = []
            valid_cities = 0
            
            for city_code in self.city_origin_freight_dict.keys():
                origin_target = self.city_origin_freight_dict.get(city_code, 0)
                dest_target = self.city_dest_freight_dict.get(city_code, 0)
                
                if origin_target <= 0 and dest_target <= 0:
                    continue
                
                origin_current = self.calculate_city_freight_origin(city_code, coefficients)
                dest_current = self.calculate_city_freight_destination(city_code, coefficients)
                
                errors = []
                
                if origin_target > 0:
                    if origin_current > 0:
                        origin_error = abs(origin_current - origin_target) / origin_target
                    else:
                        origin_error = 1.0
                    errors.append(origin_error)
                
                if dest_target > 0:
                    if dest_current > 0:
                        dest_error = abs(dest_current - dest_target) / dest_target
                    else:
                        dest_error = 1.0
                    errors.append(dest_error)
                
                if errors:
                    city_error = np.mean(errors)
                    city_errors.append(city_error)
                    valid_cities += 1
            
            if not city_errors:
                return 0.0, 1.0, 1.0, 0.0
            
            avg_error = np.mean(city_errors)
            max_error = np.max(city_errors)
            
            if avg_error < 0.2:
                fitness = 1.0 - avg_error * 2
            elif avg_error < 0.5:
                fitness = 0.6 - (avg_error - 0.2) * 1.5
            else:
                fitness = max(0.01, 0.15 - (avg_error - 0.5) * 0.3)
            
            fitness = max(0.001, min(0.999, fitness))
            
            return fitness, avg_error, max_error, fitness
            
        except Exception as e:
            logging.error(f"Error calculating fitness: {str(e)}")
            return 0.0, 1.0, 1.0, 0.0



    
class DataExporter:
    """Data export class for outputting calculation results to file"""
    
    def __init__(self, output_path: Path):
        """
        Initialize exporter
        
        Args:
            output_path: Output file path
        """
        self.output_path = output_path
        
    def export_results(self, 
                      original_data: pd.DataFrame, 
                      coefficients: Dict[GroupKey, float]) -> None:
        """
        Export calculation results to file
        
        Args:
            original_data: Original truck OD data
            coefficients: Calculated coefficient dictionary
        """
        logging.info("Exporting results")
        start_time = time.time()
        
        try:
            result_df = original_data.copy()
            
            result_df['Coefficient'] = result_df.apply(
                lambda row: coefficients.get(
                    GroupKey(
                        rated_capacity=row['核定载质量'],
                        origin_city_code=row['OCityCode'],
                        dest_city_code=row['DCityCode']
                    ),
                    0
                ),
                axis=1
            )
            
            result_df.to_csv(
                self.output_path,
                sep='\t',
                index=False,
                encoding='utf-8'
            )
            
            total_records = len(result_df)
            records_with_coef = len(result_df[result_df['Coefficient'] > 0])
            logging.info(f"Export completed: {total_records} records ({records_with_coef} with coefficients) in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Error exporting data: {str(e)}")
            raise

class Logger:
    """Log management class for configuring and managing log output"""
    
    def __init__(self, log_file: str = "freight_od_expansion20250522.log"):
        """
        Initialize log manager
        
        Args:
            log_file: Log file name
        """
        self.log_file = log_file
        self._setup_logger()
        
    def _setup_logger(self):
        """Configure log recorder"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        

    
    @staticmethod
    def log_section(section_name: str):
        """Output section marker"""
        logging.info(f"=== {section_name} ===")
    
    @staticmethod
    def log_memory_usage():
        """Log current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"Memory: {memory_info.rss / 1024 / 1024:.1f} MB")

@dataclass
class ImprovedAlgorithmConfig:
    """Improved algorithm configuration parameters"""
    population_size: int = 100
    generations: int = 1000
    mutation_rate: float = 0.15
    elite_size: int = 20
    early_stop_patience: int = 50
    relative_error_threshold: float = 0.05
    average_error_threshold: float = 0.03
    max_error_threshold: float = 0.15






class ImprovedConfigManager:
    """Improved configuration manager class"""
    
    def __init__(self, 
                 base_dir: Path = Path("D:\ChinaTrip\Output"),
                 algorithm_config: ImprovedAlgorithmConfig = None):
        """
        Initialize configuration manager
        
        Args:
            base_dir: Base directory path
            algorithm_config: Improved algorithm configuration parameters
        """
        self.base_dir = base_dir
        self.algorithm_config = algorithm_config or ImprovedAlgorithmConfig()
        

        self.input_paths = InputPaths()
        self.input_paths.BASE_DIR = base_dir
        

        self.max_memory_usage = 0.8
        self.n_jobs = -1
        
    def update_algorithm_config(self, **kwargs):
        """Update algorithm configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.algorithm_config, key):
                setattr(self.algorithm_config, key, value)
                logging.info(f"Updated config {key}: {value}")
            else:
                logging.warning(f"Unknown config parameter: {key}")
    
    def validate_paths(self):
        """Validate all required file paths"""
        required_files = [
            self.input_paths.TRIP_FILE,
            self.input_paths.FREIGHT_FILE
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required input file not found: {file_path}")

def main():
    """Main function"""
    try:
        logger = Logger()
        logger.log_section("Program Started")
        
        config_manager = ImprovedConfigManager()
        config_manager.validate_paths()
        
        logger.log_section("Loading Data")
        data_loader = DataLoader()
        trip_data = data_loader.load_trip_data()
        freight_data = data_loader.load_freight_data()
        
        logger.log_section("Preprocessing")
        trip_data = data_loader.preprocess_trip_data(trip_data)
        freight_data = data_loader.preprocess_freight_data(freight_data)
        
        logger.log_section("Optimization")
        optimizer = ImprovedGeneticAlgorithm(
            population_size=config_manager.algorithm_config.population_size,
            generations=config_manager.algorithm_config.generations,
            mutation_rate=config_manager.algorithm_config.mutation_rate,
            elite_size=config_manager.algorithm_config.elite_size,
            early_stop_patience=config_manager.algorithm_config.early_stop_patience
        )
        
        calculator = ImprovedCoefficientCalculator(
            trip_data=trip_data,
            freight_data=freight_data,
            optimizer=optimizer,
            relative_error_threshold=config_manager.algorithm_config.relative_error_threshold,
            average_error_threshold=config_manager.algorithm_config.average_error_threshold,
            max_error_threshold=config_manager.algorithm_config.max_error_threshold
        )
        
        calculator.analyze_target_data()
        logger.log_memory_usage()
        best_coefficients = calculator.calculate_coefficients()
        calculator.validate_calculations(best_coefficients)
        
        logger.log_section("Export")
        output_file = Path("D:\ChinaTrip\Output") / "EFVTrip2019CoorReviseCityDiffODExpandImproved.txt"
        exporter = DataExporter(output_file)
        exporter.export_results(trip_data, best_coefficients)
        
        logger.log_section("Completed")

    except Exception as e:
        import traceback
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()