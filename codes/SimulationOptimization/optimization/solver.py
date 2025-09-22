"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Multi-objective optimization solver module
Implements simulated annealing and parallel simulated annealing algorithms
"""

import numpy as np
import pandas as pd
import random
import time
import logging
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import pickle
import os
import gc
import copy
import json
from functools import partial
import uuid
import shutil

from optimization.objectives import ObjectiveCalculator
from simulation.scheduler import SimulationScheduler
from utils.logger import ProgressLogger, log_memory_usage


class MultiObjectiveOptimizer:
    """Multi-objective optimizer implementing various optimization algorithms"""
    
    def __init__(self, config, logger, simulation_scheduler, n_workers=None):
        """
        Initialize multi-objective optimizer
        
        Args:
            config (dict): Configuration dictionary
            logger (logging.Logger): Logger instance
            simulation_scheduler (SimulationScheduler): Simulation scheduler
            n_workers (int, optional): Number of parallel worker threads
        """
        self.config = config
        self.logger = logger
        self.scheduler = simulation_scheduler
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        
        self.objective_calculator = ObjectiveCalculator(
            simulation_scheduler=simulation_scheduler,
            config=config,
            logger=logger
        )
        
        self.optimization_config = config['optimization']
        self.sa_config = self.optimization_config['simulated_annealing']
        
        self.best_solution = None
        self.best_objective = float('inf')
        self.best_objectives = {}
        
        self.current_solution = None
        self.current_objective = float('inf')
        self.current_objectives = {}
        
        self.optimization_history = []
        
        random.seed(self.sa_config['random_seed'])
        np.random.seed(self.sa_config['random_seed'])
    
    def optimize(self):
        """
        Execute optimization process
        
        Returns:
            dict: Best solution information including solution, objective, objectives, and history
        """
        self.logger.info("Starting multi-objective optimization...")
        start_time = time.time()
        
        algorithm = self.sa_config.get('algorithm', 'simulated_annealing')
        
        if algorithm == 'simulated_annealing':
            self.logger.info("Using simulated annealing algorithm")
            self.simulated_annealing()
        elif algorithm == 'parallel_simulated_annealing':
            self.logger.info("Using parallel simulated annealing algorithm")
            self.parallel_simulated_annealing()
        else:
            self.logger.error(f"Unknown optimization algorithm: {algorithm}")
            return None
        
        self.logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Best objective value: {self.best_objective:.4f}")
        
        self._apply_solution(self.best_solution)
        
        return {
            'solution': self.best_solution,
            'objective': self.best_objective,
            'objectives': self.best_objectives,
            'history': self.optimization_history
        }
    
    def simulated_annealing(self):
        """Execute single-threaded simulated annealing algorithm"""
        initial_temperature = self.sa_config['initial_temperature']
        cooling_rate = self.sa_config['cooling_rate']
        iterations_per_temp = self.sa_config['iterations_per_temp']
        min_temperature = self.sa_config['min_temperature']
        max_iterations = self.sa_config['max_iterations']
        
        temperature = initial_temperature
        iteration = 0
        
        self.current_solution = self._generate_initial_solution()
        
        self._apply_solution(self.current_solution)
        max_trajectories = self.config.get('simulation', {}).get('max_trajectories', None)
        simulation_results = self.scheduler.simulate(max_trajectories=max_trajectories)
        self.current_objective, self.current_objectives = self.objective_calculator.calculate_weighted_objective(simulation_results)
        
        self.best_solution = self.current_solution.copy()
        self.best_objective = self.current_objective
        self.best_objectives = self.current_objectives.copy()
        
        self.optimization_history.append({
            'iteration': iteration,
            'temperature': temperature,
            'objective': self.current_objective,
            'total_cost': self.current_objectives['total_cost'],
            'total_travel_time': self.current_objectives['total_travel_time'],
            'total_ghg': self.current_objectives['total_ghg'],
            'accepted': True
        })
        
        self.logger.info(f"Initial objective value: {self.current_objective:.4f}")
        
        progress = ProgressLogger(self.logger, max_iterations, "Simulated annealing optimization")
        progress.start()
        
        output_interval = self.optimization_config.get('output_interval', 10)
        last_output_iteration = -output_interval
        
        while temperature > min_temperature and iteration < max_iterations:
            for i in range(iterations_per_temp):
                new_solution = self._generate_neighbor_solution(self.current_solution)

                self._apply_solution(new_solution)
                max_trajectories = self.config.get('simulation', {}).get('max_trajectories', None)
                simulation_results = self.scheduler.simulate(max_trajectories=max_trajectories)
                
                if hasattr(self.scheduler, 'infeasible_solutions') and self.scheduler.infeasible_solutions:
                    infeasible_count = len(self.scheduler.infeasible_solutions)
                    
                    if infeasible_count > 0:
                        first_reason = self.scheduler.infeasible_solutions[0].get('reason', 'Unknown')
                        self.logger.info(f"Infeasible solution: {infeasible_count} vehicles, reason: {first_reason}")
                    
                    penalty_factor = 10.0
                    new_objective = float('inf')
                    new_objectives = {
                        'total_cost': float('inf'),
                        'total_travel_time': float('inf'),
                        'total_ghg': float('inf'),
                        'normalized_cost': float('inf'),
                        'normalized_travel_time': float('inf'),
                        'normalized_ghg': float('inf')
                    }
                    
                    self.scheduler.infeasible_solutions = []
                    
                    accepted = False
                else:
                    new_objective, new_objectives = self.objective_calculator.calculate_weighted_objective(simulation_results)
                    
                    delta = new_objective - self.current_objective
                    acceptance_probability = min(1.0, np.exp(-delta / temperature))
                    
                    accepted = (delta < 0 or random.random() < acceptance_probability)

                    if new_objective < self.best_objective:
                        self.best_solution = new_solution.copy()
                        self.best_objective = new_objective
                        self.best_objectives = new_objectives.copy()
                        self.logger.info(f"New best solution found: {self.best_objective:.4f}")
                
                if iteration - last_output_iteration >= output_interval:
                    self._output_current_best_solution(iteration, temperature)
                    last_output_iteration = iteration
                
                self.optimization_history.append({
                    'iteration': iteration,
                    'temperature': temperature,
                    'objective': new_objective,
                    'total_cost': new_objectives['total_cost'],
                    'total_travel_time': new_objectives['total_travel_time'],
                    'total_ghg': new_objectives['total_ghg'],
                    'accepted': accepted
                })
                
                max_history_items = self.optimization_config.get('max_history_items', 1000)
                if len(self.optimization_history) > max_history_items:
                    self._save_partial_history()
                    self.optimization_history = self.optimization_history[-100:]
                    
                    import gc
                    gc.collect()
                
                iteration += 1
                
                progress.update()
                
                if iteration >= max_iterations:
                    break
            
            temperature *= cooling_rate
        
        progress.finish()
        
        self._save_optimization_history()

    def parallel_simulated_annealing(self):
        """Execute parallel simulated annealing algorithm"""
        initial_temperature = self.sa_config['initial_temperature']
        cooling_rate = self.sa_config['cooling_rate']
        iterations_per_temp = self.sa_config['iterations_per_temp']
        min_temperature = self.sa_config['min_temperature']
        max_iterations = self.sa_config['max_iterations']
        
        temperature = initial_temperature
        iteration = 0
        
        self.current_solution = self._generate_initial_solution()
        
        self._apply_solution(self.current_solution)
        max_trajectories = self.config.get('simulation', {}).get('max_trajectories', 1000)
        simulation_results = self.scheduler.simulate(max_trajectories=max_trajectories)
        self.current_objective, self.current_objectives = self.objective_calculator.calculate_weighted_objective(simulation_results)
        
        self.best_solution = self.current_solution.copy()
        self.best_objective = self.current_objective
        self.best_objectives = self.current_objectives.copy()
        
        self.optimization_history.append({
            'iteration': iteration,
            'temperature': temperature,
            'objective': self.current_objective,
            'total_cost': self.current_objectives['total_cost'],
            'total_travel_time': self.current_objectives['total_travel_time'],
            'total_ghg': self.current_objectives['total_ghg'],
            'accepted': True
        })
        
        self.logger.info(f"Initial objective value: {self.current_objective:.4f}")
        
        progress = ProgressLogger(self.logger, max_iterations, "Parallel simulated annealing optimization")
        progress.start()
        
        output_interval = self.optimization_config.get('output_interval', 10)
        last_output_iteration = -output_interval
        
        n_parallel = min(self.n_workers, 4)
        
        base_eval_dir = Path(self.scheduler.cache_dir.parent) / "eval_cache"
        os.makedirs(base_eval_dir, exist_ok=True)
        
        try:
            while temperature > min_temperature and iteration < max_iterations:
                batch_size = min(iterations_per_temp, max_iterations - iteration)
                n_batches = min(n_parallel, batch_size)
                if n_batches <= 0:
                    break
                
                neighbor_solutions = []
                for _ in range(n_parallel):
                    neighbor = self._generate_neighbor_solution(self.current_solution)
                    neighbor['memmap_suffix'] = f"_{uuid.uuid4().hex[:6]}"
                    neighbor_solutions.append(neighbor)
                
                with mp.Pool(n_parallel) as pool:
                    eval_tasks = []
                    for solution in neighbor_solutions:
                        eval_dir = base_eval_dir / f"eval{solution['memmap_suffix']}"
                        eval_tasks.append((solution, eval_dir))
                    
                    eval_results = pool.map(self._evaluate_solution_wrapper, eval_tasks)
                
                for solution_idx, ((solution, eval_dir), (objective, objectives, is_infeasible)) in enumerate(zip(eval_tasks, eval_results)):
                    current_iteration = iteration + solution_idx
                    
                    delta = objective - self.current_objective
                    acceptance_probability = min(1.0, np.exp(-delta / temperature))
                    
                    accepted = False
                    if (not is_infeasible) and (delta < 0 or random.random() < acceptance_probability):
                        self.current_solution = solution.copy()
                        self.current_objective = objective
                        self.current_objectives = objectives.copy()
                        accepted = True
                        
                        if objective < self.best_objective:
                            self.best_solution = solution.copy()
                            self.best_objective = objective
                            self.best_objectives = objectives.copy()
                            self.logger.info(f"New best solution found: {self.best_objective:.4f}")
                    
                    self.optimization_history.append({
                        'iteration': current_iteration,
                        'temperature': temperature,
                        'objective': objective,
                        'total_cost': objectives.get('total_cost', 0),
                        'total_travel_time': objectives.get('total_travel_time', 0),
                        'total_ghg': objectives.get('total_ghg', 0),
                        'accepted': accepted
                    })
                
                    if current_iteration - last_output_iteration >= output_interval:
                        self._output_current_best_solution(current_iteration, temperature)
                        last_output_iteration = current_iteration
                    
                    max_history_items = self.optimization_config.get('max_history_items', 1000)
                    if len(self.optimization_history) > max_history_items:
                        self._save_partial_history()
                        self.optimization_history = self.optimization_history[-100:]
                        
                        gc.collect()
                    
                    progress.update()
                    
                    try:
                        if os.path.exists(eval_dir):
                            shutil.rmtree(eval_dir, ignore_errors=True)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temp evaluation directory: {e}")
                
                iteration += n_parallel
                
                temperature *= cooling_rate
            
            progress.finish()
            
            self._save_optimization_history()
        
        finally:
            try:
                if os.path.exists(base_eval_dir):
                    shutil.rmtree(base_eval_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp evaluation directory: {e}")

    def _evaluate_solution_wrapper(self, args):
        """
        Wrapper function for evaluating solutions in parallel processing
        
        Args:
            args (tuple): (solution, eval_dir) tuple
        
        Returns:
            tuple: (objective, objectives, is_infeasible)
        """
        solution, eval_dir = args
        return self._evaluate_solution(solution, eval_dir)

    def _evaluate_solution(self, solution, eval_dir):
        """
        Evaluate single solution for parallel processing
        
        Args:
            solution (dict): Solution to be evaluated
            eval_dir (Path): Evaluation cache directory
        
        Returns:
            tuple: (objective, objectives, is_infeasible)
        """
        try:
            os.makedirs(eval_dir, exist_ok=True)
            
            scheduler_copy = SimulationScheduler(
                config=self.config,
                logger=self.logger,
                processed_data=self.scheduler.data,
                cache_dir=eval_dir,
                n_workers=1,
                memmap_suffix=solution.get('memmap_suffix', '')
            )
            
            scheduler_copy.configure_infrastructure(solution['station_config'])
            scheduler_copy.assign_vehicle_types(solution['vehicle_assignments'])
            
            max_trajectories = self.config.get('simulation', {}).get('max_trajectories', 1000)
            simulation_results = scheduler_copy.simulate(
                batch_size=200,
                max_trajectories=max_trajectories
            )
            
            is_infeasible = hasattr(scheduler_copy, 'infeasible_solutions') and scheduler_copy.infeasible_solutions
                
            objective = float('inf')
            objectives = {
                'total_cost': float('inf'),
                'total_travel_time': float('inf'),
                'total_ghg': float('inf'),
                'normalized_cost': 1.0,
                'normalized_travel_time': 1.0,
                'normalized_ghg': 1.0
            }
                
            if is_infeasible:
                infeasible_count = len(scheduler_copy.infeasible_solutions)
                if infeasible_count > 0:
                    first_reason = scheduler_copy.infeasible_solutions[0].get('reason', 'Unknown')
                    self.logger.info(f"Infeasible solution: {infeasible_count} vehicles, reason: {first_reason}")
                
                total_vehicles = len(solution['vehicle_assignments'])
                infeasible_ratio = infeasible_count / max(1, total_vehicles)
                
                base_value = 1000.0
                scaled_penalty = base_value * (1.0 + 10.0 * infeasible_ratio)
                
                cost_weight = self.objective_calculator.cost_weight if hasattr(self.objective_calculator, 'cost_weight') else 0.4
                travel_time_weight = self.objective_calculator.travel_time_weight if hasattr(self.objective_calculator, 'travel_time_weight') else 0.3
                ghg_weight = self.objective_calculator.ghg_weight if hasattr(self.objective_calculator, 'ghg_weight') else 0.3
                
                objective = scaled_penalty
                objectives = {
                    'total_cost': scaled_penalty * cost_weight,
                    'total_travel_time': scaled_penalty * travel_time_weight, 
                    'total_ghg': scaled_penalty * ghg_weight,
                    'normalized_cost': 1.0 + infeasible_ratio,
                    'normalized_travel_time': 1.0 + infeasible_ratio,
                    'normalized_ghg': 1.0 + infeasible_ratio
                }
            else:
                objective_calculator = ObjectiveCalculator(
                    simulation_scheduler=scheduler_copy,
                    config=self.config
                )
                objective, objectives = objective_calculator.calculate_weighted_objective(simulation_results)
            
            del scheduler_copy
            gc.collect()
            
            return objective, objectives, is_infeasible
        
        except Exception as e:
            self.logger.exception(f"Parallel solution evaluation failed: {e}")
            return float('inf'), {
                'total_cost': float('inf'),
                'total_travel_time': float('inf'),
                'total_ghg': float('inf'),
                'normalized_cost': float('inf'),
                'normalized_travel_time': float('inf'),
                'normalized_ghg': float('inf')
            }, True

    def _output_current_best_solution(self, iteration, temperature):
        """Output current best solution intermediate results"""
        output_dir = Path(self.config['path_config']['output_directory']) / 'intermediate'
        os.makedirs(output_dir, exist_ok=True)
        
        summary = {
            'iteration': iteration,
            'temperature': temperature,
            'objective': self.best_objective,
            'objectives': {
                'total_cost': self.best_objectives['total_cost'],
                'total_travel_time': self.best_objectives['total_travel_time'],
                'total_ghg': self.best_objectives['total_ghg']
            },
            'solution_stats': {
                'vehicle_types': len(set(v['Type'] for v in self.best_solution['vehicle_assignments'].values() if isinstance(v, dict))),
                'station_count': len(self.best_solution['station_config']),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        summary_file = output_dir / f"summary_iter_{iteration}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.optimization_config.get('save_full_intermediate', False):
            solution_file = output_dir / f"solution_iter_{iteration}.pkl"
            with open(solution_file, 'wb') as f:
                pickle.dump(self.best_solution, f)
        
        self.logger.info(f"Iteration {iteration}, temperature {temperature:.6f}, best value {self.best_objective:.4f}")
        self.logger.info(f"Intermediate results saved to {summary_file}")
    
    def _save_partial_history(self):
        """Save partial optimization history to reduce memory usage"""
        try:
            output_dir = Path(self.config['path_config']['output_directory'])
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            history_file = output_dir / f"optimization_history_part_{timestamp}.csv"
            
            history_data = []
            for entry in self.optimization_history:
                history_data.append({
                    'iteration': entry['iteration'],
                    'temperature': entry['temperature'],
                    'objective': entry['objective'],
                    'total_cost': entry.get('total_cost', 0),
                    'total_travel_time': entry.get('total_travel_time', 0),
                    'total_ghg': entry.get('total_ghg', 0),
                    'accepted': entry.get('accepted', False)
                })
            
            pd.DataFrame(history_data).to_csv(history_file, index=False)
            self.logger.debug(f"Partial optimization history saved to {history_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save partial optimization history: {e}")
    
    def _ensure_infrastructure_coverage(self, solution):
        """
        Ensure adequate infrastructure coverage for all vehicle types
        
        Args:
            solution (dict): Solution to be optimized
        """
        vehicle_types_count = defaultdict(int)
        for v_id, v_data in solution['vehicle_assignments'].items():
            if isinstance(v_data, dict):
                vehicle_types_count[v_data['Type']] += 1
        
        bev_types = [t for t in vehicle_types_count.keys() if t.startswith('BEV_')]
        
        hfcv_types = [t for t in vehicle_types_count.keys() if t.startswith('HFCV_')]
        
        current_cs = 0
        current_bss = 0
        current_hrs = 0
        
        for station_id, config in solution['station_config'].items():
            if config.get('CS', 0) > 0:
                current_cs += 1
            
            if config.get('BSS', 0) > 0:
                current_bss += 1
            
            if config.get('HRS_01', 0) > 0:
                current_hrs += 1
        
        all_stations = [str(idx) for idx in self.scheduler.stations.index]
        available_stations = [s for s in all_stations if s not in solution['station_config']]
        
        random.shuffle(available_stations)
        
        if bev_types:
            total_bev = sum(vehicle_types_count[t] for t in bev_types)
            
            min_bss_needed = max(len(bev_types), total_bev // 30)
            
            stations_to_add = min(min_bss_needed - current_bss, len(available_stations))
            
            for i in range(stations_to_add):
                if i < len(available_stations):
                    station_id = available_stations[i]
                    bss_config = {
                        'CS': 0,
                        'FCP': 0,
                        'SCP': 0,
                        'BSS': 1
                    }
                    
                    solution['station_config'][station_id] = bss_config
                    available_stations.remove(station_id)
                    current_bss += 1
                    
                    self.logger.info(f"Added BSS station {station_id}, total BSS: {current_bss}")
        
        if hfcv_types:
            total_hfcv = sum(vehicle_types_count[t] for t in hfcv_types)
            
            min_hrs_needed = max(len(hfcv_types), total_hfcv // 30)
            
            stations_to_add = min(min_hrs_needed - current_hrs, len(available_stations))
            
            for i in range(stations_to_add):
                if i < len(available_stations):
                    station_id = available_stations[i]
                    hrs_config = {
                        'CS': 0,
                        'FCP': 0,
                        'SCP': 0,
                        'BSS': 0,
                        'HRS_01': 1
                    }
                    
                    solution['station_config'][station_id] = hrs_config
                    available_stations.remove(station_id)
                    current_hrs += 1
                    
                    self.logger.info(f"Added HRS station {station_id}, total HRS: {current_hrs}")
    
    def _generate_initial_solution(self):
        """
        Generate initial solution
        
        Returns:
            dict: Initial solution
        """
        self.logger.info("Generating initial solution...")
        
        if hasattr(self.scheduler, 'trajectory_metadata') and self.scheduler.trajectory_metadata is not None:
            self.logger.info(f"Using trajectory metadata, total records: {self.scheduler.trajectory_metadata['total_count']}")
            
            metadata = self.scheduler.trajectory_metadata
            file_path = metadata['file_path']
            header = metadata['header']
            
            batch_size = 1000
            end_idx = min(batch_size, metadata['total_count'])
            positions = metadata['positions'][:end_idx]
            
            records = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for pos in positions:
                        f.seek(pos)
                        line = f.readline().strip()
                        parts = line.split('\t')
                        
                        if len(parts) == len(header):
                            record = dict(zip(header, parts))
                            records.append(record)
                
                import pandas as pd
                trajectories = pd.DataFrame(records)
                
                type_converters = {
                    'VehicleID': lambda x: int(x) if x.isdigit() else 0,
                    'Date': lambda x: int(x) if x.isdigit() else 0,
                    'Time': lambda x: int(x) if x.isdigit() else 0,
                    '核定载质量': lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0.0
                }
                
                for col, converter in type_converters.items():
                    if col in trajectories.columns:
                        trajectories[col] = trajectories[col].apply(converter)
                
                if 'Path' in trajectories.columns:
                    trajectories['PathList'] = trajectories['Path'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
                    trajectories['PathLength'] = trajectories['PathList'].apply(len)
                
                self.logger.info(f"Successfully loaded {len(trajectories)} trajectory records")
                
                vehicles_data = self.scheduler.data['vehicles']
                bev_types = [t for t in vehicles_data['df']['Type'] if t.startswith('BEV_')]
                hfcv_types = [t for t in vehicles_data['df']['Type'] if t.startswith('HFCV_')]
                
                stations = self.scheduler.data['stations']
                
                vehicle_assignments = {}
                
                actual_vehicle_counter = 1
                actual_vehicle_ids = {}
                
                for i, trajectory in trajectories.iterrows():
                    vehicle_id = str(trajectory['VehicleID'])
                    cargo_weight = trajectory['核定载质量']
                    date = trajectory['Date']
                    path_list = trajectory['PathList']
                    
                    start_city = self._get_city_from_road(path_list[0]) if len(path_list) > 0 else None
                    end_city = self._get_city_from_road(path_list[-1]) if len(path_list) > 0 else None
                    city_pair = (start_city, end_city)
                    
                    vehicle_category = 'BEV' if random.random() < 0.7 else 'HFCV'
                    
                    suitable_types = []
                    if vehicle_category == 'BEV':
                        suitable_types = [t for t in bev_types if self._is_suitable_cargo_weight(t, cargo_weight)]
                    else:
                        suitable_types = [t for t in hfcv_types if self._is_suitable_cargo_weight(t, cargo_weight)]
                    
                    if not suitable_types:
                        vehicle_category = 'HFCV' if vehicle_category == 'BEV' else 'BEV'
                        if vehicle_category == 'BEV':
                            suitable_types = [t for t in bev_types if self._is_suitable_cargo_weight(t, cargo_weight)]
                        else:
                            suitable_types = [t for t in hfcv_types if self._is_suitable_cargo_weight(t, cargo_weight)]
                    
                    if not suitable_types:
                        if vehicle_category == 'BEV':
                            suitable_types = bev_types if bev_types else ['BEV_01']
                        else:
                            suitable_types = hfcv_types if hfcv_types else ['HFCV_01']
                    
                    vehicle_type = random.choice(suitable_types)
                    
                    day_city_key = (date, city_pair)
                    if day_city_key in actual_vehicle_ids:
                        actual_vehicle_id = actual_vehicle_ids[day_city_key]
                    else:
                        actual_vehicle_id = f"V{actual_vehicle_counter}"
                        actual_vehicle_counter += 1
                        actual_vehicle_ids[day_city_key] = actual_vehicle_id
                    
                    vehicle_assignments[vehicle_id] = {
                        'Type': vehicle_type,
                        'ActualVehicleID': actual_vehicle_id
                    }
                
                station_config = {}
                
                station_count = min(
                    self.optimization_config['constraints']['max_stations'],
                    int(len(stations) * 0.2)
                )
                station_count = min(station_count, len(stations))
                
                if station_count > 0 and len(stations) > 0:
                    station_indices = [str(idx) for idx in stations.index]
                    selected_stations = random.sample(station_indices, station_count)
                    
                    cs_count = int(station_count * 0.34)
                    bss_count = int(station_count * 0.33)
                    hrs_count = station_count - cs_count - bss_count
                    
                    min_cs = 50
                    min_bss = 50
                    min_hrs = 50
                    
                    cs_count = max(cs_count, min_cs)
                    bss_count = max(bss_count, min_bss)
                    hrs_count = max(hrs_count, min_hrs)
                    
                    station_types = ['CS'] * cs_count + ['BSS'] * bss_count + ['HRS'] * hrs_count
                    random.shuffle(station_types)
                    
                    for station_id, station_type in zip(selected_stations, station_types):
                        if station_type == 'CS':
                            station_config[station_id] = {
                                'CS': 1,
                                'FCP': random.randint(2, 5),
                                'SCP': random.randint(3, 8),
                                'BSS': 0
                            }
                        elif station_type == 'BSS' and bev_types:
                            bss_config = {
                                'CS': 0,
                                'FCP': 0,
                                'SCP': 0,
                                'BSS': 1
                            }
                            
                            station_config[station_id] = bss_config
                            
                        elif station_type == 'HRS':
                            hrs_type = 'HRS_01'
                            
                            hrs_config = {
                                'CS': 0,
                                'FCP': 0,
                                'SCP': 0,
                                'BSS': 0
                            }
                            
                            hrs_config[hrs_type] = 1
                            
                            station_config[station_id] = hrs_config
                
                solution = {
                    'vehicle_assignments': vehicle_assignments,
                    'station_config': station_config
                }
                
                self._ensure_infrastructure_coverage(solution)
                
                self.logger.info(f"Initial solution generated with {len(vehicle_assignments)} vehicle assignments and {len(station_config)} station configs")
                return solution
                
            except Exception as e:
                self.logger.error(f"Failed to load trajectory data from file: {e}")
                return self._create_default_solution()
        else:
            self.logger.error("No trajectory metadata found, cannot use streaming load")
            return self._create_default_solution()
    
    def _create_default_solution(self):
        """Create empty default solution when trajectory data cannot be loaded"""
        self.logger.warning("Creating default solution structure")
        
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [t for t in vehicles_data['df']['Type'] if t.startswith('BEV_')]
        hfcv_types = [t for t in vehicles_data['df']['Type'] if t.startswith('HFCV_')]
                
        vehicle_assignments = {}
        vehicle_count = 100
        for i in range(1, vehicle_count + 1):
            vehicle_id = f"V{i}"
            
            if random.random() < 0.5:
                vehicle_type = random.choice(bev_types)
            else:
                vehicle_type = random.choice(hfcv_types)
                
            actual_vehicle_id = f"AV{i}"
            
            vehicle_assignments[vehicle_id] = {
                'Type': vehicle_type,
                'ActualVehicleID': actual_vehicle_id
            }
        
        station_config = {}
        
        station_count = min(
            self.optimization_config['constraints']['min_stations'],
            len(self.scheduler.data['stations'])
        )
        
        if station_count > 0:
            selected_stations = random.sample(list(self.scheduler.data['stations'].index), station_count)
            
            cs_count = int(station_count * 0.33)
            bss_count = int(station_count * 0.33)
            hrs_count = station_count - cs_count - bss_count
            
            station_types = ['CS'] * cs_count + ['BSS'] * bss_count + ['HRS'] * hrs_count
            random.shuffle(station_types)
            
            for station_id, station_type in zip(selected_stations, station_types):
                if station_type == 'CS':
                    station_config[station_id] = {
                        'CS': 1,
                        'FCP': random.randint(2, 5),
                        'SCP': random.randint(3, 8),
                        'BSS': 0
                    }
                elif station_type == 'BSS' and bev_types:
                    bss_config = {
                        'CS': 0,
                        'FCP': 0,
                        'SCP': 0,
                        'BSS': 1
                    }

                    station_config[station_id] = bss_config
                    
                elif station_type == 'HRS':
                    hrs_type = 'HRS_01'
                    
                    hrs_config = {
                        'CS': 0,
                        'FCP': 0,
                        'SCP': 0,
                        'BSS': 0
                    }
                    
                    hrs_config[hrs_type] = 1
                    
                    station_config[station_id] = hrs_config
        
        solution = {
            'vehicle_assignments': vehicle_assignments,
            'station_config': station_config
        }
        
        self.logger.info(f"Default solution generated with {len(vehicle_assignments)} vehicle assignments and {len(station_config)} station configs")
        return solution
    
    def _generate_neighbor_solution(self, solution):
        """
        Generate neighbor solution
        
        Args:
            solution (dict): Current solution
            
        Returns:
            dict: Neighbor solution
        """
        new_solution = {
            'vehicle_assignments': {},
            'station_config': {}
        }
        
        perturbation_type = random.choice([
            'change_vehicle_type',
            'change_station_type',
            'add_station',
            'remove_station',
            'adjust_charging_posts'
        ])
        
        if perturbation_type in ['change_vehicle_type']:
            new_solution['vehicle_assignments'] = solution['vehicle_assignments'].copy()
            new_solution['station_config'] = solution['station_config']
        else:
            new_solution['station_config'] = solution['station_config'].copy()
            new_solution['vehicle_assignments'] = solution['vehicle_assignments']
        
        if perturbation_type == 'change_vehicle_type':
            self._perturb_vehicle_types(new_solution)
        
        elif perturbation_type == 'change_station_type':
            self._perturb_station_types(new_solution)
        
        elif perturbation_type == 'add_station':
            self._perturb_add_station(new_solution)
        
        elif perturbation_type == 'remove_station':
            self._perturb_remove_station(new_solution)
        
        elif perturbation_type == 'adjust_charging_posts':
            self._perturb_charging_posts(new_solution)
        
        self._ensure_infrastructure_coverage(new_solution)
        
        return new_solution
    
    def _perturb_vehicle_types(self, solution):
        """
        Perturb vehicle types
        
        Args:
            solution (dict): Solution to be perturbed
        """
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [t for t in vehicles_data['df']['Type'] if t.startswith('BEV_')]
        hfcv_types = [t for t in vehicles_data['df']['Type'] if t.startswith('HFCV_')]
        
        vehicle_assignments = solution['vehicle_assignments']
        
        if not vehicle_assignments:
            self.logger.warning("No vehicle assignments, cannot perturb vehicle types")
            return
        
        change_count = max(1, int(len(vehicle_assignments) * 0.1))
        change_count = min(change_count, len(vehicle_assignments))
        
        selected_vehicles = random.sample(list(vehicle_assignments.keys()), change_count)
                
        for vehicle_id in selected_vehicles:
            current_type = vehicle_assignments[vehicle_id]['Type']
            current_category = 'BEV' if current_type.startswith('BEV_') else 'HFCV'
            
            if random.random() < 0.2:
                new_category = 'HFCV' if current_category == 'BEV' else 'BEV'
            else:
                new_category = current_category
            
            if new_category == 'BEV':
                new_type = random.choice(bev_types)
            else:
                new_type = random.choice(hfcv_types)
            
            vehicle_assignments[vehicle_id]['Type'] = new_type
    
    def _perturb_station_types(self, solution):
        """
        Perturb station types
        
        Args:
            solution (dict): Solution to be perturbed
        """
        station_config = solution['station_config']
        if not station_config:
            return
        
        change_count = max(1, int(len(station_config) * 0.2))
        
        selected_stations = random.sample(list(station_config.keys()), min(change_count, len(station_config)))
        
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [t for t in vehicles_data['df']['Type'] if t.startswith('BEV_')]
        
        for station_id in selected_stations:
            current_config = station_config[station_id]
            
            current_type = None
            if current_config.get('CS', 0) > 0:
                current_type = 'CS'
            elif current_config.get('BSS', 0) > 0:
                current_type = 'BSS'
            else:
                current_type = 'HRS'
            
            new_type = random.choice(['CS', 'BSS', 'HRS'])
            while new_type == current_type:
                new_type = random.choice(['CS', 'BSS', 'HRS'])
            
            new_config = {
                'CS': 0,
                'FCP': 0,
                'SCP': 0,
                'BSS': 0
            }
            if new_type == 'CS':
                new_config['CS'] = 1
                new_config['FCP'] = random.randint(2, 5)
                new_config['SCP'] = random.randint(3, 8)
            
            elif new_type == 'BSS':
                new_config['BSS'] = 1
            
            else:  # HRS
                hrs_type = 'HRS_01'
                new_config[hrs_type] = 1
            
            station_config[station_id] = new_config
    
    def _perturb_add_station(self, solution):
        """
        Perturb by adding stations
        
        Args:
            solution (dict): Solution to be perturbed
        """
        station_config = solution['station_config']
        all_stations = set(self.scheduler.data['stations'].index)
        used_stations = set(station_config.keys())
        unused_stations = all_stations - used_stations
        
        max_stations = self.optimization_config['constraints']['max_stations']
        if len(station_config) >= max_stations or not unused_stations:
            return
        
        add_count = max(1, int(len(station_config) * 0.05))
        add_count = min(add_count, max_stations - len(station_config))
        
        selected_stations = random.sample(list(unused_stations), min(add_count, len(unused_stations)))
        
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [t for t in vehicles_data['df']['Type'] if t.startswith('BEV_')]
        
        for station_id in selected_stations:
            station_type = random.choice(['CS', 'BSS', 'HRS'])
            
            if station_type == 'CS':
                fcp_count = random.randint(2, 5)
                scp_count = random.randint(3, 8)
                
                station_config[station_id] = {
                    'CS': 1,
                    'FCP': fcp_count,
                    'SCP': scp_count,
                    'BSS': 0
                }
            
            elif station_type == 'BSS':
                bss_config = {
                    'CS': 0,
                    'FCP': 0,
                    'SCP': 0,
                    'BSS': 1
                }
                
                station_config[station_id] = bss_config
            
            else:  # HRS
                hrs_type = 'HRS_01'
                
                hrs_config = {
                    'CS': 0,
                    'FCP': 0,
                    'SCP': 0,
                    'BSS': 0
                }
                
                hrs_config[hrs_type] = 1
                
                station_config[station_id] = hrs_config
    
    def _perturb_remove_station(self, solution):
        """
        Perturb by removing stations
        
        Args:
            solution (dict): Solution to be perturbed
        """
        station_config = solution['station_config']
        if not station_config:
            return
        
        min_stations = self.optimization_config['constraints']['min_stations']
        if len(station_config) <= min_stations:
            return
        
        remove_count = max(1, int(len(station_config) * 0.05))
        remove_count = min(remove_count, len(station_config) - min_stations)
        
        if remove_count > 0:
            selected_stations = random.sample(list(station_config.keys()), remove_count)
            for station_id in selected_stations:
                del station_config[station_id]
            
    def _perturb_charging_posts(self, solution):
        """
        Perturb charging post quantities
        
        Args:
            solution (dict): Solution to be perturbed
        """
        station_config = solution['station_config']
        charging_stations = [
            station_id for station_id, config in station_config.items()
            if config.get('CS', 0) > 0
        ]
        
        if not charging_stations:
            return
        
        adjust_count = max(1, int(len(charging_stations) * 0.5))
        selected_stations = random.sample(charging_stations, min(adjust_count, len(charging_stations)))
        
        for station_id in selected_stations:
            config = station_config[station_id]
            
            fcp_count = config.get('FCP', 0)
            if random.random() < 0.5:
                config['FCP'] = fcp_count + random.randint(1, 2)
            else:
                config['FCP'] = max(1, fcp_count - random.randint(1, 2))
            
            scp_count = config.get('SCP', 0)
            if random.random() < 0.5:
                config['SCP'] = scp_count + random.randint(1, 3)
            else:
                config['SCP'] = max(1, scp_count - random.randint(1, 3))
    
    def _apply_solution(self, solution):
        """
        Apply solution to scheduler
        
        Args:
            solution (dict): Solution to apply
        """
        self.scheduler.configure_infrastructure(solution['station_config'])
        
        self.scheduler.assign_vehicle_types(solution['vehicle_assignments'])
    
    def _get_city_from_road(self, road_id):
        """
        Get city code from road ID
        
        Args:
            road_id (str): Road ID
            
        Returns:
            int: City code
        """
        try:
            if road_id in self.scheduler.road_network.index:
                return self.scheduler.road_network.loc[road_id]['CityCode']
            else:
                return 0
        except Exception as e:
            self.logger.debug(f"Failed to get city code from road ID: {e}, road ID: {road_id}")
            return 0
    
    def _is_suitable_cargo_weight(self, vehicle_type, cargo_weight):
        """
        Check if vehicle type is suitable for specific cargo weight
        
        Args:
            vehicle_type (str): Vehicle type
            cargo_weight (float): Rated cargo weight
            
        Returns:
            bool: Whether suitable
        """
        try:
            vehicle_data = self.scheduler.data['vehicles']['dict'][vehicle_type]
            return abs(vehicle_data['CargoWeight'] - cargo_weight) < 0.01
        except Exception as e:
            self.logger.debug(f"Failed to check cargo weight compatibility: {e}, vehicle type: {vehicle_type}, weight: {cargo_weight}")
            return False
    
    def _save_optimization_history(self):
        """Save optimization history to file"""
        try:
            output_dir = Path(self.config['path_config']['output_directory'])
            os.makedirs(output_dir, exist_ok=True)
            
            history_file = output_dir / "optimization_history.csv"
            
            history_data = []
            for entry in self.optimization_history:
                history_data.append({
                    'iteration': entry['iteration'],
                    'temperature': entry['temperature'],
                    'objective': entry['objective'],
                    'cost': entry['objectives']['total_cost'],
                    'travel_time': entry['objectives']['total_travel_time'],
                    'ghg': entry['objectives']['total_ghg'],
                    'accepted': entry['accepted']
                })
            
            pd.DataFrame(history_data).to_csv(history_file, index=False)
            self.logger.info(f"Optimization history saved to {history_file}")
            
            solution_file = output_dir / "best_solution.pkl"
            with open(solution_file, 'wb') as f:
                pickle.dump(self.best_solution, f)
            self.logger.info(f"Best solution saved to {solution_file}")
            
            objective_file = output_dir / "best_objectives.csv"
            pd.DataFrame([self.best_objectives]).to_csv(objective_file, index=False)
            self.logger.info(f"Best objectives saved to {objective_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization history: {e}")
    
    def export_results(self):
        """Export optimization results"""
        self._apply_solution(self.best_solution)
        
        self.scheduler.simulate()
        
        self.scheduler.export_results()


