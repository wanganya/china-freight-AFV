"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Pareto-based Adaptive Enhanced Simulated Annealing optimizer
"""

import copy
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from optimization.objectives import ObjectiveCalculator
from simulation.scheduler import SimulationScheduler
from utils.logger import ProgressLogger


class MultiObjectiveOptimizer:
    def __init__(self, config, logger, simulation_scheduler, n_workers=None):
        self.config = config
        self.logger = logger
        self.scheduler = simulation_scheduler
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        self.objective_calculator = ObjectiveCalculator(simulation_scheduler, config, logger)
        self.optimization_config = config['optimization']
        self.sa_config = self.optimization_config['simulated_annealing']
        self.archive_size = int(self.sa_config.get('archive_size', 200))
        self.pareto_archive = []
        self.best_solution = None
        self.best_objective = float('inf')
        self.best_objectives = {}
        self.current_solution = None
        self.current_objectives = {}
        self.optimization_history = []
        self._partial_history_file = None
        seed = int(self.sa_config.get('random_seed', 42))
        random.seed(seed)
        np.random.seed(seed)

    def optimize(self):
        self.logger.info("Starting Pareto multi-objective optimization")
        start_time = time.time()
        algorithm = str(self.sa_config.get('algorithm', 'parallel_aesa')).lower()
        if algorithm in {'aesa', 'simulated_annealing'}:
            self.aesa()
        elif algorithm in {'parallel_aesa', 'parallel_simulated_annealing'}:
            self.parallel_aesa()
        else:
            raise ValueError(f"Unknown optimization algorithm: {algorithm}")
        self._refresh_best_from_archive()
        if self.best_solution is None:
            raise RuntimeError("Optimization completed without a feasible Pareto solution")
        self._apply_solution(self.best_solution)
        self._save_optimization_history()
        self.logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        self.logger.info(f"Pareto archive size: {len(self.pareto_archive)}")
        self.logger.info(f"Best compromise score: {self.best_objective:.6f}")
        return {
            'solution': copy.deepcopy(self.best_solution),
            'objective': self.best_objective,
            'objectives': copy.deepcopy(self.best_objectives),
            'pareto_front': [
                {
                    'solution': copy.deepcopy(entry['solution']),
                    'objectives': copy.deepcopy(entry['objectives']),
                    'iteration': entry.get('iteration', 0),
                    'crowding_distance': entry.get('crowding_distance', 0.0)
                }
                for entry in self.pareto_archive
            ],
            'history': list(self.optimization_history)
        }

    def simulated_annealing(self):
        return self.aesa()

    def parallel_simulated_annealing(self):
        return self.parallel_aesa()

    def aesa(self):
        initial_temperature = float(self.sa_config['initial_temperature'])
        min_temperature = float(self.sa_config['min_temperature'])
        iterations_per_temp = int(self.sa_config['iterations_per_temp'])
        max_iterations = int(self.sa_config['max_iterations'])
        temperature = initial_temperature
        iteration = 0
        stagnation_levels = 0
        self.current_solution, self.current_objectives = self._initialize_feasible_state()
        self._update_archive(self.current_solution, self.current_objectives, iteration)
        self._refresh_best_from_archive()
        self._record_history(iteration, temperature, self.current_objectives, True, True, 'initial', 1.0, 0)
        progress = ProgressLogger(self.logger, max_iterations, "Pareto-AESA optimization")
        progress.start()
        output_interval = int(self.optimization_config.get('output_interval', 10))
        last_output_iteration = -output_interval
        while temperature > min_temperature and iteration < max_iterations:
            stage_target = min(iterations_per_temp, max_iterations - iteration)
            accepted_count = 0
            archive_additions = 0
            for _ in range(stage_target):
                candidate = self._generate_neighbor_solution(self.current_solution)
                candidate_objectives, infeasible, _ = self._evaluate_with_current_scheduler(candidate)
                accepted = False
                probability = 0.0
                relation = 'infeasible'
                archive_added = False
                if not infeasible:
                    probability, relation = self._acceptance_probability(
                        candidate_objectives,
                        self.current_objectives,
                        temperature,
                        initial_temperature
                    )
                    accepted = random.random() < probability
                    if accepted:
                        self.current_solution = self._clean_solution(candidate)
                        self.current_objectives = copy.deepcopy(candidate_objectives)
                        accepted_count += 1
                    archive_added = self._update_archive(candidate, candidate_objectives, iteration + 1)
                    if archive_added:
                        archive_additions += 1
                        self._refresh_best_from_archive()
                iteration += 1
                self._record_history(
                    iteration,
                    temperature,
                    candidate_objectives,
                    accepted,
                    archive_added,
                    relation,
                    probability,
                    0
                )
                if iteration - last_output_iteration >= output_interval:
                    self._output_current_best_solution(iteration, temperature)
                    last_output_iteration = iteration
                self._manage_history_memory()
                progress.update()
                if iteration >= max_iterations:
                    break
            acceptance_rate = accepted_count / max(1, stage_target)
            improvement_rate = archive_additions / max(1, stage_target)
            if archive_additions == 0:
                stagnation_levels += 1
            else:
                stagnation_levels = 0
            temperature = self._adaptive_temperature(
                temperature,
                initial_temperature,
                min_temperature,
                acceptance_rate,
                improvement_rate,
                stagnation_levels
            )
            restart_after = int(self.sa_config.get('restart_after_stagnation', 10))
            if stagnation_levels >= restart_after and self.pareto_archive:
                entry = self._select_restart_entry()
                self.current_solution = copy.deepcopy(entry['solution'])
                self.current_objectives = copy.deepcopy(entry['objectives'])
                stagnation_levels = 0
        progress.finish()

    def parallel_aesa(self):
        initial_temperature = float(self.sa_config['initial_temperature'])
        min_temperature = float(self.sa_config['min_temperature'])
        iterations_per_temp = int(self.sa_config['iterations_per_temp'])
        max_iterations = int(self.sa_config['max_iterations'])
        n_parallel = max(1, min(self.n_workers, int(self.sa_config.get('max_parallel_chains', 4))))
        temperature = initial_temperature
        iteration = 0
        stagnation_levels = 0
        initial_solution, initial_objectives = self._initialize_feasible_state()
        self.current_solution = copy.deepcopy(initial_solution)
        self.current_objectives = copy.deepcopy(initial_objectives)
        self._update_archive(initial_solution, initial_objectives, iteration)
        self._refresh_best_from_archive()
        chains = [
            {
                'solution': copy.deepcopy(initial_solution),
                'objectives': copy.deepcopy(initial_objectives)
            }
            for _ in range(n_parallel)
        ]
        self._record_history(iteration, temperature, initial_objectives, True, True, 'initial', 1.0, 0)
        progress = ProgressLogger(self.logger, max_iterations, "Parallel Pareto-AESA optimization")
        progress.start()
        output_interval = int(self.optimization_config.get('output_interval', 10))
        last_output_iteration = -output_interval
        base_eval_dir = Path(self.scheduler.cache_dir.parent) / "eval_cache"
        os.makedirs(base_eval_dir, exist_ok=True)
        try:
            with mp.Pool(n_parallel) as pool:
                while temperature > min_temperature and iteration < max_iterations:
                    stage_target = min(iterations_per_temp, max_iterations - iteration)
                    stage_done = 0
                    accepted_count = 0
                    archive_additions = 0
                    while stage_done < stage_target and iteration < max_iterations:
                        batch_size = min(n_parallel, stage_target - stage_done, max_iterations - iteration)
                        eval_tasks = []
                        chain_indices = []
                        for chain_index in range(batch_size):
                            parent = chains[chain_index]
                            candidate = self._generate_neighbor_solution(parent['solution'])
                            suffix = f"_{uuid.uuid4().hex[:8]}"
                            candidate['memmap_suffix'] = suffix
                            eval_dir = base_eval_dir / f"eval{suffix}"
                            eval_tasks.append((candidate, eval_dir))
                            chain_indices.append(chain_index)
                        eval_results = pool.map(self._evaluate_solution_wrapper, eval_tasks)
                        for local_index, ((candidate, eval_dir), result) in enumerate(zip(eval_tasks, eval_results)):
                            candidate_objectives, infeasible, _ = result
                            chain_index = chain_indices[local_index]
                            parent = chains[chain_index]
                            accepted = False
                            probability = 0.0
                            relation = 'infeasible'
                            archive_added = False
                            if not infeasible:
                                probability, relation = self._acceptance_probability(
                                    candidate_objectives,
                                    parent['objectives'],
                                    temperature,
                                    initial_temperature
                                )
                                accepted = random.random() < probability
                                if accepted:
                                    parent['solution'] = self._clean_solution(candidate)
                                    parent['objectives'] = copy.deepcopy(candidate_objectives)
                                    accepted_count += 1
                                archive_added = self._update_archive(candidate, candidate_objectives, iteration + 1)
                                if archive_added:
                                    archive_additions += 1
                                    self._refresh_best_from_archive()
                            iteration += 1
                            stage_done += 1
                            self._record_history(
                                iteration,
                                temperature,
                                candidate_objectives,
                                accepted,
                                archive_added,
                                relation,
                                probability,
                                chain_index
                            )
                            if iteration - last_output_iteration >= output_interval:
                                self._output_current_best_solution(iteration, temperature)
                                last_output_iteration = iteration
                            self._manage_history_memory()
                            progress.update()
                            try:
                                if os.path.exists(eval_dir):
                                    shutil.rmtree(eval_dir, ignore_errors=True)
                            except Exception as exc:
                                self.logger.warning(f"Failed to remove evaluation directory {eval_dir}: {exc}")
                    acceptance_rate = accepted_count / max(1, stage_target)
                    improvement_rate = archive_additions / max(1, stage_target)
                    if archive_additions == 0:
                        stagnation_levels += 1
                    else:
                        stagnation_levels = 0
                    temperature = self._adaptive_temperature(
                        temperature,
                        initial_temperature,
                        min_temperature,
                        acceptance_rate,
                        improvement_rate,
                        stagnation_levels
                    )
                    restart_after = int(self.sa_config.get('restart_after_stagnation', 10))
                    if stagnation_levels >= restart_after and self.pareto_archive:
                        restart_entries = self._select_restart_entries(n_parallel)
                        for chain_index, entry in enumerate(restart_entries):
                            chains[chain_index]['solution'] = copy.deepcopy(entry['solution'])
                            chains[chain_index]['objectives'] = copy.deepcopy(entry['objectives'])
                        stagnation_levels = 0
            self.current_solution = copy.deepcopy(self.best_solution)
            self.current_objectives = copy.deepcopy(self.best_objectives)
        finally:
            try:
                if os.path.exists(base_eval_dir):
                    shutil.rmtree(base_eval_dir, ignore_errors=True)
            except Exception as exc:
                self.logger.warning(f"Failed to clean evaluation directory {base_eval_dir}: {exc}")
            progress.finish()

    def _initialize_feasible_state(self):
        attempts = int(self.sa_config.get('max_initial_attempts', 20))
        candidate = self._generate_initial_solution()
        last_reason = None
        for attempt in range(attempts):
            objectives, infeasible, ratio = self._evaluate_with_current_scheduler(candidate)
            if not infeasible:
                return self._clean_solution(candidate), objectives
            last_reason = f"infeasible ratio {ratio:.6f}"
            candidate = self._generate_neighbor_solution(candidate)
        raise RuntimeError(f"Unable to generate a feasible initial solution after {attempts} attempts: {last_reason}")

    def _evaluate_with_current_scheduler(self, solution):
        try:
            self._apply_solution(solution)
            max_trajectories = self.config.get('simulation', {}).get('max_trajectories')
            simulation_results = self.scheduler.simulate(max_trajectories=max_trajectories)
            infeasible_count = len(getattr(self.scheduler, 'infeasible_solutions', []) or [])
            total_vehicles = max(1, len(solution.get('vehicle_assignments', {})))
            infeasible_ratio = infeasible_count / total_vehicles
            if infeasible_count > 0:
                return self._infeasible_objectives(infeasible_ratio), True, infeasible_ratio
            return self.objective_calculator.calculate_objectives(simulation_results), False, 0.0
        except Exception as exc:
            self.logger.exception(f"Solution evaluation failed: {exc}")
            return self._infeasible_objectives(1.0), True, 1.0

    def _evaluate_solution_wrapper(self, args):
        solution, eval_dir = args
        return self._evaluate_solution(solution, eval_dir)

    def _evaluate_solution(self, solution, eval_dir):
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
            max_trajectories = self.config.get('simulation', {}).get('max_trajectories')
            simulation_results = scheduler_copy.simulate(batch_size=200, max_trajectories=max_trajectories)
            infeasible_count = len(getattr(scheduler_copy, 'infeasible_solutions', []) or [])
            total_vehicles = max(1, len(solution.get('vehicle_assignments', {})))
            infeasible_ratio = infeasible_count / total_vehicles
            if infeasible_count > 0:
                result = self._infeasible_objectives(infeasible_ratio), True, infeasible_ratio
            else:
                calculator = ObjectiveCalculator(scheduler_copy, self.config)
                result = calculator.calculate_objectives(simulation_results), False, 0.0
            del scheduler_copy
            gc.collect()
            return result
        except Exception as exc:
            self.logger.exception(f"Parallel solution evaluation failed: {exc}")
            return self._infeasible_objectives(1.0), True, 1.0

    @staticmethod
    def _infeasible_objectives(infeasible_ratio):
        return {
            'total_cost': float('inf'),
            'total_travel_time': float('inf'),
            'total_ghg': float('inf'),
            'objective_vector': (float('inf'), float('inf'), float('inf')),
            'infeasible_ratio': float(infeasible_ratio)
        }

    def _acceptance_probability(self, candidate_objectives, current_objectives, temperature, initial_temperature):
        candidate_dominates = self.objective_calculator.dominates(candidate_objectives, current_objectives)
        current_dominates = self.objective_calculator.dominates(current_objectives, candidate_objectives)
        temperature_ratio = max(float(temperature) / max(float(initial_temperature), 1e-12), 1e-12)
        if candidate_dominates:
            return 1.0, 'candidate_dominates'
        if current_dominates:
            current = self.objective_calculator.objective_vector(current_objectives)
            candidate = self.objective_calculator.objective_vector(candidate_objectives)
            scale = np.maximum(np.maximum(np.abs(current), np.abs(candidate)), 1e-12)
            deterioration = np.maximum((candidate - current) / scale, 0.0)
            delta = float(np.mean(deterioration))
            probability = float(np.exp(-delta / temperature_ratio))
            return min(1.0, max(0.0, probability)), 'current_dominates'
        candidate = self.objective_calculator.objective_vector(candidate_objectives)
        current = self.objective_calculator.objective_vector(current_objectives)
        if np.allclose(candidate, current, rtol=1e-12, atol=1e-12):
            probability = min(1.0, 0.25 * temperature_ratio)
            return probability, 'equal'
        # Non-dominated solutions are accepted according to Pareto-AESA diversity,
        # not weighted compromise scores. This avoids preference leakage during search.
        diversity_bonus = float(self.sa_config.get('diversity_acceptance_bonus', 0.15)) * self._candidate_diversity(candidate_objectives)
        probability = min(1.0, max(0.0, temperature_ratio + diversity_bonus))
        return probability, 'nondominated'

    def _pair_compromise_scores(self, candidate_objectives, current_objectives):
        items = [entry['objectives'] for entry in self.pareto_archive]
        items.extend([current_objectives, candidate_objectives])
        vectors = np.vstack([self.objective_calculator.objective_vector(item) for item in items])
        normalized = self.objective_calculator.normalize_vectors(vectors)
        weights = self.objective_calculator.compromise_weights
        scores = np.sqrt(np.sum(normalized ** 2 * weights.reshape(1, -1), axis=1))
        return float(scores[-1]), float(scores[-2])

    def _candidate_diversity(self, candidate_objectives):
        if not self.pareto_archive:
            return 1.0
        objectives = [entry['objectives'] for entry in self.pareto_archive] + [candidate_objectives]
        vectors = np.vstack([self.objective_calculator.objective_vector(item) for item in objectives])
        normalized = self.objective_calculator.normalize_vectors(vectors)
        candidate = normalized[-1]
        existing = normalized[:-1]
        distances = np.sqrt(np.sum((existing - candidate) ** 2, axis=1))
        if distances.size == 0:
            return 1.0
        return min(1.0, float(np.min(distances)) / np.sqrt(3.0))

    def _adaptive_temperature(self, temperature, initial_temperature, min_temperature, acceptance_rate, improvement_rate, stagnation_levels):
        base = float(self.sa_config.get('cooling_rate', 0.98))
        target_low = float(self.sa_config.get('target_acceptance_low', 0.20))
        target_high = float(self.sa_config.get('target_acceptance_high', 0.55))
        strength = float(self.sa_config.get('adaptation_strength', 0.20))
        if acceptance_rate < target_low:
            factor = min(1.0, base + strength * (target_low - acceptance_rate))
        elif acceptance_rate > target_high:
            factor = max(0.50, base - strength * (acceptance_rate - target_high))
        else:
            factor = base
        if improvement_rate > float(self.sa_config.get('rapid_improvement_threshold', 0.15)):
            factor = max(0.50, factor * float(self.sa_config.get('rapid_improvement_cooling_multiplier', 0.98)))
        new_temperature = temperature * factor
        reheat_after = int(self.sa_config.get('reheat_after_stagnation', 5))
        if stagnation_levels >= reheat_after:
            reheat_factor = float(self.sa_config.get('reheat_factor', 1.20))
            new_temperature = max(new_temperature, min(initial_temperature, temperature * reheat_factor))
        return max(float(min_temperature) * 0.999999, float(new_temperature))

    def _solution_signature(self, solution):
        """Create a stable decision-space signature for archive duplicate control."""
        vehicles = tuple(sorted(
            (str(k), str(v.get('Type')), float(v.get('CargoWeight', 0.0)))
            for k, v in solution.get('vehicle_assignments', {}).items()
            if isinstance(v, dict)
        ))
        stations = tuple(sorted(
            (str(k), tuple(sorted((str(a), float(b)) for a, b in v.items())))
            for k, v in solution.get('station_config', {}).items()
        ))
        return hash((vehicles, stations))

    def _update_archive(self, solution, objectives, iteration):
        vector = self.objective_calculator.objective_vector(objectives)
        if not np.all(np.isfinite(vector)):
            return False
        for entry in self.pareto_archive:
            existing = self.objective_calculator.objective_vector(entry['objectives'])
            if np.allclose(existing, vector, rtol=1e-10, atol=1e-10):
                if entry.get('solution_signature') == self._solution_signature(solution):
                    return False
            if self.objective_calculator.dominates(entry['objectives'], objectives):
                return False
        retained = []
        for entry in self.pareto_archive:
            if not self.objective_calculator.dominates(objectives, entry['objectives']):
                retained.append(entry)
        retained.append({
            'solution': self._clean_solution(solution),
            'objectives': copy.deepcopy(objectives),
            'iteration': int(iteration),
            'crowding_distance': 0.0,
            'solution_signature': self._solution_signature(solution)
        })
        self.pareto_archive = retained
        self._prune_archive()
        return True

    def _prune_archive(self):
        while len(self.pareto_archive) > self.archive_size:
            objectives = [entry['objectives'] for entry in self.pareto_archive]
            distances = self.objective_calculator.crowding_distances(objectives)
            for index, distance in enumerate(distances):
                self.pareto_archive[index]['crowding_distance'] = float(distance)
            finite_indices = [i for i, distance in enumerate(distances) if np.isfinite(distance)]
            if finite_indices:
                remove_index = min(finite_indices, key=lambda i: distances[i])
            else:
                remove_index = len(self.pareto_archive) - 1
            del self.pareto_archive[remove_index]

    def _refresh_best_from_archive(self):
        if not self.pareto_archive:
            self.best_solution = None
            self.best_objective = float('inf')
            self.best_objectives = {}
            return
        objectives = [entry['objectives'] for entry in self.pareto_archive]
        distances = self.objective_calculator.crowding_distances(objectives)
        for index, distance in enumerate(distances):
            self.pareto_archive[index]['crowding_distance'] = float(distance)
        index, score, normalized = self.objective_calculator.select_compromise_solution(objectives)
        entry = self.pareto_archive[index]
        self.best_solution = copy.deepcopy(entry['solution'])
        self.best_objective = float(score)
        self.best_objectives = copy.deepcopy(entry['objectives'])
        self.best_objectives['normalized_cost'] = float(normalized[0])
        self.best_objectives['normalized_travel_time'] = float(normalized[1])
        self.best_objectives['normalized_ghg'] = float(normalized[2])
        self.best_objectives['compromise_score'] = float(score)

    def _select_restart_entry(self):
        if len(self.pareto_archive) == 1:
            return self.pareto_archive[0]
        distances = self.objective_calculator.crowding_distances([entry['objectives'] for entry in self.pareto_archive])
        order = list(np.argsort(-np.nan_to_num(distances, nan=0.0, posinf=1e12)))
        pool_size = max(1, len(order) // 2)
        return self.pareto_archive[random.choice(order[:pool_size])]

    def _select_restart_entries(self, count):
        if not self.pareto_archive:
            return []
        distances = self.objective_calculator.crowding_distances([entry['objectives'] for entry in self.pareto_archive])
        order = list(np.argsort(-np.nan_to_num(distances, nan=0.0, posinf=1e12)))
        result = []
        for i in range(count):
            result.append(self.pareto_archive[order[i % len(order)]])
        return result

    def _compromise_score(self, objectives):
        items = [entry['objectives'] for entry in self.pareto_archive]
        if not items:
            return float('inf')
        items.append(objectives)
        vectors = np.vstack([self.objective_calculator.objective_vector(item) for item in items])
        normalized = self.objective_calculator.normalize_vectors(vectors)
        vector = normalized[-1]
        return float(np.sqrt(np.sum(vector ** 2 * self.objective_calculator.compromise_weights)))

    def _record_history(self, iteration, temperature, objectives, accepted, archive_added, relation, probability, chain):
        self.optimization_history.append({
            'iteration': int(iteration),
            'temperature': float(temperature),
            'objective': self._compromise_score(objectives) if np.all(np.isfinite(self.objective_calculator.objective_vector(objectives))) else float('inf'),
            'total_cost': float(objectives.get('total_cost', float('inf'))),
            'total_travel_time': float(objectives.get('total_travel_time', float('inf'))),
            'total_ghg': float(objectives.get('total_ghg', float('inf'))),
            'accepted': bool(accepted),
            'archive_added': bool(archive_added),
            'dominance_relation': relation,
            'acceptance_probability': float(probability),
            'archive_size': len(self.pareto_archive),
            'chain': int(chain)
        })

    def _manage_history_memory(self):
        max_history_items = int(self.optimization_config.get('max_history_items', 1000))
        if len(self.optimization_history) > max_history_items:
            self._save_partial_history()
            self.optimization_history = []
            gc.collect()

    def _output_current_best_solution(self, iteration, temperature):
        if self.best_solution is None:
            return
        output_dir = Path(self.config['path_config']['output_directory']) / 'intermediate'
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            'iteration': int(iteration),
            'temperature': float(temperature),
            'compromise_score': float(self.best_objective),
            'pareto_archive_size': len(self.pareto_archive),
            'objectives': {
                'total_cost': float(self.best_objectives['total_cost']),
                'total_travel_time': float(self.best_objectives['total_travel_time']),
                'total_ghg': float(self.best_objectives['total_ghg'])
            },
            'solution_stats': {
                'vehicle_types': len(set(v['Type'] for v in self.best_solution['vehicle_assignments'].values() if isinstance(v, dict))),
                'station_count': len(self.best_solution['station_config']),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        summary_file = output_dir / f"summary_iter_{iteration}.json"
        with open(summary_file, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        if self.optimization_config.get('save_full_intermediate', False):
            with open(output_dir / f"solution_iter_{iteration}.pkl", 'wb') as file:
                pickle.dump(self.best_solution, file)

    def _save_partial_history(self):
        if not self.optimization_history:
            return
        output_dir = Path(self.config['path_config']['output_directory'])
        os.makedirs(output_dir, exist_ok=True)
        if self._partial_history_file is None:
            self._partial_history_file = output_dir / "optimization_history_partial.csv"
        frame = self._history_frame(self.optimization_history)
        frame.to_csv(
            self._partial_history_file,
            mode='a',
            header=not self._partial_history_file.exists(),
            index=False
        )

    @staticmethod
    def _clean_solution(solution):
        cleaned = copy.deepcopy(solution)
        cleaned.pop('memmap_suffix', None)
        return cleaned

    def _get_hrs_types(self):
        types = list(self.scheduler.data.get('hydrogen_stations', {}).get('dict', {}).keys())
        types = [str(item) for item in types if str(item).startswith('HRS_')]
        return sorted(types) if types else ['HRS_01']

    def _empty_station_config(self):
        return {'CS': 0, 'FCP': 0, 'SCP': 0, 'BSS': 0}

    def _has_any_facility(self, config):
        if config.get('CS', 0) > 0 or config.get('BSS', 0) > 0:
            return True
        return any(key.startswith('HRS_') and value > 0 for key, value in config.items())

    def _activate_facility(self, config, facility):
        result = dict(config)
        if facility == 'CS':
            result['CS'] = 1
            result['FCP'] = max(1, int(result.get('FCP', 0) or random.randint(2, 5)))
            result['SCP'] = max(1, int(result.get('SCP', 0) or random.randint(3, 8)))
        elif facility == 'BSS':
            result['BSS'] = 1
        elif facility == 'HRS':
            for key in list(result.keys()):
                if key.startswith('HRS_'):
                    result.pop(key, None)
            result[random.choice(self._get_hrs_types())] = 1
        return result

    def _deactivate_facility(self, config, facility):
        result = dict(config)
        if facility == 'CS':
            result['CS'] = 0
            result['FCP'] = 0
            result['SCP'] = 0
        elif facility == 'BSS':
            result['BSS'] = 0
        elif facility == 'HRS':
            for key in list(result.keys()):
                if key.startswith('HRS_'):
                    result.pop(key, None)
        return result

    def _random_station_config(self):
        config = self._empty_station_config()
        facilities = ['CS', 'BSS', 'HRS']
        count = random.randint(1, len(facilities))
        for facility in random.sample(facilities, count):
            config = self._activate_facility(config, facility)
        return config

    def _ensure_infrastructure_coverage(self, solution):
        vehicle_types = [v['Type'] for v in solution['vehicle_assignments'].values() if isinstance(v, dict) and 'Type' in v]
        needs_bev = any(vehicle_type.startswith('BEV_') for vehicle_type in vehicle_types)
        needs_hfcv = any(vehicle_type.startswith('HFCV_') for vehicle_type in vehicle_types)
        station_config = solution['station_config']
        all_stations = [str(index) for index in self.scheduler.stations.index]
        unused = [station_id for station_id in all_stations if station_id not in station_config]
        random.shuffle(unused)
        max_stations = int(self.optimization_config['constraints']['max_stations'])
        has_cs = any(config.get('CS', 0) > 0 for config in station_config.values())
        has_bss = any(config.get('BSS', 0) > 0 for config in station_config.values())
        has_hrs = any(any(key.startswith('HRS_') and value > 0 for key, value in config.items()) for config in station_config.values())
        requirements = []
        if needs_bev and not has_cs:
            requirements.append('CS')
        if needs_bev and not has_bss:
            requirements.append('BSS')
        if needs_hfcv and not has_hrs:
            requirements.append('HRS')
        for facility in requirements:
            if len(station_config) >= max_stations or not unused:
                break
            station_id = unused.pop()
            config = self._empty_station_config()
            station_config[station_id] = self._activate_facility(config, facility)

    def _generate_initial_solution(self):
        if hasattr(self.scheduler, 'trajectory_metadata') and self.scheduler.trajectory_metadata is not None:
            metadata = self.scheduler.trajectory_metadata
            file_path = metadata['file_path']
            header = metadata['header']
            total_count = int(metadata['total_count'])
            max_trajectories = self.config.get('simulation', {}).get('max_trajectories')
            assignment_limit = self.optimization_config.get('assignment_limit')
            limit = total_count
            if max_trajectories:
                limit = min(limit, int(max_trajectories))
            if assignment_limit:
                limit = min(limit, int(assignment_limit))
            positions = metadata['positions'][:limit]
            vehicles_data = self.scheduler.data['vehicles']
            bev_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('BEV_')]
            hfcv_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('HFCV_')]
            vehicle_assignments = {}
            index = {name: position for position, name in enumerate(header)}
            vehicle_col = index.get('VehicleID', 0)
            cargo_col = index.get('核定载质量')
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for position in positions:
                        file.seek(position)
                        parts = file.readline().strip().split('\t')
                        if len(parts) != len(header):
                            continue
                        vehicle_id = str(parts[vehicle_col])
                        if vehicle_id in vehicle_assignments:
                            continue
                        try:
                            cargo_weight = float(parts[cargo_col]) if cargo_col is not None else 0.0
                        except Exception:
                            cargo_weight = 0.0
                        category = 'BEV' if random.random() < 0.7 else 'HFCV'
                        category_types = bev_types if category == 'BEV' else hfcv_types
                        suitable = [vehicle_type for vehicle_type in category_types if self._is_suitable_cargo_weight(vehicle_type, cargo_weight)]
                        if not suitable:
                            category_types = hfcv_types if category == 'BEV' else bev_types
                            suitable = [vehicle_type for vehicle_type in category_types if self._is_suitable_cargo_weight(vehicle_type, cargo_weight)]
                        if not suitable:
                            suitable = bev_types + hfcv_types
                        if not suitable:
                            raise RuntimeError("No BEV or HFCV vehicle types are available")
                        vehicle_type = random.choice(suitable)
                        vehicle_assignments[vehicle_id] = {
                            'Type': vehicle_type,
                            'ActualVehicleID': vehicle_id,
                            'CargoWeight': cargo_weight
                        }
            except Exception as exc:
                self.logger.error(f"Failed to generate assignments from trajectory metadata: {exc}")
                return self._create_default_solution()
            stations = self.scheduler.data['stations']
            station_config = {}
            station_fraction = float(self.optimization_config.get('initial_station_fraction', 0.20))
            station_count = min(int(self.optimization_config['constraints']['max_stations']), int(len(stations) * station_fraction))
            station_count = max(min(int(self.optimization_config['constraints'].get('min_stations', 1)), len(stations)), min(station_count, len(stations)))
            if station_count > 0:
                selected = random.sample([str(index) for index in stations.index], station_count)
                for station_id in selected:
                    station_config[station_id] = self._random_station_config()
            solution = {'vehicle_assignments': vehicle_assignments, 'station_config': station_config}
            self._ensure_infrastructure_coverage(solution)
            return solution
        return self._create_default_solution()

    def _create_default_solution(self):
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('BEV_')]
        hfcv_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('HFCV_')]
        available_types = bev_types + hfcv_types
        if not available_types:
            raise RuntimeError("No vehicle types are available")
        vehicle_assignments = {}
        for index in range(1, 101):
            vehicle_type = random.choice(available_types)
            vehicle_data = self.scheduler.data['vehicles']['dict'].get(vehicle_type, {})
            vehicle_assignments[str(index)] = {
                'Type': vehicle_type,
                'ActualVehicleID': str(index),
                'CargoWeight': float(vehicle_data.get('CargoWeight', 0.0))
            }
        stations = self.scheduler.data['stations']
        min_stations = min(int(self.optimization_config['constraints'].get('min_stations', 1)), len(stations))
        station_config = {}
        if min_stations > 0:
            for station_id in random.sample([str(index) for index in stations.index], min_stations):
                station_config[station_id] = self._random_station_config()
        solution = {'vehicle_assignments': vehicle_assignments, 'station_config': station_config}
        self._ensure_infrastructure_coverage(solution)
        return solution

    def _generate_neighbor_solution(self, solution):
        new_solution = {
            'vehicle_assignments': solution['vehicle_assignments'].copy(),
            'station_config': solution['station_config'].copy()
        }
        perturbations = [
            'change_vehicle_type',
            'change_station_type',
            'add_station',
            'remove_station',
            'adjust_charging_posts'
        ]
        if len(self._get_hrs_types()) > 1:
            perturbations.append('adjust_hrs_capacity')
        perturbation_type = random.choice(perturbations)
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
        elif perturbation_type == 'adjust_hrs_capacity':
            self._perturb_hrs_capacity(new_solution)
        self._ensure_infrastructure_coverage(new_solution)
        return new_solution

    def _perturb_vehicle_types(self, solution):
        vehicle_assignments = solution['vehicle_assignments']
        if not vehicle_assignments:
            return
        vehicles_data = self.scheduler.data['vehicles']
        bev_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('BEV_')]
        hfcv_types = [str(t) for t in vehicles_data['df']['Type'] if str(t).startswith('HFCV_')]
        fraction = float(self.optimization_config.get('vehicle_mutation_fraction', 0.10))
        change_count = min(len(vehicle_assignments), max(1, int(len(vehicle_assignments) * fraction)))
        for vehicle_id in random.sample(list(vehicle_assignments.keys()), change_count):
            current = dict(vehicle_assignments[vehicle_id])
            current_type = current['Type']
            cargo_weight = float(current.get('CargoWeight', self.scheduler.data['vehicles']['dict'].get(current_type, {}).get('CargoWeight', 0.0)))
            current_category = 'BEV' if current_type.startswith('BEV_') else 'HFCV'
            switch_probability = float(self.optimization_config.get('vehicle_category_switch_probability', 0.20))
            new_category = ('HFCV' if current_category == 'BEV' else 'BEV') if random.random() < switch_probability else current_category
            candidates = bev_types if new_category == 'BEV' else hfcv_types
            suitable = [vehicle_type for vehicle_type in candidates if self._is_suitable_cargo_weight(vehicle_type, cargo_weight)]
            if not suitable:
                suitable = candidates
            if suitable:
                current['Type'] = random.choice(suitable)
                vehicle_assignments[vehicle_id] = current

    def _perturb_station_types(self, solution):
        station_config = solution['station_config']
        if not station_config:
            return
        change_count = min(len(station_config), max(1, int(len(station_config) * 0.20)))
        for station_id in random.sample(list(station_config.keys()), change_count):
            config = dict(station_config[station_id])
            facility = random.choice(['CS', 'BSS', 'HRS'])
            if facility == 'CS':
                active = config.get('CS', 0) > 0
            elif facility == 'BSS':
                active = config.get('BSS', 0) > 0
            else:
                active = any(key.startswith('HRS_') and value > 0 for key, value in config.items())
            config = self._deactivate_facility(config, facility) if active else self._activate_facility(config, facility)
            if not self._has_any_facility(config):
                alternatives = [item for item in ['CS', 'BSS', 'HRS'] if item != facility]
                config = self._activate_facility(config, random.choice(alternatives))
            station_config[station_id] = config

    def _perturb_add_station(self, solution):
        station_config = solution['station_config']
        all_stations = {str(index) for index in self.scheduler.data['stations'].index}
        unused = list(all_stations - set(station_config.keys()))
        max_stations = int(self.optimization_config['constraints']['max_stations'])
        if len(station_config) >= max_stations or not unused:
            return
        add_count = min(len(unused), max_stations - len(station_config), max(1, int(max(1, len(station_config)) * 0.05)))
        for station_id in random.sample(unused, add_count):
            station_config[station_id] = self._random_station_config()

    def _perturb_remove_station(self, solution):
        station_config = solution['station_config']
        min_stations = int(self.optimization_config['constraints'].get('min_stations', 1))
        if len(station_config) <= min_stations:
            return
        remove_count = min(len(station_config) - min_stations, max(1, int(len(station_config) * 0.05)))
        for station_id in random.sample(list(station_config.keys()), remove_count):
            del station_config[station_id]

    def _perturb_charging_posts(self, solution):
        station_config = solution['station_config']
        charging_stations = [station_id for station_id, config in station_config.items() if config.get('CS', 0) > 0]
        if not charging_stations:
            return
        adjust_count = min(len(charging_stations), max(1, int(len(charging_stations) * 0.50)))
        for station_id in random.sample(charging_stations, adjust_count):
            config = dict(station_config[station_id])
            fcp = int(config.get('FCP', 1))
            scp = int(config.get('SCP', 1))
            config['FCP'] = max(1, fcp + random.choice([-2, -1, 1, 2]))
            config['SCP'] = max(1, scp + random.choice([-3, -2, -1, 1, 2, 3]))
            station_config[station_id] = config

    def _perturb_hrs_capacity(self, solution):
        hrs_types = self._get_hrs_types()
        if len(hrs_types) <= 1:
            return
        hrs_stations = []
        for station_id, config in solution['station_config'].items():
            if any(key.startswith('HRS_') and value > 0 for key, value in config.items()):
                hrs_stations.append(station_id)
        if not hrs_stations:
            return
        station_id = random.choice(hrs_stations)
        config = dict(solution['station_config'][station_id])
        active = [key for key, value in config.items() if key.startswith('HRS_') and value > 0]
        alternatives = [hrs_type for hrs_type in hrs_types if hrs_type not in active]
        if alternatives:
            for key in list(config.keys()):
                if key.startswith('HRS_'):
                    config.pop(key, None)
            config[random.choice(alternatives)] = 1
            solution['station_config'][station_id] = config

    def _apply_solution(self, solution):
        self.scheduler.configure_infrastructure(solution['station_config'])
        self.scheduler.assign_vehicle_types(solution['vehicle_assignments'])

    def _get_city_from_road(self, road_id):
        try:
            if road_id in self.scheduler.road_network.index:
                return self.scheduler.road_network.loc[road_id]['CityCode']
        except Exception:
            pass
        return 0

    def _is_suitable_cargo_weight(self, vehicle_type, cargo_weight):
        try:
            vehicle_data = self.scheduler.data['vehicles']['dict'][vehicle_type]
            return abs(float(vehicle_data['CargoWeight']) - float(cargo_weight)) < 0.01
        except Exception:
            return False

    @staticmethod
    def _history_frame(history):
        rows = []
        for entry in history:
            rows.append({
                'iteration': entry['iteration'],
                'temperature': entry['temperature'],
                'objective': entry['objective'],
                'cost': entry['total_cost'],
                'travel_time': entry['total_travel_time'],
                'ghg': entry['total_ghg'],
                'total_cost': entry['total_cost'],
                'total_travel_time': entry['total_travel_time'],
                'total_ghg': entry['total_ghg'],
                'accepted': entry['accepted'],
                'archive_added': entry['archive_added'],
                'dominance_relation': entry['dominance_relation'],
                'acceptance_probability': entry['acceptance_probability'],
                'archive_size': entry['archive_size'],
                'chain': entry['chain']
            })
        return pd.DataFrame(rows)

    def _save_optimization_history(self):
        output_dir = Path(self.config['path_config']['output_directory'])
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        if self._partial_history_file is not None and self._partial_history_file.exists():
            frames.append(pd.read_csv(self._partial_history_file))
        if self.optimization_history:
            frames.append(self._history_frame(self.optimization_history))
        history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        history.to_csv(output_dir / "optimization_history.csv", index=False)
        if self._partial_history_file is not None and self._partial_history_file.exists():
            self._partial_history_file.unlink()
        self._refresh_best_from_archive()
        with open(output_dir / "best_solution.pkl", 'wb') as file:
            pickle.dump(self.best_solution, file)
        pd.DataFrame([self.best_objectives]).to_csv(output_dir / "best_objectives.csv", index=False)
        pareto_rows = []
        for index, entry in enumerate(self.pareto_archive):
            objectives = entry['objectives']
            solution = entry['solution']
            assignments = solution.get('vehicle_assignments', {})
            bev_count = sum(1 for value in assignments.values() if isinstance(value, dict) and str(value.get('Type', '')).startswith('BEV_'))
            hfcv_count = sum(1 for value in assignments.values() if isinstance(value, dict) and str(value.get('Type', '')).startswith('HFCV_'))
            pareto_rows.append({
                'pareto_id': index,
                'iteration': entry.get('iteration', 0),
                'total_cost': objectives['total_cost'],
                'total_travel_time': objectives['total_travel_time'],
                'total_ghg': objectives['total_ghg'],
                'crowding_distance': entry.get('crowding_distance', 0.0),
                'station_count': len(solution.get('station_config', {})),
                'bev_assignments': bev_count,
                'hfcv_assignments': hfcv_count
            })
        pd.DataFrame(pareto_rows).to_csv(output_dir / "pareto_front.csv", index=False)
        with open(output_dir / "pareto_solutions.pkl", 'wb') as file:
            pickle.dump(self.pareto_archive, file)
        best_summary = {
            'compromise_score': self.best_objective,
            'objectives': {
                'total_cost': self.best_objectives.get('total_cost'),
                'total_travel_time': self.best_objectives.get('total_travel_time'),
                'total_ghg': self.best_objectives.get('total_ghg')
            },
            'pareto_archive_size': len(self.pareto_archive)
        }
        with open(output_dir / "best_compromise.json", 'w', encoding='utf-8') as file:
            json.dump(best_summary, file, indent=2, ensure_ascii=False)

    def export_results(self):
        if self.best_solution is None:
            raise RuntimeError("No best compromise solution is available")
        self._apply_solution(self.best_solution)
        self.scheduler.simulate(max_trajectories=self.config.get('simulation', {}).get('max_trajectories'))
        self.scheduler.export_results()
