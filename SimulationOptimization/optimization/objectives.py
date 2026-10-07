"""
Author: Shiqi (Anya) WANG
Date: 2025/3/8
Description: Objective function implementation module
Defines calculation methods for total cost, total travel time, and total GHG emissions
"""

import logging
import numpy as np


class ObjectiveCalculator:
    def __init__(self, simulation_scheduler, config, logger=None):
        self.scheduler = simulation_scheduler
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        optimization_config = config.get('optimization', {})
        weights = optimization_config.get('compromise_weights', optimization_config.get('objective_weights', {}))
        self.compromise_weights = np.array([
            float(weights.get('cost', 1.0 / 3.0)),
            float(weights.get('travel_time', 1.0 / 3.0)),
            float(weights.get('ghg', 1.0 / 3.0))
        ], dtype=float)
        if not np.all(np.isfinite(self.compromise_weights)) or np.sum(self.compromise_weights) <= 0:
            self.compromise_weights = np.ones(3, dtype=float) / 3.0
        else:
            self.compromise_weights = np.maximum(self.compromise_weights, 0.0)
            total = np.sum(self.compromise_weights)
            self.compromise_weights = self.compromise_weights / total if total > 0 else np.ones(3, dtype=float) / 3.0
        # Normalization is only for post-search compromise reporting.
        # It is intentionally not required by Pareto dominance decisions.
        self.normalization_reference = None

    def calculate_total_cost(self, simulation_results):
        cost_records = simulation_results.get('cost_records')
        if cost_records is None:
            self.logger.error("Missing cost records in simulation results")
            return 0.0
        capital_cost = float(cost_records.get('capital_cost', 0.0))
        energy_cost = float(cost_records.get('energy_cost', 0.0))
        return capital_cost + energy_cost

    def calculate_total_travel_time(self, simulation_results):
        driving_statistics = simulation_results.get('driving_statistics')
        if driving_statistics is None:
            self.logger.error("Missing driving statistics in simulation results")
            return 0.0
        driving_time = sum(float(d.get('TotalDrivingTime', 0.0)) for d in driving_statistics)
        rest_time = sum(float(d.get('TotalRestTime', 0.0)) for d in driving_statistics)
        refill_time = sum(float(d.get('EnergyRefillTime', 0.0)) for d in driving_statistics)
        detour_time = sum(float(d.get('DetourTime', 0.0)) for d in driving_statistics)
        queue_time = 0.0
        for record in simulation_results.get('station_service_records', []):
            if record.get('EventType') == 'ARRIVAL':
                queue_time += float(record.get('WaitingTime', 0.0))
        return driving_time + rest_time + refill_time + detour_time + queue_time

    def calculate_total_ghg(self, simulation_results):
        ghg_records = simulation_results.get('ghg_records')
        if ghg_records is None:
            self.logger.error("Missing GHG records in simulation results")
            return 0.0
        infrastructure_ghg = float(ghg_records.get('infrastructure_ghg', 0.0))
        vehicle_ghg = float(ghg_records.get('vehicle_ghg', 0.0))
        return infrastructure_ghg + vehicle_ghg

    def calculate_objectives(self, simulation_results):
        total_cost = self.calculate_total_cost(simulation_results)
        total_travel_time = self.calculate_total_travel_time(simulation_results)
        total_ghg = self.calculate_total_ghg(simulation_results)
        return {
            'total_cost': total_cost,
            'total_travel_time': total_travel_time,
            'total_ghg': total_ghg,
            'objective_vector': (total_cost, total_travel_time, total_ghg)
        }

    @staticmethod
    def objective_vector(objectives):
        if 'objective_vector' in objectives:
            return np.asarray(objectives['objective_vector'], dtype=float)
        return np.asarray([
            objectives.get('total_cost', np.inf),
            objectives.get('total_travel_time', np.inf),
            objectives.get('total_ghg', np.inf)
        ], dtype=float)

    @staticmethod
    def dominates(first, second, atol=1e-12):
        a = ObjectiveCalculator.objective_vector(first)
        b = ObjectiveCalculator.objective_vector(second)
        if not np.all(np.isfinite(a)):
            return False
        if not np.all(np.isfinite(b)):
            return True
        no_worse = np.all(a <= b + atol)
        strictly_better = np.any(a < b - atol)
        return bool(no_worse and strictly_better)

    def set_normalization_reference(self, objectives_list):
        if not objectives_list:
            return
        matrix = np.vstack([self.objective_vector(o) for o in objectives_list])
        self.normalization_reference = {
            'ideal': np.min(matrix, axis=0),
            'nadir': np.max(matrix, axis=0)
        }

    def normalize_vectors(self, vectors):
        matrix = np.asarray(vectors, dtype=float)
        if matrix.size == 0:
            return matrix
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if self.normalization_reference is None:
            self.set_normalization_reference([{'objective_vector': row} for row in matrix])
        ideal = self.normalization_reference['ideal']
        nadir = self.normalization_reference['nadir']
        span = nadir - ideal
        span = np.where(span > 0, span, 1.0)
        return (matrix - ideal) / span

    @staticmethod
    def crowding_distances(objectives_list):
        n = len(objectives_list)
        if n == 0:
            return np.array([], dtype=float)
        if n <= 2:
            return np.full(n, np.inf, dtype=float)
        vectors = np.vstack([ObjectiveCalculator.objective_vector(o) for o in objectives_list])
        distances = np.zeros(n, dtype=float)
        for dimension in range(vectors.shape[1]):
            order = np.argsort(vectors[:, dimension], kind='mergesort')
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            minimum = vectors[order[0], dimension]
            maximum = vectors[order[-1], dimension]
            span = maximum - minimum
            if span <= 0:
                continue
            for position in range(1, n - 1):
                index = order[position]
                if np.isinf(distances[index]):
                    continue
                previous_value = vectors[order[position - 1], dimension]
                next_value = vectors[order[position + 1], dimension]
                distances[index] += (next_value - previous_value) / span
        return distances

    def select_compromise_solution(self, objectives_list):
        if not objectives_list:
            return None, float('inf'), None
        vectors = np.vstack([self.objective_vector(o) for o in objectives_list])
        if self.normalization_reference is None:
            self.set_normalization_reference(objectives_list)
        normalized = self.normalize_vectors(vectors)
        weighted_squared = normalized ** 2 * self.compromise_weights.reshape(1, -1)
        scores = np.sqrt(np.sum(weighted_squared, axis=1))
        index = int(np.argmin(scores))
        return index, float(scores[index]), normalized[index]
