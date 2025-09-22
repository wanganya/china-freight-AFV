"""
Author: Shiqi (Anya) WANG
Date: 2025/6/23
Description: This script is used to generate the detailed freight trips for each freight delivery task.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    QgsPointXY,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsFeatureRequest,
    QgsWkbTypes
)
from qgis.analysis import (
    QgsGraphBuilder,
    QgsGraphAnalyzer
)
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='traffic_assignment.log'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for storing paths and parameters"""
    def __init__(self):
        self.ROAD_NETWORK_PATH = r"D:\Road\OSMOutput\OSM_Highway_no2nd_Code.gpkg"
        self.OD_DATA_PATH = r"D:\ChinaTrip\Output\EFVTrip2019CoorReviseCityDiffODCargoExpand20250217.txt"
        self.VOLUME_BASE_PATH = r"D:\Road\OSMOutput\002VolumeDayUpdate"
        self.MONTH_DIST_PATH = r"D:\Road\OSMOutput\001VolumeMonUpdate\MonthDistribution.txt"
        self.OUTPUT_BASE_PATH = r"D:\ChinaTrip\Output\Trajectory"
        
        self.SEASONS = {
            'Winter': ('20190114', '20190120'),
            'Spring': ('20190418', '20190424'),
            'Summer': ('20190716', '20190722'),
            'Autumn': ('20191106', '20191112')
        }
        
        self.ALPHA = 0.15
        self.BETA = 4.0
        
        self.DEFAULT_CAPACITIES = {
            'motorway': 55000,
            'motorway_link': 55000,
            'trunk': 20000,
            'trunk_link': 20000,
            'primary': 15000,
            'primary_link': 15000
        }
        
        self.K_PATHS = 50
        self.MAX_ITERATIONS = 20
        
        self.USE_TARGET_MATCHING = True
        self.TARGET_MATCHING_MAX_ITERATIONS = 3000
        self.HEURISTIC_MAX_ITERATIONS = 3000
        
        Path(self.OUTPUT_BASE_PATH).mkdir(parents=True, exist_ok=True)

class DataPreprocessor:
    """Data preprocessing class"""
    def __init__(self, config: Config):
        self.config = config

    def preprocess_od_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess OD data, return full data and unique OD pairs"""
        logger.info("Starting OD data preprocessing...")
        try:
            df = pd.read_csv(self.config.OD_DATA_PATH, sep='\t', encoding='utf-8')
            
            df = df[
                (df['Coefficient'] > 0) &
                df['OLon'].notna() &
                df['OLat'].notna() &
                df['DLon'].notna() &
                df['DLat'].notna()
            ]
            
            df['OD_Key'] = df.apply(
                lambda x: f"{x['OLon']}_{x['OLat']}_{x['DLon']}_{x['DLat']}",
                axis=1
            )
            
            unique_ods = df.drop_duplicates(subset=['OD_Key']).copy()
            unique_ods['UniqueID'] = range(len(unique_ods))
            
            od_key_to_unique_id = dict(zip(unique_ods['OD_Key'], unique_ods['UniqueID']))
            df['UniqueID'] = df['OD_Key'].map(od_key_to_unique_id)
            
            output_path = os.path.join(self.config.OUTPUT_BASE_PATH, 'preprocessed_od.txt')
            df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
            
            unique_ods_path = os.path.join(self.config.OUTPUT_BASE_PATH, 'unique_ods.txt')
            unique_ods.to_csv(unique_ods_path, sep='\t', index=False, encoding='utf-8')
            
            logger.info(f"Total OD records: {len(df)}, Unique OD pairs: {len(unique_ods)}")
            
            return df, unique_ods
            
        except Exception as e:
            logger.error(f"OD data preprocessing failed: {str(e)}")
            raise

class CoefficientCorrector:
    """Coefficient correction class"""
    def __init__(self, config: Config):
        self.config = config
    
    def correct_coefficients(self, od_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Correct coefficients"""
        logger.info("Starting coefficient correction...")
        try:
            month_dist = pd.read_csv(self.config.MONTH_DIST_PATH, sep='\t', encoding='utf-8')
            
            month_dist['Month'] = month_dist['Month'].astype(str).str.zfill(2)

            seasonal_data = {}
            for season, (start_date, end_date) in self.config.SEASONS.items():
                month = start_date[4:6]
                
                month_ratio = month_dist[month_dist['Month'] == month]['MonthRatio'].iloc[0]
                                
                season_data = od_data.copy()
                season_data['Coefficient'] = (
                    season_data['Coefficient'] * 
                    month_ratio * 
                    7/30
                )
                
                season_data['Coefficient'] = season_data['Coefficient'].round().astype(int)
                
                output_path = os.path.join(
                    self.config.OUTPUT_BASE_PATH,
                    f'corrected_coefficient_{season}.txt'
                )
                season_data.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
                
                seasonal_data[season] = season_data
            
            return seasonal_data
            
        except Exception as e:
            logger.error(f"Coefficient correction failed: {str(e)}")
            raise

class BPRCalculator:
    """BPR function calculator"""
    def __init__(self, config: Config):
        self.config = config
        
    def get_default_capacity(self, road_type: str) -> float:
        """Get default capacity"""
        return self.config.DEFAULT_CAPACITIES.get(road_type, 15000)
        
    def calculate_travel_time(self,
                            free_flow_time: float,
                            volume: float,
                            capacity: float) -> float:
        """
        Calculate actual travel time using BPR function
        travel_time = free_flow_time * (1 + α * (volume/capacity)^β)
        """
        if capacity <= 0:
            return free_flow_time
        
        ratio = volume / capacity
        return free_flow_time * (1 + self.config.ALPHA * (ratio ** self.config.BETA))

class FreightVolumeCalculator:
    """Freight volume calculator"""
    def __init__(self, config: Config):
        self.config = config
        self.time_slots = [
            (1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24)
        ]
        
    def _process_daily_volume(self, file_path: str, date_str: str) -> pd.DataFrame:
        """Process daily volume data"""
        try:
            df = pd.read_excel(file_path)
            
            required_columns = [
                '中货车加权流量', '大货车加权流量', '特大货加权流量',
                '集装箱加权流量', '拖拉机加权流量'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in {file_path}: {missing_columns}")
                return None
            
            df['hourly_vehicle_count'] = (
                df['中货车加权流量'] +
                df['大货车加权流量'] +
                df['特大货加权流量'] +
                df['集装箱加权流量'] +
                df['拖拉机加权流量']
            )
            
            df['date'] = date_str
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _calculate_4hour_constraints(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate 4-hour time slot constraints"""
        constraints_list = []
        
        for slot_idx, (start_hour, end_hour) in enumerate(self.time_slots):
            slot_data = daily_data[
                (daily_data['小时'] >= start_hour) & 
                (daily_data['小时'] <= end_hour)
            ]
            
            if len(slot_data) == 0:
                continue
                
            grouped = slot_data.groupby('路线简码')['hourly_vehicle_count'].sum().reset_index()
            grouped['time_slot'] = slot_idx
            grouped['start_hour'] = start_hour
            grouped['end_hour'] = end_hour
            grouped['date'] = daily_data['date'].iloc[0]
            grouped.rename(columns={'hourly_vehicle_count': 'constraint_4hour'}, inplace=True)
            
            constraints_list.append(grouped)
        
        if constraints_list:
            return pd.concat(constraints_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_total_volume(self) -> Dict[str, pd.DataFrame]:
        """Calculate total freight volume with 4-hour intervals"""
        logger.info("Calculating total freight volume (4-hour intervals)...")
        try:
            volume_data = {}
            
            for season, (start_date, end_date) in self.config.SEASONS.items():
                logger.info(f"Processing {season} season data...")
                
                current_date = datetime.strptime(start_date, '%Y%m%d')
                end_date = datetime.strptime(end_date, '%Y%m%d')
                
                season_constraints = []
                total_days = (end_date - current_date).days + 1
                
                with tqdm(total=total_days, desc=f"Processing {season} data") as pbar:
                    while current_date <= end_date:
                        date_str = current_date.strftime('%Y%m%d')
                        file_path = os.path.join(
                            self.config.VOLUME_BASE_PATH,
                            f"{date_str}_update.xls"
                        )
                        
                        daily_volume = self._process_daily_volume(file_path, date_str)
                        if daily_volume is not None:
                            daily_constraints = self._calculate_4hour_constraints(daily_volume)
                            if len(daily_constraints) > 0:
                                season_constraints.append(daily_constraints)
                        
                        current_date += pd.Timedelta(days=1)
                        pbar.update(1)
                
                if not season_constraints:
                    logger.warning(f"No valid data for {season} season")
                    continue
                
                season_df = pd.concat(season_constraints, ignore_index=True)
                season_df['season'] = season
                
                output_path = os.path.join(
                    self.config.OUTPUT_BASE_PATH,
                    f'constraints_4hour_{season}.txt'
                )
                season_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
                
                summary_path = os.path.join(
                    self.config.OUTPUT_BASE_PATH,
                    f'constraints_summary_{season}.txt'
                )
                summary_df = season_df.groupby(['date', 'time_slot'])[
                    'constraint_4hour'
                ].agg(['mean', 'std', 'min', 'max']).reset_index()
                summary_df.to_csv(summary_path, sep='\t', index=False, encoding='utf-8')
                
                volume_data[season] = season_df
            
            return volume_data
            
        except Exception as e:
            logger.error(f"Freight volume calculation failed: {str(e)}")
            raise
    
    def get_volume_constraints_4hour(self, 
                                   season: str,
                                   date: str,
                                   time_slot: int) -> pd.DataFrame:
        """Get volume constraints for specific 4-hour time slot"""
        try:
            constraints_file = os.path.join(
                self.config.OUTPUT_BASE_PATH,
                f'constraints_4hour_{season}.txt'
            )
            
            if not os.path.exists(constraints_file):
                logger.warning(f"Constraints file not found: {constraints_file}")
                return None
            
            df = pd.read_csv(constraints_file, sep='\t')
            
            constraints = df[
                (df['date'] == date) & 
                (df['time_slot'] == time_slot)
            ][['路线简码', 'constraint_4hour', 'start_hour', 'end_hour']]
            
            return constraints
            
        except Exception as e:
            logger.error(f"Failed to get volume constraints: {str(e)}")
            raise

class NetworkAnalysis:
    """Network analysis class"""
    def __init__(self, config: Config):
        self.config = config
        self.qgs = None
        self.network_layer = None
        self.graph = None
        self.vertices = {}
        self.vertex_coords = {}
        self.edge_properties = {}
        self.edge_to_original_id = {}
        self.major_road_types = {
            'motorway', 'trunk', 'primary', 
            'motorway_link', 'trunk_link', 'primary_link'
        }
    
    def initialize_qgis(self):
        """Initialize QGIS environment"""
        if self.qgs is None:
            from qgis.core import QgsApplication
            self.qgs = QgsApplication([], False)
            self.qgs.setPrefixPath("C:/ProgramData/anaconda3/envs/DistributionOSM/Library", True)
            self.qgs.initQgis()    
        
    def __del__(self):
        """Clean up QGIS instance"""
        if self.qgs:
            self.qgs.exitQgis()

    def load_network(self):
        """Load network data"""
        logger.info("Loading network data...")
        try:
            self.initialize_qgis()
            
            self.network_layer = QgsVectorLayer(
                self.config.ROAD_NETWORK_PATH,
                "road_network",
                "ogr"
            )
            if not self.network_layer.isValid():
                raise Exception("Failed to load network layer")
            
            QgsProject.instance().addMapLayer(self.network_layer)
            
            builder = QgsGraphBuilder(
                self.network_layer.crs(),
                True,
                0.0001,
                "EPSG:4326"
            )
            
            self.add_vertices_and_edges(builder)
            self.graph = builder.graph()
            
            logger.info(f"Network loaded successfully, {len(self.vertices)} vertices")
            
        except Exception as e:
            logger.error(f"Network loading failed: {str(e)}")
            raise

    def add_vertices_and_edges(self, builder):
        """Add vertices and edges"""
        vertex_id = 0
        edge_counter = 0
        
        fields = self.network_layer.fields()
        id_field = None
        for field in fields:
            if field.name().lower() == 'id':
                id_field = field.name()
                break
        if not id_field:
            raise Exception("Could not find ID field in the network layer")
        
        for feature in tqdm(self.network_layer.getFeatures(), 
                          desc="Adding vertices and edges", 
                          total=self.network_layer.featureCount()):
            try:
                original_id = feature[id_field]
                geom = feature.geometry()
                if geom.wkbType() != QgsWkbTypes.MultiLineString:
                    continue
                
                attrs = {
                    'oneway': feature['oneway'],
                    'time_init': feature['time_init'],
                    'fclass': feature['fclass'],
                    'length': feature['length'],
                    'speed_init': feature['speed_init'],
                    'code': feature['Code']
                }
                
                multiline = geom.asMultiPolyline()
                for line in multiline:
                    start_point = QgsPointXY(line[0])
                    end_point = QgsPointXY(line[-1])
                    
                    start_key = (start_point.x(), start_point.y())
                    end_key = (end_point.x(), end_point.y())
                    
                    if start_key not in self.vertices:
                        self.vertices[start_key] = vertex_id
                        self.vertex_coords[vertex_id] = start_key
                        builder.addVertex(vertex_id, start_point)
                        vertex_id += 1
                    
                    if end_key not in self.vertices:
                        self.vertices[end_key] = vertex_id
                        self.vertex_coords[vertex_id] = end_key
                        builder.addVertex(vertex_id, end_point)
                        vertex_id += 1
                    
                    start_id = self.vertices[start_key]
                    end_id = self.vertices[end_key]
                    
                    if attrs['oneway'] in ['F', 'T']:
                        builder.addEdge(start_id, start_point, end_id, end_point, [attrs['time_init']])
                        self.edge_to_original_id[edge_counter] = original_id
                        self.edge_properties[edge_counter] = attrs
                        edge_counter += 1

                    if attrs['oneway'] in ['B', 'T']:
                        builder.addEdge(end_id, end_point, start_id, start_point, [attrs['time_init']])
                        self.edge_to_original_id[edge_counter] = original_id
                        self.edge_properties[edge_counter] = attrs
                        edge_counter += 1
                        
            except Exception as e:
                logger.warning(f"Error processing feature: {str(e)}")
                continue
    
    def find_nearest_vertices(self, point: QgsPointXY, k: int = 10) -> List[int]:
        """Find k nearest vertices to given point"""
        from heapq import heappush, heappop
        
        distances = []
        point_x, point_y = point.x(), point.y()
        
        for vertex_id, coord in self.vertex_coords.items():
            dist = ((coord[0] - point_x)**2 + (coord[1] - point_y)**2)**0.5
            heappush(distances, (dist, vertex_id))
        
        return [heappop(distances)[1] for _ in range(min(k, len(distances)))]
    
    def get_edge_fclass(self, edge_id: int) -> str:
        """Get edge road type"""
        try:
            return self.edge_properties.get(edge_id, {}).get('fclass', 'secondary')
        except:
            return 'secondary'

    def is_major_road(self, fclass: str) -> bool:
        """Check if it's a major road type"""
        return fclass in self.major_road_types

    def can_connect_to_next(self, current_fclass: str, next_fclass: str) -> bool:
        """
        Check if current road can connect to next road
        Rules:
        - Low-grade roads can connect to any road
        - High-grade roads can only connect to high-grade roads
        """
        if not self.is_major_road(current_fclass):
            return True
        else:
            return self.is_major_road(next_fclass)

    def path_contains_major_road(self, path_edges: List[int]) -> bool:
        """Check if path contains at least one major road segment"""
        for edge_id in path_edges:
            fclass = self.get_edge_fclass(edge_id)
            if self.is_major_road(fclass):
                return True
        return False

    def build_path_with_constraints(self, 
                                   start_vertex: int, 
                                   end_vertex: int,
                                   tree: List[int]) -> List[int]:
        """
        Build path with road grade constraints
        Returns edge ID list that meets constraints, or None if path doesn't meet constraints
        """
        if tree[end_vertex] == -1:
            return None
        
        path_edges = []
        current_vertex = end_vertex
        
        while current_vertex != start_vertex:
            if current_vertex == -1 or tree[current_vertex] == -1:
                return None
            
            edge_id = tree[current_vertex]
            path_edges.append(edge_id)
            
            edge = self.graph.edge(edge_id)
            current_vertex = edge.fromVertex()
        
        path_edges.reverse()
        
        for i in range(len(path_edges) - 1):
            current_fclass = self.get_edge_fclass(path_edges[i])
            next_fclass = self.get_edge_fclass(path_edges[i + 1])
            
            if not self.can_connect_to_next(current_fclass, next_fclass):
                return None
        
        if not self.path_contains_major_road(path_edges):
            return None
        
        return path_edges

    def convert_edge_path_to_vertex_path(self, edge_path: List[int]) -> List[str]:
        """Convert edge ID path to vertex ID path"""
        if not edge_path:
            return []
        
        vertex_path = []
        
        first_edge = self.graph.edge(edge_path[0])
        vertex_path.append(str(first_edge.fromVertex()))
        
        for edge_id in edge_path:
            edge = self.graph.edge(edge_id)
            vertex_path.append(str(edge.toVertex()))
        
        return vertex_path

    def find_k_paths_with_constraints(self, 
                                       start_point: QgsPointXY, 
                                       end_point: QgsPointXY,
                                       background_volume: Dict[str, float] = None) -> List[str]:
        """
        Find k paths that meet road grade constraints
        Returns path list, each path is a comma-separated string of original segment IDs
        """
        valid_paths = []
        paths_set = set()
        
        start_vertices = self.find_nearest_vertices(start_point, k=5)
        end_vertices = self.find_nearest_vertices(end_point, k=5)
        
        max_attempts = len(start_vertices) * len(end_vertices)
        attempts = 0
        
        for start_vertex in start_vertices:
            for end_vertex in end_vertices:
                if len(valid_paths) >= self.config.K_PATHS:
                    break
                
                attempts += 1
                if attempts > max_attempts:
                    break
                
                try:
                    tree, cost = QgsGraphAnalyzer.dijkstra(self.graph, start_vertex, 0)
                    
                    edge_path = self.build_path_with_constraints(start_vertex, end_vertex, tree)
                    
                    if edge_path is None:
                        continue
                    
                    original_path = []
                    for edge_id in edge_path:
                        original_id = self.edge_to_original_id.get(edge_id)
                        if original_id is not None:
                            original_path.append(str(original_id))
                    
                    if not original_path:
                        continue
                    
                    path_str = ','.join(original_path)
                    if path_str not in paths_set:
                        valid_paths.append(path_str)
                        paths_set.add(path_str)
                        
                except Exception as e:
                    logger.warning(f"Path search error: {str(e)}")
                    continue
            
            if len(valid_paths) >= self.config.K_PATHS:
                break
        
        if len(valid_paths) < self.config.K_PATHS and len(valid_paths) > 0:
            logger.info(f"Found {len(valid_paths)} constraint-compliant paths (target: {self.config.K_PATHS})")
        
        return valid_paths[:self.config.K_PATHS]

    def generate_paths_sequential(self, unique_ods: pd.DataFrame, season: str, logger: logging.Logger) -> None:
        """Generate paths sequentially (process unique OD pairs only)"""
        output_path = os.path.join(
            self.config.OUTPUT_BASE_PATH,
            f'paths_{season}.txt'
        )
        
        try:
            start_date = datetime.strptime(self.config.SEASONS[season][0], '%Y%m%d')
            end_date = datetime.strptime(self.config.SEASONS[season][1], '%Y%m%d')
            
            results = []
            for current_date in pd.date_range(start_date, end_date):
                for hour in range(1, 25):
                    date_str = current_date.strftime('%Y%m%d')
                    
                    for idx, row in tqdm(
                        unique_ods.iterrows(),
                        desc=f"Processing {date_str} {hour:02d}h",
                        total=len(unique_ods)
                    ):
                        try:
                            start_point = QgsPointXY(float(row['OLon']), float(row['OLat']))
                            end_point = QgsPointXY(float(row['DLon']), float(row['DLat']))
                            
                            paths = self.find_k_paths_with_constraints(start_point, end_point)
                            
                            logger.info(f"Processed OD pair {row['UniqueID']}, found {len(paths)} valid paths")
                            
                            for path_idx, path in enumerate(paths):
                                results.append({
                                    'UniqueID': row['UniqueID'],
                                    'PathID': path_idx,
                                    'Path': path,
                                    'Season': season,
                                    'Date': date_str,
                                    'Hour': hour
                                })
                                
                        except Exception as e:
                            logger.error(f"Error processing OD pair {row['UniqueID']}: {str(e)}")
                            continue
                    
                    if len(results) >= 100:
                        df = pd.DataFrame(results)
                        df.to_csv(
                            output_path,
                            mode='a',
                            header=not os.path.exists(output_path),
                            index=False,
                            sep='\t'
                        )
                        logger.info(f"Saved {len(results)} paths to {output_path}")
                        results = []
            
            if results:
                df = pd.DataFrame(results)
                df.to_csv(
                    output_path,
                    mode='a',
                    header=not os.path.exists(output_path),
                    index=False,
                    sep='\t'
                )
                logger.info(f"Saved final {len(results)} paths to {output_path}")
                
        except Exception as e:
            logger.error(f"Path generation failed: {str(e)}")
            raise

class TargetMatchingOptimizer:
    """Target matching optimizer - make actual volume as close as possible to target constraints"""
    
    def __init__(self, config: Config):
        self.config = config
        self.bpr_calculator = BPRCalculator(config)
        self.volume_calculator = FreightVolumeCalculator(config)
        
    def calculate_violation_penalty(self, actual_volume: float, target_volume: float) -> float:
        """Calculate deviation penalty from target"""
        deviation = abs(actual_volume - target_volume)
        return deviation ** 2
    
    def identify_high_demand_segments(self, season: str) -> Dict[str, float]:
        """Identify high-demand segments (segments with high target values)"""
        segment_targets = {}
        
        constraints_file = os.path.join(
            self.config.OUTPUT_BASE_PATH,
            f'constraints_4hour_{season}.txt'
        )
        constraints_df = pd.read_csv(constraints_file, sep='\t')
        
        for _, row in constraints_df.iterrows():
            code = row['路线简码']
            target = row['constraint_4hour']
            
            if code not in segment_targets:
                segment_targets[code] = []
            segment_targets[code].append(target)
        
        avg_targets = {
            code: np.mean(targets) 
            for code, targets in segment_targets.items()
        }
        
        logger.info(f"Segment target demand analysis completed, {len(avg_targets)} segments")
        return avg_targets
    
    def generate_demand_aware_initial_solution(self,
                                             vehicles: pd.DataFrame,
                                             available_paths: pd.DataFrame,
                                             season: str) -> pd.DataFrame:
        """Generate demand-aware initial solution"""
        logger.info("Generating demand-aware initial solution...")
        
        segment_targets = self.identify_high_demand_segments(season)
        
        current_allocation = defaultdict(lambda: defaultdict(int))
        target_gaps = self.calculate_target_gaps(segment_targets, current_allocation)
        
        assignment = self.demand_gap_greedy_assignment(
            vehicles, available_paths, target_gaps, season
        )
        
        return assignment
    
    def calculate_target_gaps(self, 
                            segment_targets: Dict[str, float],
                            current_allocation: Dict) -> Dict:
        """Calculate target gaps"""
        target_gaps = {}
        
        for segment, target in segment_targets.items():
            for time_slot in range(6):
                current = current_allocation[time_slot][segment]
                gap = target - current
                target_gaps[(time_slot, segment)] = {
                    'target': target,
                    'current': current,
                    'gap': gap,
                    'urgency': abs(gap)
                }
        
        return target_gaps
    
    def demand_gap_greedy_assignment(self,
                                   vehicles: pd.DataFrame,
                                   available_paths: pd.DataFrame,
                                   target_gaps: Dict,
                                   season: str) -> pd.DataFrame:
        """Greedy assignment based on demand gap (considering PCU weights)"""
        assignment = []
        current_allocation = defaultdict(lambda: defaultdict(int))
        
        for (unique_id, time_slot), group in vehicles.groupby(['UniqueID', 'time_slot']):
            od_paths = available_paths[available_paths['UniqueID'] == unique_id]
            if len(od_paths) == 0:
                continue
            
            total_pcu_weight = group['PCU_weight'].sum()
            
            best_path = self.select_gap_minimizing_path(
                od_paths['Path'].tolist(), 
                time_slot, 
                total_pcu_weight,
                target_gaps,
                current_allocation
            )
            
            for _, vehicle in group.iterrows():
                assignment.append({
                    'VehicleID': len(assignment),
                    'UniqueID': unique_id,
                    'Path': best_path,
                    'Date': vehicle['date'],
                    'time_slot': time_slot,
                    'Index': vehicle['Index'],
                    '核定载质量': vehicle['核定载质量'],
                    'PCU_weight': vehicle['PCU_weight'],
                    'OLon': vehicle['OLon'],
                    'OLat': vehicle['OLat'],
                    'DLon': vehicle['DLon'],
                    'DLat': vehicle['DLat'],
                    'OCityCode': vehicle['OCityCode'],
                    'DCityCode': vehicle['DCityCode']
                })
                
                self.update_allocation_state(best_path, time_slot, current_allocation, vehicle['PCU_weight'])
                
                self.update_target_gaps(best_path, time_slot, target_gaps, current_allocation, vehicle['PCU_weight'])
        
        return pd.DataFrame(assignment)
    
    def select_gap_minimizing_path(self,
                                 candidate_paths: List[str],
                                 time_slot: int,
                                 total_pcu_weight: float,
                                 target_gaps: Dict,
                                 current_allocation: Dict) -> str:
        """Select path that minimizes target gap (considering PCU weights and BPR travel time)"""
        best_path = candidate_paths[0]
        best_score = float('-inf')
        
        for path in candidate_paths:
            path_segments = path.split(',')
            
            improvement_score = 0.0
            
            for segment in path_segments:
                gap_key = (time_slot, segment)
                if gap_key in target_gaps:
                    target = target_gaps[gap_key]['target']
                    current = target_gaps[gap_key]['current']
                    
                    before_deviation = abs(current - target)
                    
                    after_current = current + total_pcu_weight
                    after_deviation = abs(after_current - target)
                    
                    improvement = before_deviation - after_deviation
                    improvement_score += improvement
            
            travel_time = self.calculate_path_travel_time_with_current_volume(
                path, time_slot, current_allocation
            )
            
            time_penalty = travel_time * 0.1
            combined_score = improvement_score - time_penalty
            
            if combined_score > best_score:
                best_score = combined_score
                best_path = path
        
        return best_path
    
    def calculate_path_travel_time_with_current_volume(self,
                                                     path: str,
                                                     time_slot: int,
                                                     current_allocation: Dict) -> float:
        """Calculate path travel time considering current allocated volume (using BPR function)"""
        total_time = 0.0
        path_segments = path.split(',')
        
        for segment_id in path_segments:
            try:
                segment_props = self.get_segment_properties(segment_id)
                if not segment_props:
                    continue
                
                free_flow_time = segment_props.get('time_init', 0)
                capacity = segment_props.get('capacity', 0)
                
                if capacity <= 0:
                    road_type = segment_props.get('fclass', 'primary')
                    capacity = self.bpr_calculator.get_default_capacity(road_type)
                
                current_volume = current_allocation[time_slot].get(segment_id, 0)
                
                actual_time = self.bpr_calculator.calculate_travel_time(
                    free_flow_time, current_volume, capacity
                )
                total_time += actual_time
                
            except Exception as e:
                logger.warning(f"Error calculating travel time for segment {segment_id}: {str(e)}")
                continue
        
        return total_time
    
    def get_segment_properties(self, segment_id: str) -> Dict:
        """Get segment properties"""
        return {
            'time_init': 1.0,
            'capacity': 15000,
            'fclass': 'primary'
        }
    
    def update_allocation_state(self, 
                              path: str, 
                              time_slot: int, 
                              current_allocation: Dict,
                              pcu_weight: float = 1.0):
        """Update current allocation state (considering PCU weights)"""
        path_segments = path.split(',')
        for segment in path_segments:
            current_allocation[time_slot][segment] += pcu_weight
    
    def update_target_gaps(self, 
                         path: str, 
                         time_slot: int, 
                         target_gaps: Dict,
                         current_allocation: Dict,
                         pcu_weight: float = 1.0):
        """Dynamically update target gaps (considering PCU weights)"""
        path_segments = path.split(',')
        for segment in path_segments:
            gap_key = (time_slot, segment)
            if gap_key in target_gaps:
                current = current_allocation[time_slot][segment]
                target = target_gaps[gap_key]['target']
                new_gap = target - current
                
                target_gaps[gap_key]['current'] = current
                target_gaps[gap_key]['gap'] = new_gap
                target_gaps[gap_key]['urgency'] = abs(new_gap)

    def dynamic_rebalancing_optimization(self,
                                       initial_assignment: pd.DataFrame,
                                       available_paths: pd.DataFrame,
                                       season: str,
                                       max_iterations: int = 30) -> pd.DataFrame:
        """Dynamic rebalancing optimization"""
        logger.info("Starting dynamic rebalancing optimization...")
        
        current_assignment = initial_assignment.copy()
        
        for iteration in range(max_iterations):
            worst_deviations = self.identify_worst_deviations(current_assignment, season)
            
            if not worst_deviations:
                logger.info(f"Reached balance at iteration {iteration}")
                break
            
            improved = self.rebalance_worst_segments(
                current_assignment, 
                available_paths, 
                worst_deviations[:5],
                season
            )
            
            if not improved:
                break
            
            total_deviation = self.calculate_total_deviation(current_assignment, season)
            logger.info(f"Iteration {iteration}: Total deviation = {total_deviation:.2f}")
        
        return current_assignment

    def identify_worst_deviations(self, assignment: pd.DataFrame, season: str) -> List[Dict]:
        """Identify segments with worst deviations"""
        deviations = []
        actual_usage = self.calculate_actual_usage(assignment)
        target_constraints = self.get_constraint_limits(season)
        
        for (time_slot, segment), actual in actual_usage.items():
            if (time_slot, segment) in target_constraints:
                target = target_constraints[(time_slot, segment)]
                deviation = abs(actual - target)
                
                deviations.append({
                    'time_slot': time_slot,
                    'segment': segment,
                    'actual': actual,
                    'target': target,
                    'deviation': deviation,
                    'direction': 'over' if actual > target else 'under'
                })
        
        deviations.sort(key=lambda x: x['deviation'], reverse=True)
        return deviations

    def rebalance_worst_segments(self, 
                               assignment: pd.DataFrame,
                               available_paths: pd.DataFrame,
                               worst_deviations: List[Dict],
                               season: str) -> bool:
        """Rebalance segments with worst deviations"""
        improved = False
        
        for deviation in worst_deviations:
            if deviation['direction'] == 'over':
                improved |= self.reduce_segment_load(
                    assignment, available_paths, deviation, season
                )
            else:
                improved |= self.increase_segment_load(
                    assignment, available_paths, deviation, season
                )
        
        return improved

    def calculate_actual_usage(self, assignment: pd.DataFrame) -> Dict:
        """Calculate actual segment usage (considering PCU weights)"""
        usage = defaultdict(int)
        
        for _, row in assignment.iterrows():
            path_segments = row['Path'].split(',')
            time_slot = row['time_slot']
            pcu_weight = row.get('PCU_weight', 1.0)
            
            for segment in path_segments:
                usage[(time_slot, segment)] += pcu_weight
        
        return usage

    def get_constraint_limits(self, season: str) -> Dict:
        """Get constraint limits"""
        constraints = {}
        constraints_file = os.path.join(
            self.config.OUTPUT_BASE_PATH,
            f'constraints_4hour_{season}.txt'
        )
        
        if os.path.exists(constraints_file):
            df = pd.read_csv(constraints_file, sep='\t')
            for _, row in df.iterrows():
                key = (row['time_slot'], row['路线简码'])
                constraints[key] = row['constraint_4hour']
        
        return constraints

    def calculate_total_deviation(self, assignment: pd.DataFrame, season: str) -> float:
        """Calculate total deviation"""
        actual_usage = self.calculate_actual_usage(assignment)
        target_constraints = self.get_constraint_limits(season)
        
        total_deviation = 0
        for (time_slot, segment), actual in actual_usage.items():
            if (time_slot, segment) in target_constraints:
                target = target_constraints[(time_slot, segment)]
                total_deviation += abs(actual - target)
        
        return total_deviation

class PathAssignment:
    """Path assignment class - enhanced with heuristic optimization"""
    def __init__(self, config: Config):
        self.config = config
        self.bpr_calculator = BPRCalculator(config)
        self.volume_calculator = FreightVolumeCalculator(config)
        self.time_slots = [
            (1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24)
        ]
    
    def calculate_vehicle_pcu_weight(self, load_capacity: float) -> float:
        """
        Calculate vehicle PCU weight based on load capacity
        If load_capacity<3, weight=1; <5, weight=1.5; <10, weight=3; >=10, weight=4
        """
        if pd.isna(load_capacity) or load_capacity <= 0:
            return 1.0
        
        if load_capacity < 3:
            return 1.0
        elif load_capacity < 5:
            return 1.5
        elif load_capacity < 10:
            return 3.0
        else:
            return 4.0
    
    def add_vehicle_weights(self, vehicles: pd.DataFrame) -> pd.DataFrame:
        """Add PCU weights to vehicle data"""
        vehicles = vehicles.copy()
        vehicles['PCU_weight'] = vehicles['核定载质量'].apply(self.calculate_vehicle_pcu_weight)
        return vehicles
    
    def distribute_vehicles_to_time_slots(self,
                                        vehicles: pd.DataFrame,
                                        season: str) -> pd.DataFrame:
        """Distribute vehicles to 4-hour time slots"""
        logger.info("Distributing vehicles to 4-hour time slots...")
        
        constraints_file = os.path.join(
            self.config.OUTPUT_BASE_PATH,
            f'constraints_4hour_{season}.txt'
        )
        constraints_df = pd.read_csv(constraints_file, sep='\t')
        
        slot_ratios = {}
        for date in vehicles['date'].unique():
            day_constraints = constraints_df[constraints_df['date'] == date]
            total_volume = day_constraints['constraint_4hour'].sum()
            if total_volume > 0:
                slot_ratios[date] = (
                    day_constraints.groupby('time_slot')['constraint_4hour'].sum() / total_volume
                )
        
        vehicles['time_slot'] = None
        
        for date in tqdm(vehicles['date'].unique(), desc="Assigning time slots"):
            date_vehicles = vehicles[vehicles['date'] == date]
            if date not in slot_ratios:
                continue
            
            ratios = slot_ratios[date]
            num_vehicles = len(date_vehicles)
            
            if len(ratios) > 0:
                ratios = ratios / ratios.sum()
                slot_distribution = np.random.multinomial(
                    num_vehicles,
                    ratios.values
                )
                
                slot_assignments = []
                for slot_idx, count in enumerate(slot_distribution):
                    slot_assignments.extend([ratios.index[slot_idx]] * count)
                
                np.random.shuffle(slot_assignments)
                vehicles.loc[date_vehicles.index, 'time_slot'] = slot_assignments
        
        return vehicles

    def calculate_weighted_volume_by_code(self,
                                        assigned_paths: pd.DataFrame,
                                        network_analysis: 'NetworkAnalysis') -> pd.DataFrame:
        """Calculate weighted average volume by Code (considering PCU weights)"""
        volume_by_code = {}
        
        for _, row in assigned_paths.iterrows():
            path_segments = row['Path'].split(',')
            pcu_weight = row.get('PCU_weight', 1.0)
            
            for segment_id in path_segments:
                for feature in network_analysis.network_layer.getFeatures():
                    if str(feature['id']) == segment_id:
                        code = feature['Code']
                        length = feature['length']
                        
                        if pd.notna(code) and code != '':
                            key = (row['Date'], row['time_slot'], code)
                            if key not in volume_by_code:
                                volume_by_code[key] = {'total_volume': 0, 'total_length': 0}
                            
                            volume_by_code[key]['total_volume'] += pcu_weight * length
                            volume_by_code[key]['total_length'] += length
                        break
        
        result_list = []
        for (date, time_slot, code), data in volume_by_code.items():
            if data['total_length'] > 0:
                weighted_avg_volume = data['total_volume'] / data['total_length']
                result_list.append({
                    'date': date,
                    'time_slot': time_slot,
                    'code': code,
                    'weighted_volume': weighted_avg_volume
                })
        
        return pd.DataFrame(result_list)

    def calculate_assignment_error(self,
                                 assigned_paths: pd.DataFrame,
                                 network_analysis: 'NetworkAnalysis',
                                 season: str) -> float:
        """Calculate assignment error"""
        actual_volumes = self.calculate_weighted_volume_by_code(assigned_paths, network_analysis)
        
        total_error = 0
        error_count = 0
        
        for date in assigned_paths['date'].unique():
            for time_slot in range(6):
                constraints = self.volume_calculator.get_volume_constraints_4hour(
                    season, date, time_slot
                )
                
                if constraints is None or len(constraints) == 0:
                    continue
                
                for _, constraint_row in constraints.iterrows():
                    code = constraint_row['路线简码']
                    target_volume = constraint_row['constraint_4hour']
                    
                    actual_row = actual_volumes[
                        (actual_volumes['date'] == date) &
                        (actual_volumes['time_slot'] == time_slot) &
                        (actual_volumes['code'] == code)
                    ]
                    
                    actual_volume = actual_row['weighted_volume'].iloc[0] if len(actual_row) > 0 else 0
                    
                    error = abs(actual_volume - target_volume)
                    total_error += error
                    error_count += 1
        
        return total_error / error_count if error_count > 0 else float('inf')

    def heuristic_optimization(self,
                             vehicles: pd.DataFrame,
                             available_paths: pd.DataFrame,
                             network_analysis: 'NetworkAnalysis',
                             season: str,
                             max_iterations: int = 3000) -> pd.DataFrame:
        """Heuristic optimization algorithm - improved version"""
        logger.info("Starting heuristic optimization...")
        
        best_assignment = None
        best_error = float('inf')
        no_improvement_count = 0
        
        for iteration in tqdm(range(max_iterations), desc="Heuristic optimization"):
            current_assignment = []
            
            for _, vehicle in vehicles.iterrows():
                vehicle_paths = available_paths[
                    available_paths['UniqueID'] == vehicle['UniqueID']
                ]
                
                if len(vehicle_paths) == 0:
                    continue
                
                selected_path = vehicle_paths.sample(1).iloc[0]
                
                current_assignment.append({
                    'VehicleID': len(current_assignment),
                    'Date': vehicle['date'],
                    'time_slot': vehicle['time_slot'],
                    'Index': vehicle['Index'],
                    '核定载质量': vehicle['核定载质量'],
                    'PCU_weight': vehicle.get('PCU_weight', 1.0),
                    'OLon': vehicle['OLon'],
                    'OLat': vehicle['OLat'],
                    'DLon': vehicle['DLon'],
                    'DLat': vehicle['DLat'],
                    'OCityCode': vehicle['OCityCode'],
                    'DCityCode': vehicle['DCityCode'],
                    'UniqueID': vehicle['UniqueID'],
                    'Path': selected_path['Path']
                })
            
            if not current_assignment:
                continue
            
            current_df = pd.DataFrame(current_assignment)
            current_error = self.calculate_assignment_error(current_df, network_analysis, season)
            
            if current_error < best_error:
                best_error = current_error
                best_assignment = current_df.copy()
                no_improvement_count = 0
                logger.info(f"Iteration {iteration}: Found better assignment, error = {best_error:.4f}")
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= 20:
                logger.info(f"No improvement for {no_improvement_count} consecutive iterations, stopping early")
                break
        
        logger.info(f"Optimization completed, final error: {best_error:.4f}")
        return best_assignment if best_assignment is not None else pd.DataFrame()

    def assign_paths(self, 
                    od_data: pd.DataFrame,
                    available_paths: pd.DataFrame,
                    network_analysis: 'NetworkAnalysis',
                    season: str,
                    use_target_matching: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Assign paths to vehicles - supports multiple optimization strategies"""
        optimization_type = "Target matching optimization" if use_target_matching else "Heuristic optimization"
        logger.info(f"Starting path assignment for {season} season ({optimization_type})...")
        
        try:
            vehicles = self.distribute_vehicles_to_days(od_data.copy(), season)
            
            vehicles = self.distribute_vehicles_to_time_slots(vehicles, season)
            
            vehicles = self.add_vehicle_weights(vehicles)
            
            if use_target_matching:
                target_optimizer = TargetMatchingOptimizer(self.config)
                
                initial_assignment = target_optimizer.generate_demand_aware_initial_solution(
                    vehicles, available_paths, season
                )
                
                optimized_assignment = target_optimizer.dynamic_rebalancing_optimization(
                    initial_assignment, available_paths, season, self.config.TARGET_MATCHING_MAX_ITERATIONS
                )
                
                output_suffix = 'target_matching'
            else:
                optimized_assignment = self.heuristic_optimization(
                    vehicles, available_paths, network_analysis, season, self.config.HEURISTIC_MAX_ITERATIONS
                )
                
                output_suffix = 'heuristic'
            
            violations_df = pd.DataFrame()
            
            output_path = os.path.join(
                self.config.OUTPUT_BASE_PATH,
                f'optimized_assignment_{season}.txt'
            )
            optimized_assignment.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
            
            return optimized_assignment, violations_df
            
        except Exception as e:
            logger.error(f"Path assignment failed: {str(e)}")
            raise
    
    def distribute_vehicles_to_days(self, vehicles: pd.DataFrame, season: str) -> pd.DataFrame:
        """Distribute vehicles to days"""
        constraints_file = os.path.join(
            self.config.OUTPUT_BASE_PATH,
            f'constraints_4hour_{season}.txt'
        )
        constraints_df = pd.read_csv(constraints_file, sep='\t')
        
        daily_volumes = constraints_df.groupby('date')['constraint_4hour'].sum()
        total_volume = daily_volumes.sum()
        daily_ratios = daily_volumes / total_volume
        
        vehicles['date'] = None
        
        for idx, row in tqdm(vehicles.iterrows(), desc="Assigning dates", total=len(vehicles)):
            num_copies = int(row['Coefficient'])
            if num_copies <= 0:
                continue
            
            day_distribution = np.random.multinomial(
                num_copies,
                daily_ratios.values
            )
            
            new_rows = []
            for day_idx, count in enumerate(day_distribution):
                if count > 0:
                    for _ in range(count):
                        new_row = row.copy()
                        new_row['date'] = daily_ratios.index[day_idx]
                        new_rows.append(new_row)
            
            if new_rows:
                vehicles = pd.concat([
                    vehicles,
                    pd.DataFrame(new_rows)
                ], ignore_index=True)
        
        vehicles = vehicles.dropna(subset=['date'])
        
        return vehicles
    
    def check_volume_constraints(self,
                               assigned_paths: pd.DataFrame,
                               season: str) -> pd.DataFrame:
        """Check and record volume constraint violations"""
        violations = []
        
        for date in assigned_paths['date'].unique():
            for hour in range(1, 25):
                constraints = self.volume_calculator.get_volume_constraints(
                    season, date, hour
                )
                if constraints is None:
                    continue
                
                time_slice = assigned_paths[
                    (assigned_paths['date'] == date) &
                    (assigned_paths['hour'] == hour)
                ]
                
                for _, constraint_row in constraints.iterrows():
                    code = constraint_row['路线简码']
                    if pd.isna(code) or code == '':
                        continue
                    
                    path_count = sum(
                        1 for _, path_row in time_slice.iterrows()
                        if code in path_row['Path']
                    )
                    
                    if path_count > constraint_row['constraint']:
                        violations.append({
                            'date': date,
                            'hour': hour,
                            'code': code,
                            'assigned_volume': path_count,
                            'constraint': constraint_row['constraint'],
                            'violation_ratio': path_count / constraint_row['constraint']
                        })
        
        return pd.DataFrame(violations)

class TrajectoryVisualizer:
    """Trajectory visualization class"""
    def __init__(self, config: Config):
        self.config = config
        
    def create_volume_map(self,
                         network_gdf: gpd.GeoDataFrame,
                         trajectory_df: pd.DataFrame,
                         season: str,
                         output_path: str):
        """Create volume map"""
        import matplotlib.pyplot as plt
        
        logger.info(f"Creating volume map for {season} season...")
        
        try:
            edge_counts = {}
            for _, row in tqdm(trajectory_df.iterrows(), desc="Calculating segment usage frequency"):
                path_edges = row['Path'].split(',')
                pcu_weight = row.get('PCU_weight', 1.0)
                
                for i in range(len(path_edges) - 1):
                    edge_id = f"{path_edges[i]}-{path_edges[i+1]}"
                    edge_counts[edge_id] = edge_counts.get(edge_id, 0) + pcu_weight
            
            network_gdf['volume'] = network_gdf.apply(
                lambda x: edge_counts.get(f"{x['id']}", 0),
                axis=1
            )
            
            fig, ax = plt.subplots(figsize=(15, 15))
            
            network_gdf.plot(
                ax=ax,
                color='gray',
                alpha=0.5,
                linewidth=0.5
            )
            
            max_volume = network_gdf['volume'].max()
            if max_volume > 0:
                network_gdf[network_gdf['volume'] > 0].plot(
                    ax=ax,
                    color='blue',
                    linewidth=network_gdf['volume'].apply(
                        lambda x: 0.5 + 4.5 * (x / max_volume)
                    ),
                    alpha=0.6
                )
            
            ax.set_title(f'{season} Season Traffic Volume')
            ax.set_axis_off()
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=1, label='Low Volume'),
                Line2D([0], [0], color='blue', linewidth=3, label='Medium Volume'),
                Line2D([0], [0], color='blue', linewidth=5, label='High Volume')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Volume map saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create volume map: {str(e)}")
            raise

def process_single_season(season: str, config: Config) -> None:
    """Process single season data"""
    try:
        process_logger = logging.getLogger(f'{season}_process')
        fh = logging.FileHandler(f'traffic_assignment_{season}.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        process_logger.addHandler(fh)
        process_logger.setLevel(logging.INFO)
        
        process_logger.info(f"Starting {season} season processing")
        
        preprocessor = DataPreprocessor(config)
        corrector = CoefficientCorrector(config)
        volume_calculator = FreightVolumeCalculator(config)
        network_analysis = NetworkAnalysis(config)
        path_assignment = PathAssignment(config)
        visualizer = TrajectoryVisualizer(config)
        
        full_od_data, unique_ods = preprocessor.preprocess_od_data()
        process_logger.info(f"Data preprocessing completed, {len(full_od_data)} records, {len(unique_ods)} unique OD pairs")
        
        seasonal_coefficients = corrector.correct_coefficients(full_od_data)
        season_od = seasonal_coefficients[season]
        process_logger.info("Coefficient correction completed")
        
        volume_data = volume_calculator.calculate_total_volume()
        process_logger.info("Volume calculation completed")
        
        network_analysis.initialize_qgis()
        network_analysis.load_network()
        process_logger.info("Network loading completed")
        
        paths_file = os.path.join(config.OUTPUT_BASE_PATH, f'paths_{season}.txt')
        if not os.path.exists(paths_file):
            process_logger.info("Starting path generation...")
            network_analysis.generate_paths_sequential(unique_ods, season, process_logger)
        
        available_paths = pd.read_csv(paths_file, sep='\t')
        process_logger.info(f"Read {len(available_paths)} path records")
        
        use_target_matching = config.USE_TARGET_MATCHING
        process_logger.info(f"Using {'target matching optimization' if use_target_matching else 'traditional heuristic optimization'} strategy")
        
        trajectory_df, violations_df = path_assignment.assign_paths(
            season_od,
            available_paths,
            network_analysis,
            season,
            use_target_matching
        )
        process_logger.info("Path assignment completed")
        
        network_gdf = gpd.read_file(config.ROAD_NETWORK_PATH)
        map_output_path = os.path.join(
            config.OUTPUT_BASE_PATH,
            f'TrafficVolume_{season}.png'
        )
        visualizer.create_volume_map(
            network_gdf,
            trajectory_df,
            season,
            map_output_path
        )
        process_logger.info("Visualization completed")
        
        if network_analysis.qgs:
            network_analysis.qgs.exitQgis()
        
        process_logger.info(f"{season} season processing completed")
        
    except Exception as e:
        process_logger.error(f"{season} season processing failed: {str(e)}")
        raise

def main_parallel():
    """Main function - parallel version"""
    try:
        config = Config()
        logging.info("Configuration initialization completed")
        
        os.makedirs(config.OUTPUT_BASE_PATH, exist_ok=True)
        
        num_processes = min(len(config.SEASONS), mp.cpu_count())
        with mp.Pool(num_processes) as pool:
            tasks = [(season, config) for season in config.SEASONS]
            pool.starmap(process_single_season, tasks)
        
        logging.info("All seasons processing completed")
        
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")
        raise
    finally:
        logging.info("Program execution completed")

if __name__ == "__main__":
    main_parallel()