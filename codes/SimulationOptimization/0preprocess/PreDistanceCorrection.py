"""
Author: Shiqi (Anya) WANG
Date: 2025/6/4
Description: Distance correction coefficient calculation module (independent operation)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import logging
from scipy.spatial.distance import cdist
import yaml
import openpyxl


class CityDistanceCorrectionCalculator:
    
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.city_shp_path = "D:/ChinaStation/NewBaidu/City.shp"
        self.road_network_path = "D:/Road/OSMOutput/ChinaHighway.gpkg"
        self.traffic_volume_path = "D:/Road/RoadProcess/Output/002VolumeDayUpdate"
        self.output_path = "D:/ChinaEFVSimulation/ChinaEFVParameter/CityDistanceCorrection.pkl"
        
        self.seasons_dates = {
            'Winter': list(range(20190114, 20190121)),
            'Spring': list(range(20190418, 20190425)),
            'Summer': list(range(20190716, 20190723)),
            'Autumn': list(range(20191106, 20191113))
        }
        
        self.all_dates = []
        for season, dates in self.seasons_dates.items():
            self.all_dates.extend(dates)
        
        self.logger.info(f"Processing 28 days of data: {self.all_dates}")
        
        self.city_shapes = None
        self.road_network = None
        self.traffic_data = None
        
    def run(self):
        try:
            self.logger.info("Starting distance correction coefficient calculation...")
            
            self._load_data()
            
            city_factors = self._calculate_city_correction_factors()
            
            self._save_correction_factors(city_factors)
            
            self.logger.info("Distance correction coefficient calculation completed")
            return city_factors
            
        except Exception as e:
            self.logger.error(f"Distance correction coefficient calculation failed: {e}")
            raise
    
    def _load_data(self):
        self.logger.info("Loading data...")
        
        self.logger.info(f"Loading city boundary data: {self.city_shp_path}")
        self.city_shapes = gpd.read_file(self.city_shp_path, encoding='gbk')
        
        if '市代码' in self.city_shapes.columns:
            self.city_shapes['CityCode'] = self.city_shapes['市代码']
        elif 'CityCode' not in self.city_shapes.columns:
            raise ValueError("Missing '市代码' or 'CityCode' field in city shapefile")
        
        self.logger.info(f"City boundary data loaded: {len(self.city_shapes)} cities")
        
        self.logger.info(f"Loading road network data: {self.road_network_path}")
        self.road_network = gpd.read_file(self.road_network_path)
        self.road_network['ID'] = self.road_network['ID'].astype(str)
        self.logger.info(f"Road network data loaded: {len(self.road_network)} roads")
        
        self.logger.info(f"Loading traffic data: {self.traffic_volume_path}")
        self._load_traffic_data()
        
        self.logger.info("Data loading complete")
    
    def _load_traffic_data(self):
        self.traffic_data = {}
        volume_dir = Path(self.traffic_volume_path)
        
        loaded_dates = []
        missing_dates = []
        
        for date in self.all_dates:
            file_path = volume_dir / f"{date}_update.xls"
            
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    
                    required_fields = ['路线简码', '小时', '拥挤度']
                    if all(field in df.columns for field in required_fields):
                        self.traffic_data[date] = df
                        loaded_dates.append(date)
                    else:
                        self.logger.warning(f"Missing required fields in {file_path}")
                        missing_dates.append(date)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
                    missing_dates.append(date)
            else:
                self.logger.warning(f"Traffic data file not found: {file_path}")
                missing_dates.append(date)
        
        self.logger.info(f"Traffic data loaded: {len(loaded_dates)} days, missing: {len(missing_dates)} days")
        if missing_dates:
            self.logger.warning(f"Missing dates: {missing_dates}")
    
    def _calculate_city_correction_factors(self):
        self.logger.info("Calculating city distance correction factors...")
        
        city_factors = {}
        
        for _, city in self.city_shapes.iterrows():
            city_code = city['CityCode']
            city_name = city.get('城市名称', city.get('Name', f"City_{city_code}"))
            
            self.logger.info(f"Processing city: {city_name} (Code: {city_code})")
            
            try:
                base_factor = self._calculate_city_base_factor(city_code, city.geometry)
                
                time_factors = self._calculate_city_time_factors(city_code)
                
                city_factors[city_code] = {
                    'city_name': city_name,
                    'base_factor': base_factor,
                    'time_factors': time_factors
                }
                
                self.logger.info(f"City {city_name} calculation complete, base factor: {base_factor:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to calculate correction factor for city {city_name}: {e}")
                city_factors[city_code] = {
                    'city_name': city_name,
                    'base_factor': 1.3,
                    'time_factors': {date: [1.0] * 24 for date in self.all_dates}
                }
        
        self.logger.info(f"All city distance correction factors calculated: {len(city_factors)} cities")
        return city_factors
    
    def _calculate_city_base_factor(self, city_code, city_geometry):
        city_roads = self.road_network[
            self.road_network['CityCode'] == city_code
        ]
        
        if len(city_roads) == 0:
            self.logger.warning(f"No road data for city {city_code}, using default coefficient")
            return 1.3
        
        factors = []
        
        density_factor = self._calculate_road_density_factor(city_roads, city_geometry)
        factors.append(density_factor)
        
        complexity_factor = self._calculate_road_complexity_factor(city_roads)
        factors.append(complexity_factor)
        
        connectivity_factor = self._calculate_connectivity_factor(city_roads)
        factors.append(connectivity_factor)
        
        tortuosity_factor = self._calculate_tortuosity_factor(city_roads)
        factors.append(tortuosity_factor)
        
        weights = [0.3, 0.25, 0.25, 0.2]  
        base_factor = sum(f * w for f, w in zip(factors, weights))
        
        base_factor = max(1.1, min(2.5, base_factor))
        
        return base_factor
    
    def _calculate_road_density_factor(self, city_roads, city_geometry):
        try:
            city_area_km2 = city_geometry.area / 1e6
            
            if city_area_km2 <= 0:
                return 1.2
            
            total_length_km = city_roads['Long_meter'].sum() / 1000
            
            road_density = total_length_km / city_area_km2
            
            if road_density < 1.0:
                return 1.15
            elif road_density < 2.0:
                return 1.25
            elif road_density < 4.0:
                return 1.4
            else:
                return 1.6
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate road density: {e}")
            return 1.2
    
    def _calculate_road_complexity_factor(self, city_roads):
        try:
            road_types = city_roads['fclass'].value_counts(normalize=True)
            
            type_weights = {
                'trunk': 0.1,
                'primary': 0.2,
                'secondary': 0.3,
                'tertiary': 0.4,
                'residential': 0.5,
                'service': 0.6
            }
            
            complexity = 0.0
            for road_type, proportion in road_types.items():
                weight = type_weights.get(road_type, 0.4)
                complexity += proportion * weight
            
            return 1.1 + complexity * 0.8
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate road complexity: {e}")
            return 1.25
    
    def _calculate_connectivity_factor(self, city_roads):
        try:
            road_count = len(city_roads)
            avg_length = city_roads['Long_meter'].mean()
            
            if avg_length <= 0:
                return 1.2
            
            connectivity_ratio = road_count / (avg_length / 1000)
            
            if connectivity_ratio < 0.5:
                return 1.1
            elif connectivity_ratio < 1.0:
                return 1.2
            elif connectivity_ratio < 2.0:
                return 1.35
            else:
                return 1.5
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate connectivity: {e}")
            return 1.2

    def _calculate_tortuosity_factor(self, city_roads):
        try:
            tortuosity_values = []
            
            for _, road in city_roads.iterrows():
                if road.geometry is not None:
                    try:
                        if hasattr(road.geometry, 'geoms'):
                            coords = list(road.geometry.geoms[0].coords)
                        elif hasattr(road.geometry, 'coords'):
                            coords = list(road.geometry.coords)
                        else:
                            continue
                        
                        if len(coords) >= 2:
                            start_point = coords[0]
                            end_point = coords[-1]
                            straight_distance = np.sqrt(
                                (end_point[0] - start_point[0])**2 + 
                                (end_point[1] - start_point[1])**2
                            )
                            
                            if straight_distance > 0:
                                actual_length = road['Long_meter']
                                tortuosity = actual_length / straight_distance
                                tortuosity_values.append(tortuosity)
                                
                    except Exception as e:
                        continue
            
            if tortuosity_values:
                avg_tortuosity = np.mean(tortuosity_values)
                return min(1.1 + (avg_tortuosity - 1.0) * 0.3, 1.8)
            else:
                return 1.2
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate tortuosity: {e}")
            return 1.2
    
    def _calculate_city_time_factors(self, city_code):
        time_factors = {}
        
        city_roads = self.road_network[
            self.road_network['CityCode'] == city_code
        ]
        road_codes = city_roads['Code'].unique()
        
        if len(road_codes) == 0:
            for date in self.all_dates:
                time_factors[date] = [1.0] * 24
            return time_factors
        
        for date in self.all_dates:
            daily_factors = []
            
            for hour in range(24):
                hour_congestions = []
                
                for road_code in road_codes:
                    congestion = self._get_road_congestion(road_code, date, hour)
                    if congestion is not None:
                        hour_congestions.append(congestion)
                
                if hour_congestions:
                    avg_congestion = np.mean(hour_congestions)
                    time_factor = 1.0 + avg_congestion * 0.6
                else:
                    time_factor = 1.0
                
                daily_factors.append(round(time_factor, 3))
            
            time_factors[date] = daily_factors
        
        return time_factors
    
    def _get_road_congestion(self, road_code, date, hour):
        if date not in self.traffic_data:
            return None
        
        df = self.traffic_data[date]
        matching_records = df[
            (df['路线简码'] == road_code) & 
            (df['小时'] == hour + 1)         ]
        
        if not matching_records.empty:
            return matching_records['拥挤度'].iloc[0]
        else:
            return None
    
    def _save_correction_factors(self, city_factors):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            metadata = {
                'version': '1.0',
                'description': 'Urban distance correction coefficient - based on road network morphology and congestion',
                'dates_covered': self.all_dates,
                'seasons': self.seasons_dates,
                'total_cities': len(city_factors),
                'data_structure': {
                    'city_code': {
                        'city_name': 'str',
                        'base_factor': 'float',
                        'time_factors': {
                            'date (YYYYMMDD)': '[24 hourly factors]'
                        }
                    }
                }
            }
            
            final_data = {
                'metadata': metadata,
                'city_factors': city_factors
            }
            
            with open(self.output_path, 'wb') as f:
                pickle.dump(final_data, f)
            
            self.logger.info(f"Distance correction coefficient saved to: {self.output_path}")
            
            json_path = self.output_path.replace('.pkl', '_sample.json')
            import json
            
            sample_data = {
                'metadata': metadata,
                'sample_cities': {}
            }
            
            sample_count = 0
            for city_code, city_data in city_factors.items():
                if sample_count >= 3:
                    break
                
                sample_time_factors = {}
                date_count = 0
                for date, factors in city_data['time_factors'].items():
                    if date_count >= 2:
                        break
                    sample_time_factors[date] = factors
                    date_count += 1
                
                sample_data['sample_cities'][city_code] = {
                    'city_name': city_data['city_name'],
                    'base_factor': city_data['base_factor'],
                    'sample_time_factors': sample_time_factors
                }
                sample_count += 1
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Sample data saved to: {json_path}")
            
            total_time_points = len(self.all_dates) * 24
            self.logger.info(f"Generated time correction factors: {len(city_factors)} cities × {total_time_points} time points = {len(city_factors) * total_time_points}")
            
        except Exception as e:
            self.logger.error(f"Failed to save distance correction coefficient: {e}")
            raise


def main():
    calculator = CityDistanceCorrectionCalculator()
    city_factors = calculator.run()
    
    print("\n=== Sample results ===")
    for city_code, factors in list(city_factors.items())[:2]:
        print(f"City: {factors['city_name']} (Code: {city_code})")
        print(f"  Base factor: {factors['base_factor']:.3f}")
        
        sample_dates = list(factors['time_factors'].keys())[:2]
        for date in sample_dates:
            print(f"  {date} Time factors (first 6 hours): {factors['time_factors'][date][:6]}")
        print()


if __name__ == "__main__":
    main()