"""
Main program entry point
TheMainlandofChinaHighwayAlternativeFuelFreightVehicleSimulationandOptimizationSystem
"""

import os
import time
import logging
import argparse
import yaml
import multiprocessing as mp
from pathlib import Path

from data_processor.loader import DataLoader
from data_processor.preprocessor import DataPreprocessor
from simulation.scheduler import SimulationScheduler
from optimization.solver import MultiObjectiveOptimizer
from utils.logger import setup_logger

import pandas as pd
pd.options.mode.chained_assignment = None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='The Mainland of China Highway Alternative Fuel Freight Vehicle Simulation and Optimization System')
    parser.add_argument('--config', type=str, default='config/parameters.yaml',
                        help='Configuration file path')
    parser.add_argument('--season', type=str, default='Winter',
                        choices=['Winter', 'Spring', 'Summer', 'Autumn'],
                        help='Simulation season selection')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel processing threads (0: use all available CPU cores)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel simulated annealing algorithm')
    parser.add_argument('--sample-ratio', type=float, default=None,
                        help='Data sampling ratio (0-1), for quick testing')
    parser.add_argument('--max-trajectories', type=int, default=6000000,
                        help='Maximum number of trajectories to process, for memory usage limitation')
    return parser.parse_args()

def main():
    """Main function"""
    start_time = time.time()
    
    args = parse_args()
    

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    

    config['optimization'] = config.get('optimization', {})
    config['optimization']['early_stop_all_infeasible'] = False 
    config['optimization']['early_stop_threshold'] = 0.6  
    

    output_dir = Path(config['path_config']['output_directory'])
    cache_dir = output_dir / 'cache'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    

    log_file = output_dir / f'simulation_{args.season}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger('main', log_file, logging.DEBUG if args.debug else logging.INFO)
    

    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    logger.info(f"Using {n_workers} workers")
    

    sample_ratio = args.sample_ratio or config['data_processing'].get('sample_ratio', 1.0)
    max_trajectories = args.max_trajectories
    

    config['simulation'] = config.get('simulation', {})
    config['simulation']['season'] = args.season
    if max_trajectories:
        config['simulation']['max_trajectories'] = max_trajectories
        logger.info(f"Max trajectories: {max_trajectories}")
    else:
        if 'max_trajectories' in config['simulation']:
            del config['simulation']['max_trajectories']
            logger.info("Processing all trajectories")
    
    if args.parallel:
        logger.info("Using parallel algorithm")
        config['optimization']['simulated_annealing']['algorithm'] = 'parallel_simulated_annealing'
    
    logger.info(f"Starting: season={args.season}, debug={args.debug}, parallel={args.parallel}")
    
    try:
        logger.info("Loading data...")
        data_loader = DataLoader(config, logger, season=args.season, sample_ratio=sample_ratio)
        

        data = data_loader.load_all_data()
        logger.info(f"Data loaded in {time.time() - start_time:.2f}s")
        
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(config, logger, n_workers)
        processed_data = preprocessor.process(data)
        logger.info(f"Data preprocessed in {time.time() - start_time:.2f}s")
        

        del data
        import gc
        gc.collect()
        
        logger.info("Starting scheduler...")
        scheduler = SimulationScheduler(config, logger, processed_data, cache_dir, n_workers)
        
        logger.info("Starting optimization...")
        optimizer = MultiObjectiveOptimizer(
            config=config,
            logger=logger,
            simulation_scheduler=scheduler,
            n_workers=n_workers
        )
        

        optimization_results = optimizer.optimize()
        
        logger.info("Exporting results...")
        optimizer.export_results()
        
        logger.info(f"Completed in {time.time() - start_time:.2f}s")
    
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise
    

if __name__ == "__main__":
    main()