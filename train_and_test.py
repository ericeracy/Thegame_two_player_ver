import sys
import os
import numpy as np
import random

import torch

from turn_based_and_test import Agent
import time
import logging
from pathlib import Path
import argparse

def main():
  parser=argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--seed', type=int, default=101, help='set seed')
  parser.add_argument('--save', action='store_true', help='save checkpoint and load log')
  parser.add_argument('--hidden_dim', type=int, default=128, help='the number of hidden units for the first layer of MLP')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
  parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
  parser.add_argument('--update_target', type=int, default=40000, help='update target steps')
  parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
  parser.add_argument('--game_mode', default='absolute-quantifier', help='game mode')
  parser.add_argument('--turn_game', type=int, default=500, help='the number of games per episode')
  parser.add_argument('--ep_min', type=float, default=0.02, help='ep minimum value')
  parser.add_argument('--max_step', type=int, default=200, help='max number of steps per game')
  parser.add_argument('--running_p', type=int, default=1000, help='steps per output during running')
  parser.add_argument('--episodes', type=int, default=1000, help='the number of episodes')
  args = parser.parse_args()

  np.random.seed(args.seed)
  random.seed(args.seed)
  torch.manual_seed(args.seed)


  timestr = time.strftime("%Y%m%d-%H%M%S")

  if args.game_mode=='no-commu':
    game_mode_abrv = 'miao'
  elif args.game_mode=='absolute-quantifier':
    game_mode_abrv = 'aq'
  elif args.game_mode=='fuzzy-quantifier':
    game_mode_abrv = 'fq'

  data_path="./data"
  Path(data_path).mkdir(parents=True, exist_ok=True)
  experiment_dir = '%s/%s_turn_based_%s'%(data_path, game_mode_abrv, timestr)
  Path(experiment_dir).mkdir(parents=True, exist_ok=True)

  log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  log_file_path = os.path.join(experiment_dir, "hp_monitor.log")
  logging.basicConfig(filename=log_file_path, format=log_format )
  # file_handler = logging.FileHandler(log_file_path)
  # formatter = logging.Formatter(log_format)
  # file_handler.setFormatter(formatter)
  # logger.addHandler(file_handler)
  logger.debug("Harmless debug Message")

  a = Agent(saving_path_time=timestr, 
            experiment_dir=experiment_dir,
            logger=logger,
            seed=args.seed, 
            save=args.save, 
            hidden_dim=args.hidden_dim, 
            lr=args.lr, 
            batch_size=args.batch_size, 
            update_target=args.update_target, 
            gamma=args.gamma, 
            game_mode=args.game_mode, 
            turn_game=args.turn_game,
            ep_min=args.ep_min, 
            running_p=args.running_p,
            episodes=args.episodes) # turn_step=250000, running_p=1000, running_save=10000, seed=101
  a.create_env()
  a.create_models()
  start_time = time.time()
  logging.info("training starts!")
  a.train()
  # logging.info(a.game_win)
  end_time = time.time()
  logging.info("training ends")
  logging.info("training time: {:.3f}s".format(end_time-start_time))

if __name__ == '__main__':
  main()
