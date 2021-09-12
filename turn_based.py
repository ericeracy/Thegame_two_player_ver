import sys

import argparse
# seed = 101 # 101, 11, 22

import numpy as np
# np.random.seed(seed)

import random
# random.seed(seed)

import torch
# torch.manual_seed(seed)
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform

from collections import deque, namedtuple
from collections import defaultdict

import csv
import time
import logging
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from replaybuffer import ReplayMemoryBuffer 
from dqnmlpmodel import DQNMLPnn
from thegame_game import thegame
from pathlib import Path



class Agent:
  def __init__(self, saving_path_time, experiment_dir, logger, game_mode='no-commu', 
               ep_min=0.02, seed=101, lr=0.0001, gamma=0.99, 
               turn_game=200, update_freq=16, update_target=10000, save=False, 
               batch_size=64, hidden_dim=128, max_step=200, running_p=1000, episodes=50):
    self.ep_A = 1.0
    self.ep_B = 1.0
    self.EP_MAX = 1.0
    self.EP_MIN = ep_min # default 0.02, # 0.1
    self.EP_greedy = 1000000
    self.EP_DECAY = (self.EP_MAX-self.EP_MIN)/self.EP_greedy
    self.seed = seed
    self.replay_buffer_size = 1000000
    self.lr = lr
    self.gamma = gamma # 0.95
    self.learning_start = 50000
    self.update_target=update_target # 1000, 40000
    self.turn_game=turn_game
    self.update_freq = update_freq # 1
    self.tau = 1e-3
    self.save = save 
    self.running_p = running_p
    self.saving_path_time = saving_path_time
    self.experiment_dir = experiment_dir
    self.episodes = episodes
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim # 128
    self.num_actions = 11
    self.monitor_n = 500

    self.ILLEGAL_ACTION_LOGITS_PENALTY=-1e6
    self.nantransitions_A = deque(maxlen=2)
    self.nantransitions_B = deque(maxlen=2)

    self.game_mode=game_mode
    self.game_win=0
    self.max_step=max_step
    self.set_history=False
    self.counttp_counter=0

    self.logger=logger

    local_para = locals()
    for i in local_para.keys():
      logging.info("{}: {}".format(i, local_para[i])) # local_para seems only print what are input as variables

    if self.set_history==False: 
      if self.game_mode=='no-commu':
        self.input_size = 8
        self.save_file_prefix='miao'
      else:
        self.input_size = 12 # cnn 是channel # 8
        if self.game_mode=='absolute-quantifier':
          self.save_file_prefix='aq'
        elif self.game_mode=='fuzzy-quantifier':
          self.save_file_prefix='fq'
    else:
      if self.game_mode=='no-commu':
        self.input_size = 58
        self.save_file_prefix='miao_history'
      else:
        self.input_size = 62
        if self.game_mode=='absolute-quantifier':
          self.save_file_prefix='aq_history'
        elif self.game_mode=='fuzzy-quantifier':
          self.save_file_prefix='fq_history'

    if self.save==True:
      experiment_path=experiment_dir
      # Path(data_path).mkdir(parents=True, exist_ok=True)
      self.log_csv = "%s/progress.csv"%(experiment_path)
      csv_header=['episode', 'ep', 'smoothed_running_step_average', 'running_step_average', 'game_win', 'game_win_smoothed', 'mean_score', 'monitor_q_value', 'mean_loss']
      with open(self.log_csv, 'w', encoding='utf-8', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(csv_header)
    # exit(1)
    
  
  def create_env(self):
    self.env = thegame(self.seed, game_mode=self.game_mode, state_history=self.set_history)
    self.env.reset()
  
  def create_models(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.behaviour_network_A = DQNMLPnn(self.input_size, self.num_actions, self.hidden_dim).to(self.device)
    self.target_network_A = DQNMLPnn(self.input_size, self.num_actions, self.hidden_dim).to(self.device)

    self.behaviour_network_A.apply(self.init_weights)
    self.target_network_A.apply(self.init_weights)

    # self.behaviour_network_B = DQNMLPnn(self.input_size, self.num_actions).to(self.device)
    # self.target_network_B = DQNMLPnn(self.input_size, self.num_actions).to(self.device)

    self.optimizer_A = optim.Adam(self.behaviour_network_A.parameters(), lr=self.lr)
    # self.optimizer_B = optim.Adam(self.behaviour_network_B.parameters(), lr=self.lr)

  def init_weights(self, m):
    if type(m) == nn.Linear:
        xavier_uniform(m.weight)
        m.bias.data.fill_(0.01) # turn off 

  def select_action(self, player, state, ep):
    state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device) # unsqueeze(0): [1,2,3,4] -> [[1,2,3,4]]
    legal_actions = self.env.get_valid_actions(player)
    probs = np.zeros(self.num_actions)
    if np.random.random() < ep:
      action_valid = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      if player==0:
        self.behaviour_network_A.eval()
        with torch.no_grad():
          action_values = self.behaviour_network_A(state)
        self.behaviour_network_A.train()
      if player==1:
        # self.behaviour_network_B.eval()
        # with torch.no_grad():
        #   action_values = self.behaviour_network_B(state)
        # self.behaviour_network_B.train()
        self.behaviour_network_A.eval()
        with torch.no_grad():
          action_values = self.behaviour_network_A(state)
        self.behaviour_network_A.train()
      action_ind = np.argmax(action_values.cpu().data.numpy())
      legal_q_values = action_values.cpu().data.numpy()[0][legal_actions]
      action_valid = legal_actions[np.argmax(legal_q_values)]
    action = action_valid
    ep = np.max( [ self.EP_MIN, ( ep - self.EP_DECAY ) ])
    return action, legal_actions, ep

  
  def update(self, player, RM):
    if player==0:
      behaviour_n = self.behaviour_network_A
      target_n = self.target_network_A
      optimizer_n = self.optimizer_A
      RM = self.R_A
    elif player==1:
      # behaviour_n = self.behaviour_network_B
      # target_n = self.target_network_B
      # optimizer_n = self.optimizer_B
      # RM = self.R_B
      behaviour_n = self.behaviour_network_A
      target_n = self.target_network_A
      optimizer_n = self.optimizer_A
      RM = self.R_A
      
    experiences = RM.sample(self.batch_size)
    print(experiences)
    exit(1)
    states, actions, rewards, next_states, dones, legal_actions_mask = experiences
    illegal_actions = 1 - legal_actions_mask
    illegal_logits = illegal_actions * self.ILLEGAL_ACTION_LOGITS_PENALTY

    q_expected = behaviour_n(states).gather(1, actions.long().unsqueeze(1)).squeeze(1) # make use of squeeze to flatten it?
    q_target_next = torch.add(target_n(next_states).detach(), illegal_logits).max(1)[0]
    q_target = rewards + (self.gamma * q_target_next * (1 - dones))
    
    loss = F.mse_loss(q_expected, q_target)
    
    optimizer_n.zero_grad()
    loss.backward()
    optimizer_n.step()
    
    
    return loss, q_expected, q_target

  def soft_update(self, behaviour_n, target_n):
    for target_param, behaviour_param in zip(target_n.parameters(), behaviour_n.parameters()):
      target_param.data.copy_(self.tau * behaviour_param.data + (1.0 - self.tau) * target_param.data)

  def hard_update(self, behaviour_n, target_n):
    for target_param, behaviour_param in zip(target_n.parameters(), behaviour_n.parameters()):
      target_param.data.copy_(behaviour_param.data)
      
  def save_model(self, Epoch, Behaviour_net, Optimizer, Loss, Path):
    torch.save({
                'epoch': Epoch,
                'model_state_dict': Behaviour_net.state_dict(),
                'optimizer_state_dict': Optimizer.state_dict(),
                'loss': Loss,
                }, Path)

  def q_value_monitor(self, behaviour_n):
    states_feed, actions_feed, _, _, _, legal_actions_mask = self.experiences_feed

    behaviour_n.eval()
    with torch.no_grad():
      q_value = behaviour_n(states_feed)
    behaviour_n.train()
    illegal_actions = 1 - legal_actions_mask
    illegal_logits = illegal_actions * self.ILLEGAL_ACTION_LOGITS_PENALTY
    q_value_max=np.amax(torch.add(q_value, illegal_logits).cpu().data.numpy(),axis=1)
    q_value_max_mean = np.mean(q_value_max)
    del states_feed, actions_feed
    return q_value_max_mean

  def act(self, state, learning_start=0):
    self.restart=0

    if self.game_turn_step%2==0:
      player = 0
      player_que = self.env.A_que
      player_transitions = self.nantransitions_A
      partner_transitions = self.nantransitions_B
    else:
      player = 1
      player_que = self.env.B_que
      player_transitions = self.nantransitions_B
      partner_transitions = self.nantransitions_A

    legal_actions = self.env.get_valid_actions(player)
    legal_actions_mask = np.zeros(self.num_actions)
    legal_actions_mask[legal_actions] = 1.0
    if learning_start:
      action = np.random.choice(legal_actions)
    else:
      action, legal_actions, self.ep_A = self.select_action(player, state, self.ep_A)
    next_state_pt, reward, done, info = self.env.step(action, player)
    self.game_turn_step+=1
    if done:
      self.game_win+=1

    now_transition = [state, action, reward, done, legal_actions_mask]
    player_transitions.append(now_transition)

    if len(player_transitions)>1:
      [one_state, one_action, one_reward, one_done, one_legal_actions_mask] = player_transitions[0]
      next_state = state
      # added reward should be the culmulative reward, because here only 2 players so one_reward
      self.R_A.add(one_state, one_action, one_reward, next_state, one_done, one_legal_actions_mask)
    if done or self.game_turn_step>=(self.max_step) or self.last_legal_actions==legal_actions==[10]:
      if learning_start==0:
        self.game_counter+=1 # game counter per g_step
        score=len(self.env.line_asc)+len(self.env.line_des)
        self.score_l.append(score)
      self.restart=1
      next_state = self.env.get_player_observation(player) # player_que
      # 如果当前action导致游戏终止，那无论什么情况的终止，都只能是目前的是next_state
      self.R_A.add(state, action, reward, next_state, done, legal_actions_mask)
      partner_transitions.append([state, None, None, None, None])
      [partner_one_state, partner_one_action, partner_one_reward, partner_one_done, partner_one_legal_actions_mask] = partner_transitions[0]
      # if not done: # if reward > 0: 
      #   reward = -1
      partner_one_reward = reward
      self.R_A.add(partner_one_state, partner_one_action, partner_one_reward, next_state, done, partner_one_legal_actions_mask)
      # print(player, [state, action, reward, next_state, legal_actions_mask])
      # self.env.reset()
      # self.env.render()
      # exit(1)
      state = self.env.reset()

      self.game_turn_step=0
      self.last_legal_actions=[]
      self.nantransitions_A=deque(maxlen=2)
      self.nantransitions_B=deque(maxlen=2)
    else:
      state = next_state_pt # 充分体现了partial observable env的特点
      self.last_legal_actions=legal_actions
    return state, reward
      

    
    
    
  
  def train(self):
    
    self.R_A=ReplayMemoryBuffer(self.replay_buffer_size)
    
    loss_ball_l=[] # print the number of losing the ball in each epoch

    total_reward_l=[]
    running_step_l=[]
    game_win_l=[]
    self.score_l=[]
    self.game_counter=0
    total_reward=0
    running_step=0

    if self.save==True:
      log_dir = './log/'+self.save_file_prefix+'_turn_based_'+self.saving_path_time
      writer = SummaryWriter(log_dir=log_dir)

    # replay start
    state = self.env.reset()
    reward_minus_A = 0
    delay = 0
    self.game_turn_step = 0
    self.last_legal_actions = []
    
    for i_ls in range(self.learning_start):
      state, _ = self.act(state, learning_start=1)

    del state, self.game_turn_step

    self.experiences_feed = self.R_A.sample(self.monitor_n)
    q_value_A = self.q_value_monitor(self.behaviour_network_A)     
    logging.info("average max q value of A before training: {:.6f}".format(q_value_A))

    t_g_step=0
    temp_game_win=0
    update_count=0

    state = self.env.reset()
    for g_step in range(1, self.episodes + 1): # episode here = epoch
      
      episode_start_time=time.time()
      
      logging.info("g_step: {}".format(g_step))

      self.game_turn_step=0
      self.last_legal_actions=[]
      self.nantransitions_A=deque(maxlen=2)
      self.nantransitions_B=deque(maxlen=2)

      training_counter = 0
      lose_ball_counter = 0
      
      running_loss_A = 0
      running_loss_B = 0

      
      running_qae_mean=0
      running_qat_mean=0
      running_qbe_mean=0
      running_qbt_mean=0

      self.game_win=0
      self.score_l=[]
      
      # for t_step in range(self.turn_step):
      while True:
        t_g_step+=1
        
        training_counter+=1
        state, reward = self.act(state, learning_start=0) 

        total_reward = total_reward + reward
       
        if (t_g_step+1) % self.update_freq == 0:
          if len(self.R_A) > self.batch_size:
            self.loss_A, q_A_e, q_A_t = self.update(0, self.R_A)

            update_count+=1

            if (t_g_step+1) % self.update_target == 0:
              self.hard_update(self.behaviour_network_A, self.target_network_A)
              # self.soft_update(self.behaviour_network_A, self.target_network_A)

              logging.info("target network updated")

            
            running_loss_A += self.loss_A.item()
            
            running_qae_mean += q_A_e.cpu().data.numpy().mean()
            running_qat_mean += q_A_t.cpu().data.numpy().mean()
            
        if t_g_step % self.running_p == (self.running_p - 1):
          mean_loss_A = running_loss_A/update_count
          logging.info('[{}, {}] avg loss: {:.6f}'.format(g_step, t_g_step + 1, mean_loss_A))
          logging.info("average expected q value A: {:.6f}".format(running_qae_mean/update_count))
          running_loss_A = 0
              
          running_qae_mean=0
          running_qat_mean=0
  
          update_count=0

          q_value_A = self.q_value_monitor(self.behaviour_network_A)    
          logging.info("average max q value of A: {:.6f}".format(q_value_A))

              
        if self.game_counter==self.turn_game:
          logging.info("{}, break".format(self.game_counter))
          break

      running_step = t_g_step-running_step
      running_step_l.append(running_step)
      smoothed_running_step=np.mean(running_step_l[-30:])
      if self.save==True:
        checkpoint_dir = './checkpoint/%s_turn_based_%s'%(self.save_file_prefix, self.saving_path_time)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        path_A='%s/checkpoint_g%d_A'%(checkpoint_dir, g_step)
        if self.game_win>temp_game_win:
          self.save_model(g_step, self.behaviour_network_A, self.optimizer_A, self.loss_A, path_A)
          temp_game_win = self.game_win
      
      logging.info("game counter per g_step: {}".format(self.game_counter))
      self.game_counter=0
      q_value_A = self.q_value_monitor(self.behaviour_network_A)     
      # q_value_B = self.q_value_monitor(self.behaviour_network_B)   
      logging.info("after epoch {:d}, average max q value of A: {:.3f}".format(g_step, q_value_A))
      game_win_l.append(self.game_win)
      smoothed_game_win=np.mean(game_win_l[-30:])
      
      if self.save==True:
        writer.add_scalar('epsilon', self.ep_A, g_step) # t_g_step
        writer.add_scalar('smoothed_running_step', smoothed_running_step, g_step) # t_g_step
        writer.add_scalar('mean_max_q_value', q_value_A, g_step)
        writer.add_scalar('game_win', smoothed_game_win, g_step)
        writer.add_scalar('average_score', np.mean(self.score_l), g_step)
        writer.add_scalar('loss', mean_loss_A, g_step)
        with open(self.log_csv, 'a', encoding='utf-8', newline='') as f:
          writer_csv = csv.writer(f)
          writer_csv.writerow([g_step, self.ep_A, running_step/self.turn_game, smoothed_running_step/self.turn_game, self.game_win, smoothed_game_win, np.mean(self.score_l), q_value_A, mean_loss_A])

      logging.info("game_win: {}".format(self.game_win))
      running_step=t_g_step

      
      episode_end_time=time.time()
      logging.info("epoch {:d} training time: {:.3f}s".format(g_step, (episode_end_time-episode_start_time)))

    








