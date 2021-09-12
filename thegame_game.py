import sys
sys.executable

# seed = 102

import numpy as np
# np.random.seed(seed)

import random
# random.seed(seed)

from collections import deque, namedtuple
import time
import warnings
warnings.filterwarnings('ignore')





card_num_per_player = 2
card_sum = 50
color_sum = 5

class thegame():
  def __init__(self, seed=17, game_mode='no-commu', state_history=False):
    self.num_actions = 11
    
    self.colors=np.arange(color_sum+1)[1:]
    self.running_step_check=True
    self.running_step=0

    self.seed = seed
    np.random.seed(self.seed)
    random.seed(self.seed)
    self.mode = game_mode
    self.state_history = state_history


  def reset(self, n=card_sum):
    self.A_que = deque([], maxlen=card_num_per_player)
    self.B_que = deque([], maxlen=card_num_per_player)

    self.line_asc = deque([], maxlen=card_sum)
    self.line_des = deque([], maxlen=card_sum)

    self.Card_que = deque([], maxlen=card_sum)
    self.running_step=0

    for i in range(len(self.colors)):
      for j in range(1, int(n/color_sum)+1):
        self.Card_que.append((self.colors[i],j))
    np.random.shuffle(self.Card_que)

    for k in range(card_num_per_player):
      new_card_A = self.Card_que.pop()
      self.A_que.append(new_card_A)
      new_card_B = self.Card_que.pop()
      self.B_que.append(new_card_B)
    state = self.get_player_observation('player_0')
    return state
      
  def get_stack_state(self):
    asc = self.line_asc
    des = self.line_des
    if len(self.line_asc)==0:
      asc=[(0, 0)]
    if len(self.line_des)==0:
      des=[(0, 0)]
    return [asc[-1]]+[des[-1]]

  def get_player_observation(self, player): # get_player_que_state
    if player=='player_0' or player==0:
      partner = 'player_1'
      player_que = self.A_que
      partner_que = self.B_que
    elif player=='player_1' or player==1:
      partner = 'player_0'
      player_que = self.B_que
      partner_que = self.A_que
    stack_state = self.get_stack_state()
    if len(player_que) == 0:
      player_que = [(0, 0), (0, 0)]
    elif len(player_que) == 1:
      player_que = [player_que[-1]] + [(0, 0)]
    if len(partner_que) == 0:
      partner_que = [(0, 0), (0, 0)]
    elif len(partner_que) == 1:
      partner_que = [partner_que[-1]] + [(0, 0)]
    if self.mode=='no-commu':
      state = stack_state+list(player_que)
    elif self.mode=='absolute-quantifier':
      state = stack_state+list(player_que)+list(partner_que)
    elif self.mode=='fuzzy-quantifier':
      partner_que_p = deque([], maxlen=card_num_per_player)
      for i in range(len(partner_que)):
        if partner_que[i][-1]<=5:
          partner_que_p.append((partner_que[i][0], 0))
        elif partner_que[i][-1]>5:
          partner_que_p.append((partner_que[i][0], 1))
      state = stack_state+list(player_que)+list(partner_que_p)
    state = [item for sublist in state for item in sublist] # /10
    state = np.array(state, dtype=np.float32)
    if self.state_history==True:
      card_h = self.get_card_history()
      # print(card_h)
      state = np.concatenate((state, card_h), axis=None)
      # print(state)
    return state

  def get_card_history(self, ):
    if self.state_history==True:
      # print("1111")
      history = np.zeros(card_sum)
      card_per_color = int(card_sum/color_sum)
      for i in range(color_sum):
        for j in range(int(card_per_color)):
          ind_color = i + 1
          ind_num = j + 1
          if (ind_color, ind_num) in self.line_asc or (ind_color, ind_num) in self.line_des:
            ind = i * card_per_color + j
            history[ind] = 1.0
      # print("card_history in get card history: ", history)
      return history
    

  def card_draw(self, player_stack):
    if len(self.Card_que)!=0:
      player_stack_length = len(player_stack)
      add_length = card_num_per_player - player_stack_length
      for i in range(add_length):
        if len(self.Card_que)>0:
          new_card = self.Card_que.pop()
          player_stack.append(new_card)
    else:
      #print("Car queue is empty!")
      pass

  def check_single_line(self, new_card, order=0):
    if order==0:
      current_line = self.line_asc
    elif order==1:
      current_line = self.line_des
    else:
      print("line error input")
      return False

    if len(current_line)==0 or new_card[0] == current_line[-1][0]:
      # current_line.append(new_card)
      return True
    else:
      if order==0: # ascending order
        if new_card[1] > current_line[-1][1]:
          # current_line.append(new_card)
          return True
        elif new_card[1] < current_line[-1][1]:
          return False # not allowed
        else:
          #print("error input")
          return False
      if order==1: # descending order
        if new_card[1] < current_line[-1][1]:
          # current_line.append(new_card)
          return True
        elif new_card[1] > current_line[-1][1]:
          return False
        else:
          #print("error input")
          return False

  def check_single_distance(self, card, order=0):
    allowed = self.check_single_line(card, order)
    distance = np.inf
    if allowed == True:
      if order==0:
        distance = card[1]-1
      if order==1:
        distance = 10-card[1]
      return distance
    return False

  def play_card(self, card, order=0): # play this card
    # if self.game_end():
    #   print("game ends")
    #   return 0

    allowed = self.check_single_line(card, order)
    if not allowed:
      # print("operation not permitted ")
      return False
    else: 
      if order == 0:
        self.line_asc.append(card)
        return True
      elif order == 1:
        self.line_des.append(card)
        return True
      else:
        print("wrong input")
        return False

  def remove_card(self, order=0):
    if order==0:
      self.line_asc.pop()
    if order==1:
      self.line_des.pop()

  def get_valid_actions(self, player='player_0'):
    # there are eight available actions 
    if player=='player_0' or player==0:
      player_que = self.A_que
      partner_que = self.B_que
    elif player=='player_1' or player==1:
      player_que = self.B_que
      partner_que = self.A_que
      
    legal_actions = []
    allowed = 0

    for i in range(self.num_actions): 
      action = i
      allowed = -1
      if (action!=0 and action!=4 and action!=10) and (len(player_que)==1):
        allowed = False
        continue
      if len(player_que)==0:
        legal_actions.append(10)
        break
      if action==0:
        allowed = self.check_single_line(player_que[0], 0)

      # ASC right
      if action==1:
        allowed = self.check_single_line(player_que[1], 0)

      # ASC left right
      if action==2:
        allowed_l = self.check_single_line(player_que[0], 0)
        if allowed_l:
          self.play_card(player_que[0], 0)
          allowed_r = self.check_single_line(player_que[1], 0)
          if allowed_r:
            allowed = 1
          self.remove_card(0)
          
      # ASC right left
      if action==3:
        allowed_r = self.check_single_line(player_que[1], 0)
        if allowed_r:
          self.play_card(player_que[1], 0)
          allowed_l = self.check_single_line(player_que[0], 0)
          if allowed_l:
            allowed = 1
          self.remove_card(0)
        

      # DES left
      if action==4:
        allowed = self.check_single_line(player_que[0], 1)

      # DES right
      if action==5:
        allowed = self.check_single_line(player_que[1], 1)

      # DES left right
      if action==6:
        allowed_l = self.check_single_line(player_que[0], 1)
        if allowed_l:
          self.play_card(player_que[0], 1)
          allowed_r = self.check_single_line(player_que[1], 1)
          if allowed_r:
            allowed = 1
          self.remove_card(1)
          
      # DES right left
      if action==7:
        allowed_r = self.check_single_line(player_que[1], 1)
        if allowed_r:
          self.play_card(player_que[1], 1)
          allowed_l = self.check_single_line(player_que[0], 1)
          if allowed_l:
            allowed = 1
          self.remove_card(1)
      
      # ASC left, DES right
      if action==8:
        allowed_l = self.check_single_line(player_que[0], 0)
        allowed_r = self.check_single_line(player_que[1], 1)
        if allowed_l and allowed_r:
          allowed = 1
          
      # ASC right, DES left
      if action==9:
        allowed_l = self.check_single_line(player_que[0], 1)
        allowed_r = self.check_single_line(player_que[1], 0)
        if allowed_l and allowed_r:
          allowed = 1
          
      if action==10:
        allowed = 1 

      if allowed == 1:
        legal_actions.append(action)
    # legal_actions = np.array(legal_actions)

    return legal_actions

  def get_legal_actions_mask(self, player='player_0'):
    """
    if player==0:
      player='player_0'
    elif player==1:
      player='player_1'
    """
    legal_actions = self.get_valid_actions(player)
    legal_actions_mask = np.zeros(self.num_actions)
    legal_actions_mask[legal_actions] = 1.0
    return legal_actions_mask
    
      
      
  def step(self, action, player='player_0'):
    # print("action received for this round: ", action)
    # there are eight available actions 
    if player=='player_0' or player==0:
      partner = 'player_1'
      player_que = self.A_que
      partner_que = self.B_que
    elif player=='player_1' or player==1:
      partner = 'player_0'
      player_que = self.B_que
      partner_que = self.A_que
      
    allowed = 0
    reward = 0
      
    if (action!=0 and action!=4 and action!=10) and (len(player_que)==1):  
      allowed = False
    elif len(player_que)==0:
      # print("player_que == 0 ..")
      allowed=1
      action=10
    else:
      # print(action, len(player_que), len(self.Card_que))
      # ASC left
      if action==0:
        allowed = self.play_card(player_que[0], 0)
        if allowed:
          del player_que[0]
          # pass
          # self.card_draw(player_que)

      # ASC right
      if action==1:
        allowed = self.play_card(player_que[1], 0)
        # print("allowed: ", allowed)
        if allowed:
          del player_que[1]
          # self.card_draw(player_que)
      # ASC left right
      if action==2:
        allowed_l = self.play_card(player_que[0], 0)
        allowed_r = self.play_card(player_que[1], 0)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[0]
          del player_que[0]
          # self.card_draw(player_que)
        if allowed_l != allowed_r:
          self.remove_card(order=0)
        
          
      # ASC right left
      if action==3:
        allowed_r = self.play_card(player_que[1], 0)
        allowed_l = self.play_card(player_que[0], 0)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[0]
          del player_que[0] # 自动进位到1
          # self.card_draw(player_que)
          # print(allowed)
        if allowed_l != allowed_r:
          self.remove_card(order=0)
        
      # DES left
      if action==4:
        allowed = self.play_card(player_que[0], 1)
        if allowed:
          del player_que[0]
          # self.card_draw(player_que)
      # DES right
      if action==5:
        allowed = self.play_card(player_que[1], 1)
        if allowed:
          del player_que[1]
          # self.card_draw(player_que)
      # DES left right
      if action==6:
        allowed_l = self.play_card(player_que[0], 1)
        allowed_r = self.play_card(player_que[1], 1)
        # print("aaa,", allowed_l, allowed_r)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[1]
          del player_que[0]
          # self.card_draw(player_que)
        if allowed_l != allowed_r:
          self.remove_card(order=1)
          
      # DES right left
      if action==7:
        allowed_r = self.play_card(player_que[1], 1)
        allowed_l = self.play_card(player_que[0], 1)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[1]
          del player_que[0]
          # self.card_draw(player_que)
        if allowed_l != allowed_r:
          self.remove_card(order=1)
      
      # ASC left, DES right
      if action==8:
        allowed_l = self.play_card(player_que[0], 0)
        allowed_r = self.play_card(player_que[1], 1)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[1]
          del player_que[0]
          # self.card_draw(player_que)
        if allowed_l and not allowed_r:
          self.remove_card(order=0)
        if not allowed_l and allowed_r:
          self.remove_card(order=1)
          
      # ASC right, DES left
      if action==9:
        allowed_l = self.play_card(player_que[0], 1)
        allowed_r = self.play_card(player_que[1], 0)
        if allowed_l and allowed_r:
          allowed = 1
          del player_que[1]
          del player_que[0]
          # self.card_draw(player_que)
        if allowed_l and not allowed_r:
          self.remove_card(order=0)
        if not allowed_l and allowed_r:
          self.remove_card(order=1)
          
      if action==10:
        # print("???what happened??")
        allowed = 1
        reward=0 # -0.1, -0.01
        pass
    
    if allowed:
      reward = 0 # 1 # -1
      if len(self.Card_que)>0:
        self.card_draw(player_que)
      # if self.running_step>50:?
    # elif action!=10:
    #   reward = -1
    # elif action==10:
    if len(self.get_valid_actions(player))==1 and len(player_que)!=0:
      reward=0 # -1
      # print(reward)
      # allowed=1
      # exit(1)

      
    
    next_state_partner = self.get_player_observation(partner)
    # next_state_player = self.get_player_que_state(player_que) # this is not the real next state
    
    done = self.game_end()
    if done:
      reward = 1
    elif action==10:
      reward = 0 # -0.01, -0.1
    # else: # new
      # reward = 0 # new # 0
    # elif (action==10) and (len(self.get_valid_actions(player))==1) and (len(player_que)!=0):
      # print("?")
      # print(allowed)
      # print(self.get_valid_actions(player))
      # allowed=True
      # reward=-100
      # print(reward)
      # exit(1)

    
    info = None

    if self.running_step_check==True:
      self.running_step+=1
      if self.running_step>=100:
        # done=False
        info = "Last Turn Ended"
        self.running_step=0
    # print(reward)
    # print(allowed)
    # print("!!!", action, allowed)
    if not allowed:
      print("it is not allowed, but why?")
      print(self.get_valid_actions(player), action)
      return False
    
    return next_state_partner, reward, done, info


  def manual_play(self, player, order_t, card_index):
    if player=='player_0' or player==0:
      player_stack=self.A_que
    elif player=='player_1' or player==1:
      player_stack=self.B_que
    else:
      print("wrong player stack input")

    allowed = self.check_single_line(player_stack[card_index], order=order_t)
    if allowed:
      self.play_card(player_stack[card_index], order=order_t)
      del player_stack[card_index]
    else:
      print("wrong card, card: ", player_stack[card_index])
      return False
    self.card_draw(player_stack)


  def render(self):
    print("Current Card Queue: ")
    print(self.Card_que)
    print("\nLine Asc: ")
    print(self.line_asc)
    print("\nLine Des: ")
    print(self.line_des)
    print("\nA Queue: ")
    print(self.A_que)
    print("\nB Queue: ")
    print(self.B_que)


  def game_end(self):
    stack_length = len(self.Card_que)
    if stack_length == 0 and (len(self.A_que)==0 and len(self.B_que)==0):
      # self.render()
      # print("!!!!!!!",stack_length, self.A_que, self.B_que)
      return True
    return False
