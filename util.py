

from cmath import exp
from math import fabs
from tkinter import E
import torch
import hashlib
import os
import numpy as np
from collections import OrderedDict
import shutil
import yaml


def create_logger(args):
  from torch.utils.tensorboard import SummaryWriter
  """Use hyperparms to set a directory to output diagnostic files."""

  arg_dict = args.__dict__
  assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  if 'seed' in arg_dict:
    seed = str(arg_dict.pop("seed"))
  else:
    seed = None
  
  logdir = str(arg_dict.pop('logdir'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  if seed is None:
    arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6]
  else:
    arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed

  # output_dir = os.path.join(logdir, arg_hash)
  conf_name = args.exp_conf_path.split('/')[-1]
  output_dir = os.path.join(logdir, conf_name.replace('.yaml',''))

  # create a directory with the hyperparm hash as its name, if it doesn't
  # already exist.
  os.makedirs(output_dir, exist_ok=True)

  # Create a file with all the hyperparam settings in plaintext
  info_path = os.path.join(output_dir, "experiment.info")
  file = open(info_path, 'w')
  for key, val in arg_dict.items():
      file.write("%s: %s" % (key, val))
      file.write('\n')

  # copy the exp_conf_file
  default_env_conf_path = './exp_confs/default.yaml'

  default_conf_file = open(default_env_conf_path)
  default_exp_conf = yaml.load(default_conf_file, Loader=yaml.FullLoader)

  given_conf_file = open(args.exp_conf_path) # remove
  given_exp_conf = yaml.load(given_conf_file, Loader=yaml.FullLoader)

  merged_exp_conf = default_exp_conf
  for key in given_exp_conf.keys():


      if key in merged_exp_conf.keys():
          merged_exp_conf[key] = given_exp_conf[key]
      else:
          merged_exp_conf.update({key:given_exp_conf[key]})

  args.exp_conf_path = os.path.join(output_dir,'exp_conf.yaml')
  final_conf_file =  open(args.exp_conf_path,'w')
  yaml.dump(merged_exp_conf,final_conf_file,default_flow_style=False,sort_keys=False)

  logger = SummaryWriter(output_dir, flush_secs=0.1)
  logger.dir = output_dir

  logger.arg_hash = arg_hash
  return logger

def train_normalizer(policy,
                     min_timesteps, 
                     max_traj_len=1000, 
                     noise=0.5,
                     exp_conf_path="./exp_confs/default.yaml"
                     ):
  with torch.no_grad():
    env = env_factory(exp_conf_path)()
    env.dynamics_randomization = False

    total_t = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1

def eval_policy(model, 
                env=None, 
                episodes=5, 
                max_traj_len=400, 
                verbose=True, 
                exp_conf_path = './exp_confs/default.yaml'
                ):
  if env is None:
    env = env_factory(exp_conf_path)()

  if model.nn_type == 'policy':
    policy = model
  elif model.nn_type == 'extractor':
    policy = torch.load(model.policy_path)

  with torch.no_grad():
    steps = 0
    ep_returns = []
    
    for _ in range(episodes):
      env.dynamics_randomization = False
      
      state = torch.Tensor(env.reset())
      done = False
      traj_len = 0
      ep_return = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()
      
      while not done and traj_len < max_traj_len:

        action = policy(state)
        next_state, reward, done, _ = env.step(action.numpy())

        state = torch.Tensor(next_state)

        ep_return += reward
        traj_len += 1
        steps += 1

        if model.nn_type == 'extractor':
          pass

      ep_returns += [ep_return]
      if verbose:
        print('Return: {:6.2f}'.format(ep_return))
      
  return np.mean(ep_returns)

def eval_policy_to_plot(
                model, 
                env=None, 
                episodes=5, 
                max_traj_len=400, 
                verbose=True, 
                return_traj=False,
                exp_conf_path = './exp_confs/default.yaml',
                plotter=None
                ):
  if env is None:
    env = env_factory(exp_conf_path)()

  if model.nn_type == 'policy':
    policy = model
  elif model.nn_type == 'extractor':
    policy = torch.load(model.policy_path)

  with torch.no_grad():
    
    ep_returns = []

    for _ in range(episodes):
      env.dynamics_randomization = False





      state = torch.Tensor(env.reset())


      if env.sim.sim_params['render']['active']:
      
      #   for att in dir(env.sim.viewer):
      #     if '_' in att and '__' not in att: 
      #       print(att,getattr(env.sim.viewer,att))
        env.sim.viewer._render_every_frame = False
        env.sim.viewer._paused = True
        env.sim.viewer.cam.distance = 3
        cam_pos = [0.0, 0.0, 0.75]

        for i in range(3):        
            env.sim.viewer.cam.lookat[i]= cam_pos[i] 
        env.sim.viewer.cam.elevation = -15
        env.sim.viewer.cam.azimuth = 90



      done = False
      traj_len = 0
      ep_return = 0

      # set all loggers
      vel_head_d = []
      vel_head = []
  
      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()
      
      steps = 0
      # initialised to 0, hacky though.. suddn drop to 0 from reset 
      try:
        env.speed = 0.0 #np.random.choice([0.1,-0.1])
      except:        
        env.cmnd_base_tvel[0] = 0 #np.random.choice([0.1,-0.1])

      while not done and traj_len < max_traj_len:
        
        
        
        action = policy(state)
        next_state, reward, done, _ = env.step(action.numpy())
        
        try:
          if steps % 40 == 0:
            env.speed += 0.1 #np.random.choice([0.1,-0.1])
            env.speed = np.clip(env.speed,0,1.6)
            print("updated speed command:",env.speed)
          vel_head_d.append(env.speed)

        except:
          
          if steps % 40 == 0:
            env.cmnd_base_tvel[0] += 0.1 #np.random.choice([0.1,-0.1])
            env.cmnd_base_tvel[0] = np.clip(env.cmnd_base_lvel[0],0,1.6)
            print("updated speed command:",env.cmnd_base_lvel[0])
          vel_head_d.append(env.cmnd_base_tvel[0])

        vel_head.append(env.sim.data.qvel[ env.model_prop['drcl_biped']['ids']['base_tvel'][0] ])

        state = torch.Tensor(next_state)

        ep_return += reward
        traj_len += 1
        steps += 1

        if model.nn_type == 'extractor':
          pass

      ep_returns += [ep_return]
      if verbose:
        print('Return: {:6.2f}'.format(ep_return))
      
      if plotter != None:
        import matplotlib.pyplot as plt
        rollout_timestep = env.dt*np.arange(steps)
        plt.plot(rollout_timestep,vel_head_d,label='vel_desired')
        plt.plot(rollout_timestep,vel_head,'--',label='vel_actual')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

  return np.mean(ep_returns)

def env_factory(
                # dynamics_randomization,
                exp_conf_path,
                ):
    from functools import partial


    conf_file = open(exp_conf_path)
    exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)

    """
    Returns an *uninstantiated* environment constructor.
    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    """
    
    
    if 'robot' in exp_conf.keys() and exp_conf['robot'] == 'drcl_biped':
      import importlib

      if 'env_entry' in exp_conf.keys():
        env_class_name = exp_conf['env_entry'].split('.')[-1]
        env_file_entry = exp_conf['env_entry'].replace('.'+env_class_name,'')
        env_module = importlib.import_module(env_file_entry)
        biped_env = getattr(env_module,env_class_name) 
      else:
        biped_env = importlib.import_module('dtsd.envs.biped_osudrl').biped_env

      return partial(
                      biped_env, 
                      exp_conf_path = exp_conf_path,
                      )
    else:
      from cassie.cassie import CassieEnv
      return partial(
                      CassieEnv, 
                      exp_conf_path = exp_conf_path,
                      )