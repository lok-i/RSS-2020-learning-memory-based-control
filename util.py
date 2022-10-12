

from cmath import exp
from math import fabs
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
                return_traj=False,
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
    
    qpos_trajs = []


    if env.sim.sim_params['render']['active']:
      env.sim.viewer._paused = True
      env.sim.viewer.cam.distance = 3
      cam_pos = [0.0, 0.0, 0.75]

      for i in range(3):        
          env.sim.viewer.cam.lookat[i]= cam_pos[i] 
      env.sim.viewer.cam.elevation = -15
      env.sim.viewer.cam.azimuth = 180



    for _ in range(episodes):
      env.dynamics_randomization = False
      
      vx_d = None
      state = torch.Tensor(env.reset(vx_des=vx_d))


      done = False
      traj_len = 0
      ep_return = 0

    
      

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()
      
      qpos_traj = []
      while not done and traj_len < max_traj_len:

        if return_traj:
          qpos_traj.append(env.sim.data.qpos[:])
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
      
      if return_traj:
        qpos_traj = np.array(qpos_traj,dtype=list)
        qpos_trajs.append(qpos_traj)
  
  if return_traj:
    return np.mean(ep_returns),qpos_trajs
  else:
    return np.mean(ep_returns)

def eval_policy_to_plot(
                model, 
                env=None, 
                episodes=5, 
                max_traj_len=400, 
                verbose=True, 
                visualize=False,
                return_traj=False, 
                save_logpath = None,
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
    
    
    # qpos_trajs = []

    # data collection

    vxs_des = np.arange(-4,4,0.2)
    # vxs_des = np.arange(-2.0,-1.6,0.2)

    n_exps = vxs_des.shape[0]
    n_trials = episodes

    ep_returns = np.zeros((n_exps,n_trials),dtype=np.float32)
    vxs_0 = np.zeros((n_exps,n_trials),dtype=np.float32)
    dist_travelled = np.zeros((n_exps,n_trials),dtype=np.float32)
    e_vx_norm_avg = np.zeros((n_exps,n_trials),dtype=np.float32)
    
    for exp_no,vx_d in enumerate(vxs_des):
      print("exp no:",exp_no,'vx_des',vx_d)
      for trial_no in range(n_trials):
        env.dynamics_randomization = False

        state = torch.Tensor(env.reset(
                                      vx_des=vx_d

                                      ))
        
        vxs_0[exp_no][trial_no] = env.ref_qvel0[0]        
        done = False
        traj_len = 0
        ep_return = 0
        vx_error_norm_avg = 0 
        if hasattr(policy, 'init_hidden_state'):
          policy.init_hidden_state()
        
        # qpos_traj = []
        qpos = None
        while not done and traj_len < max_traj_len:
          

          # print(type(env.sim.qpos()))
          # qpos_traj.append(env.sim.qpos())
          
          action = policy(state)
          next_state, reward, done, _ = env.step(action.numpy())
          qpos = env.sim.qpos()

          vx_error_norm = fabs(env.sim.qvel()[0] - env.speed)
          vx_error_norm_avg += vx_error_norm
          # print(next_state.shape)

          # if visualize:
          #   env.render()
          state = torch.Tensor(next_state)

          ep_return += reward
          traj_len += 1
          steps += 1

          if model.nn_type == 'extractor':
            pass


        if verbose:
          print('\ttrial no:',trial_no,'Return: {:6.2f}'.format(ep_return))
          
        # qpos_traj = np.array(qpos_traj,dtype=list)
        # qpos_trajs.append(qpos_traj)
        
        ep_returns[exp_no][trial_no] = ep_return
        e_vx_norm_avg[exp_no][trial_no] = vx_error_norm_avg / steps 
        delta_dist = qpos[0] - env.ref_qpos0[0]
        dist_travelled[exp_no][trial_no] = delta_dist
        # exit()
        
  if save_logpath != None:
    np.savez_compressed(
                        save_logpath+'vx_d_exps',
                        vxs_des = vxs_des,
                        vxs_0 = vxs_0,
                        dist_travelled = dist_travelled,
                        ep_returns = ep_returns,
                        e_vx_norm_avg = e_vx_norm_avg,

                        )

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
      from dtsd.envs.biped_osudrl import biped_env
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