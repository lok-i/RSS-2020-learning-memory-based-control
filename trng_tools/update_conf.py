import yaml
import os


for file_name in os.listdir('./exp_confs'):
    file_path = os.path.join('./exp_confs',file_name)
    if 'skate_wheels'in file_name:
        trng_exp_conf_file = open(file_path) # remove
        trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
        trng_exp_conf_file.close()
        trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')

        os.remove(file_path)
        file_path = file_path.replace('wheels','board')
        upd_trng_exp_conf_file = open(file_path,'w') # remove
        yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
    
    if 'skate_board' in file_name:
        trng_exp_conf_file = open(file_path) # remove
        trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
        trng_exp_conf_file.close()
        # trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')
        # try:
        #     trng_exp_conf['observations'].pop('robot_wheel_vel')
        
        if 'set_robot_on_traj' in trng_exp_conf['initialisations'].keys():
            trng_exp_conf['initialisations'].pop('set_robot_on_traj')
            trng_exp_conf['initialisations'].update({'set_robot_on_skateboard_pose':None})
        # except:
        #     print(file_name,trng_exp_conf['observations'])
        # trng_exp_conf['observations'].update({'skateboard_tvel':None})
        # trng_exp_conf['observations'].update({'skateboard_avel':None})

        # os.remove(file_path)
        # file_path = file_path.replace('wheels','board')
        upd_trng_exp_conf_file = open(file_path,'w') # remove
        yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
