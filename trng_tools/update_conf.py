import yaml
import os


conf_file_paths = []
for exp_log in os.listdir("./logs"):

    if exp_log != 'cassie_results':

        if exp_log == 'sub_optimal_policies':
            for sub_exp_log in os.listdir("./logs/sub_optimal_policies"):
                conf_file_paths.append("./logs/sub_optimal_policies/"+sub_exp_log+"/exp_conf.yaml")

        else:
            conf_file_paths.append("./logs/"+exp_log+"/exp_conf.yaml")

# print(conf_file_paths)



for file_path in conf_file_paths:
    # file_path = "./logs/"+exp_log+"/exp_conf.yaml"
    trng_exp_conf_file = open(file_path) # remove
    trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
    trng_exp_conf_file.close()

    print(file_path)
    if 'commands' in trng_exp_conf.keys():
        temp_dict = {}
        for key in trng_exp_conf['commands'].keys():
            temp_dict.update({key+'_lim':trng_exp_conf['commands'][key]})
        keys_2_remove = list(trng_exp_conf['commands'].keys())
        for key in keys_2_remove:
            trng_exp_conf['commands'].pop(key)
        trng_exp_conf['commands'].update(temp_dict)

    '''
    # gen params
    trng_exp_conf['visualize_reference'] = False
    trng_exp_conf.pop('dynamics_randomization')

    # sim_params
    if 'target_marker' in trng_exp_conf['sim_params']['render'].keys():
        trng_exp_conf['sim_params'].update({'target_marker': trng_exp_conf['sim_params']['render']['target_marker']})            
    trng_exp_conf['sim_params'].pop('render')
    trng_exp_conf['sim_params'].update(
                                        {
                                            'render':False,
                                            'init_viewer_every_reset':False,

                                        })
    # terrain randomisation
    if 'terrain' in trng_exp_conf['sim_params'].keys():
        for key in trng_exp_conf['sim_params']['terrain'].keys():
            
            temp_dict = {}
            for prop in trng_exp_conf['sim_params']['terrain'][key].keys():

                if 'xlim' in prop:
                    temp_dict.update({
                                        prop.replace('xlim','x')+'_lim':
                                        trng_exp_conf['sim_params']['terrain'][key][prop]
                                        })
                elif 'ylim' in prop:
                    temp_dict.update({
                                    prop.replace('ylim','y')+'_lim':
                                    trng_exp_conf['sim_params']['terrain'][key][prop]
                                    })                        
                else:
                    temp_dict.update({
                                    prop+'_lim':
                                    trng_exp_conf['sim_params']['terrain'][key][prop]
                                    })
            
            all_keys = list(trng_exp_conf['sim_params']['terrain'][key].keys())
            for prop in all_keys:
                trng_exp_conf['sim_params']['terrain'][key].pop(prop)
            
            trng_exp_conf['sim_params']['terrain'][key].update(temp_dict)       
        
    # update trajectory
    if 'traj_slice_t' in trng_exp_conf.keys():
        trng_exp_conf.update({"traj_slice_t_lim":trng_exp_conf.pop('traj_slice_t') })
    

    '''
    upd_trng_exp_conf_file = open(file_path,'w') # remove
    yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)


# for file_name in os.listdir('./exp_confs'):
    
    
#     file_path = os.path.join('./exp_confs',file_name)
#     if 'skate_wheels'in file_name:
#         trng_exp_conf_file = open(file_path) # remove
#         trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
#         trng_exp_conf_file.close()
#         trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')

#         os.remove(file_path)
#         file_path = file_path.replace('wheels','board')
#         upd_trng_exp_conf_file = open(file_path,'w') # remove
#         yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
    
#     if 'skate_board' in file_name:
#         trng_exp_conf_file = open(file_path) # remove
#         trng_exp_conf = yaml.load(trng_exp_conf_file, Loader=yaml.FullLoader)
#         trng_exp_conf_file.close()
#         # trng_exp_conf['sim_params']['model_path'] = trng_exp_conf['sim_params']['model_path'].replace('wheels','skateboard')
#         # try:
#         #     trng_exp_conf['observations'].pop('robot_wheel_vel')
        
#         if 'set_robot_on_traj' in trng_exp_conf['initialisations'].keys():
#             trng_exp_conf['initialisations'].pop('set_robot_on_traj')
#             trng_exp_conf['initialisations'].update({'set_robot_on_skateboard_pose':None})
#         # except:
#         #     print(file_name,trng_exp_conf['observations'])
#         # trng_exp_conf['observations'].update({'skateboard_tvel':None})
#         # trng_exp_conf['observations'].update({'skateboard_avel':None})

#         # os.remove(file_path)
#         # file_path = file_path.replace('wheels','board')
#         upd_trng_exp_conf_file = open(file_path,'w') # remove
#         yaml.dump(trng_exp_conf,upd_trng_exp_conf_file,default_flow_style=False,sort_keys=False)
