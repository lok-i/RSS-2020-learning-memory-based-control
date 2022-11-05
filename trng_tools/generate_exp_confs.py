import yaml
import copy


base_exp_conf_file = open('./exp_confs/one_clip.yaml') 
base_exp_conf = yaml.load(base_exp_conf_file, Loader=yaml.FullLoader)
vary_trng_params_file = open('./exp_confs/trng_parms_to_vary.yaml') # remove
params_to_vary =  yaml.load(vary_trng_params_file, Loader=yaml.FullLoader)


exp_id = 0
for param in params_to_vary.keys():
    param_path = param.split('/')

    print(param_path)
    
    this_variant = copy.deepcopy(base_exp_conf)
    val = this_variant

    this_variant['sim_params']['render'] = False
    this_variant['visualize_reference'] = False
    this_variant.pop('return_rew_dict')

    for i in range(len(param_path)-1):
        val = val[param_path[i]]
    val.pop(param_path[-1])

    for param_variant_val in params_to_vary[param]:

        val.update({param_path[-1]:param_variant_val})
        trng_exp_conf_file =  open('./exp_confs/oce_0'+str(exp_id)+'.yaml','w')
        yaml.dump(this_variant,trng_exp_conf_file,default_flow_style=False,sort_keys=False)
        exp_id+=1
