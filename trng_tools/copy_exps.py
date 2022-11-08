
import os
import csv
from datetime import datetime


if __name__ == '__main__':
    
    exp_names = []
    remote_log_path = ' lkrajan@discovery.usc.edu:/home1/lkrajan/baseline_projects/RSS-2020-learning-memory-based-control/logs/'
    exp_confs = os.listdir('./exp_confs')
    command = 'rsync -av'

    # copy based on the confs
    for exp_name in exp_confs:
        exp_name = exp_name.replace('.yaml','')
        
        if exp_name not in ['default','tstng_exp_conf','tstng_conf','trng_parms_to_vary']:
            print('to be copied:',exp_name)
            command += remote_log_path+exp_name

    # copy base d ont he confs
    # for exp_name in [
    #                  'sb_pint_flat',
    #                  'sb_alob_expdecline',
    #                  'sb_alob_expdecline_sine',
    #                 ]:
    #     # exp_name = exp_name.replace('.yaml','')
        
    #     if exp_name != 'default' and ('tstng' not in exp_name):
    #         command += remote_log_path+exp_name





    # for exp_no in [
    #                     '111','112','113',
    #                     '121','122','123',
    #                     '211','212','213',
    #                     '221','222','223',
                        
    #                  ]:
    #     exp_name = 'skate_wheels_'+exp_no#exp_name.replace('.yaml','')
        
    #     if exp_name != 'default' and ('tstng' not in exp_name):
    #         command += remote_log_path+exp_name    

    #     exp_name = 'skate_board_'+exp_no#exp_name.replace('.yaml','')
        
    #     if exp_name != 'default' and ('tstng' not in exp_name):
    #         command += remote_log_path+exp_name 

    command += ' ./logs/'
    print(command)

    os.system(command)

