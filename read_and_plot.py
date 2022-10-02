import numpy as np
import os
import matplotlib.pyplot as plt
from cassie.trajectory import CassieTrajectory
import yaml


plot_name = 'perf_metrics2'

if plot_name == 'ref_traj_qpos':
    traj_path = os.path.join("./cassie/trajectory", "stepdata.bin")
    trajectory = CassieTrajectory(traj_path)


    for qpos_id in [
                    0,1,2,
                    # 3,4,5,6,
                    7, 8, 9, 
                    # 14, 
                    # 20, 
                    21, 22, 23, 
                    # 28, 
                    # 34
                    ]:
        # qpos_id = 8
        data = trajectory.qpos[:,qpos_id]
        plt.plot(np.arange(data.shape[0]),data,label=str(qpos_id))
    plt.legend()
    plt.show()

elif plot_name == 'ref_traj_qvel':
    traj_path = os.path.join("./cassie/trajectory", "stepdata.bin")
    trajectory = CassieTrajectory(traj_path)

    for qvel_id in [    0,1,2,
                        # 3,4,5,
                        # 6, 7, 8, 
                        # 12, 
                        # 18, 
                        # 19, 20, 21, 
                        # 25, 
                        # 31
                    ]:
        # qpos_id = 8
        data = trajectory.qvel[:,qvel_id]
        plt.plot(np.arange(data.shape[0]),data,label=str(qvel_id))
    plt.legend()
    plt.grid()
    plt.show()

elif plot_name == 'perf_metrics':
    
    
    metric_2_labels = { 
                        'dist_travelled':{
                                            'x':'Command Heading Velocity (m/s)',
                                            'y':'Distance Travelled (m)',
                                            'ylim':[-7.5,20],
                                            'xlim':[-4.5,4.5],                                        
                                        
                                        
                                        },
                        # 'ep_returns':{
                        #                 'x':'Command Heading Velocity (m/s)',
                        #                 'y':' Avg.Returns',
                        #                 'ylim':[0,300],
                        #                 'xlim':[-4.5,4.5],

                        #                 },
                        # 'e_vx_norm_avg':{
                        #                     'x':'Command Heading Velocity (m/s)',
                        #                     'y':'Norm. Avg. Heading Velocity error (m/s)',
                        #                     'ylim':[0,1],
                        #                     'xlim':[-4.5,4.5],
                        #                 },

                        }
    
    experiments_to_plot = [
                            'recur_rand',
                            'vx_d_sym1',
                            'vx_d_sym2',
                            'vx_d_sym3',
                            # 'vx_d_skew1',
                            # 'vx_d_skew2',
                            # 'vx_d_skew3',
                          ]
    exp_plot_colours = ['b','g','r','c','y','m','brown']
    plt.grid()    
    for exp_id,exp_name in enumerate(experiments_to_plot):
        log_path = "./logs/"+ exp_name +"/"

        data = np.load(log_path+'vx_d_exps'+'.npz',allow_pickle=True)

        for metric_key in metric_2_labels.keys():

            metric_mean = np.mean(data[metric_key],axis=1)
            metric_std = np.std(data[metric_key],axis=1)

            conf_file = open(log_path+exp_name+'.yaml')
            exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            # print('\texperiment configuration {}'.format(exp_conf))   

            if metric_key == 'e_vx_norm_avg':
                metric_max_val = data[metric_key].max()
                print('max. vx error of',exp_name,':',metric_max_val)

                norm_metric_data = (1/metric_max_val)*data[metric_key]
                norm_metric_mean = (1/metric_max_val)*metric_mean
                norm_metric_std = (1/metric_max_val)*metric_std

                # plt.fill_between(
                #                     data['vxs_des'], 
                #                     norm_metric_mean - norm_metric_std,
                #                     norm_metric_mean + norm_metric_std,
                #                     alpha=0.5,
                #                     color=exp_plot_colours[exp_id]
                                    
                #                     )

                plt.plot(
                            data['vxs_des'], 
                            norm_metric_mean, 
                            '-', 
                            color=exp_plot_colours[exp_id],
                            label= exp_name+",max:"+str(metric_max_val) if exp_name != 'recur_rand' else exp_name+"(baseline),max:"+str(metric_max_val)

                            )


                # training limits
                plt.text(exp_conf['vx_d_min'], norm_metric_data.min(), str(exp_conf['vx_d_min']))
                plt.text(exp_conf['vx_d_max'], norm_metric_data.min(),str(exp_conf['vx_d_max']))
                plt.vlines(
                                x=[exp_conf['vx_d_min'],exp_conf['vx_d_max']],
                                ymin=norm_metric_data.min(),
                                ymax=norm_metric_data.max(),
                                linestyles='dashed',
                                color=exp_plot_colours[exp_id],
                                linewidth=2
                            )

            else:
                plt.fill_between(
                                    data['vxs_des'], 
                                    metric_mean  - metric_std,
                                    metric_mean  + metric_std, 
                                    alpha=0.5,
                                    color=exp_plot_colours[exp_id]
                                    
                                    )

                plt.plot(
                            data['vxs_des'], 
                            metric_mean, 
                            '-', 
                            color=exp_plot_colours[exp_id],
                            label= exp_name if exp_name != 'recur_rand' else exp_name+' (baseline)'

                            )
            
                # training limits
                plt.text(exp_conf['vx_d_min'], data[metric_key].min(), str(exp_conf['vx_d_min']))
                plt.text(exp_conf['vx_d_max'], data[metric_key].min(),str(exp_conf['vx_d_max']))
                plt.vlines(
                                x=[exp_conf['vx_d_min'],exp_conf['vx_d_max']],
                                ymin=data[metric_key].min(),
                                ymax=data[metric_key].max(),
                                linestyles='dashed',
                                color=exp_plot_colours[exp_id],
                                linewidth=2
                            )
            plt.ylim(metric_2_labels[metric_key]['ylim'][0],metric_2_labels[metric_key]['ylim'][1],)
            plt.xlim(metric_2_labels[metric_key]['xlim'][0],metric_2_labels[metric_key]['xlim'][1],)

            # labels

            plt.xlabel(metric_2_labels[metric_key]['x'])
            plt.ylabel(metric_2_labels[metric_key]['y'])
            
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig('./'+metric_key+'_'+str(exp_id))




    plt.show()


elif plot_name == 'perf_metrics2':
    
    
    metric_2_labels = { 
                        'dist_travelled':{
                                            'x':'Command Heading Velocity (m/s)',
                                            'y':'Distance Travelled (m)',
                                            'ylim':[-7.5,20],
                                            'xlim':[-4.5,4.5],                                        
                                        
                                        
                                        },
                        'ep_returns':{
                                        'x':'Command Heading Velocity (m/s)',
                                        'y':' Avg.Returns',
                                        'ylim':[0,300],
                                        'xlim':[-4.5,4.5],

                                        },
                        'e_vx_norm_avg':{
                                            'x':'Command Heading Velocity (m/s)',
                                            'y':'Norm. Avg. Heading Velocity error (m/s)',
                                            'ylim':[0,1],
                                            'xlim':[-4.5,4.5],
                                        },

                        }
    
    experiments_to_plot = [
                            'recur_rand',
                            'vx_d_sym1',
                            # 'vx_d_sym2',
                            # 'vx_d_sym3',
                            'vx_d_skew2',
                            'vx_d_skew3',
                            'vx_d_skew1',

                          ]
    exp_plot_colours = [
                            'b',
                            'g',
                            'r',
                            'c',
                            'y',
                            'm',
                            'brown'
                            ]
    


    fig,axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
   

    for exp_id,exp_name in enumerate(experiments_to_plot):
        log_path = "./logs/"+ exp_name +"/"

        data = np.load(log_path+'vx_d_exps'+'.npz',allow_pickle=True)

        for metric_id, metric_key in enumerate(metric_2_labels.keys()):

            metric_mean = np.mean(data[metric_key],axis=1)
            metric_std = np.std(data[metric_key],axis=1)

            conf_file = open(log_path+exp_name+'.yaml')
            exp_conf = yaml.load(conf_file, Loader=yaml.FullLoader)
            # print('\texperiment configuration {}'.format(exp_conf))   

            if metric_key == 'e_vx_norm_avg':
                metric_max_val = data[metric_key].max()
                print('max. vx error of',exp_name,':',metric_max_val)

                norm_metric_data = (1/metric_max_val)*data[metric_key]
                norm_metric_mean = (1/metric_max_val)*metric_mean
                norm_metric_std = (1/metric_max_val)*metric_std

                # axs[metric_id].fill_between(
                #                     data['vxs_des'], 
                #                     norm_metric_mean - norm_metric_std,
                #                     norm_metric_mean + norm_metric_std,
                #                     alpha=0.5,
                #                     color=exp_plot_colours[exp_id]
                                    
                #                     )

                axs[metric_id].plot(
                            data['vxs_des'], 
                            norm_metric_mean, 
                            '-', 
                            color=exp_plot_colours[exp_id],
                            label= exp_name+",max:"+str(metric_max_val) if exp_name != 'recur_rand' else exp_name+"(baseline),max:"+str(metric_max_val)

                            )


                # training limits
                axs[metric_id].text(exp_conf['vx_d_min'], norm_metric_data.min(), str(exp_conf['vx_d_min']))
                axs[metric_id].text(exp_conf['vx_d_max'], norm_metric_data.min(),str(exp_conf['vx_d_max']))
                axs[metric_id].vlines(
                                x=[exp_conf['vx_d_min'],exp_conf['vx_d_max']],
                                ymin=norm_metric_data.min(),
                                ymax=norm_metric_data.max(),
                                linestyles='dashed',
                                color=exp_plot_colours[exp_id],
                                linewidth=2
                            )

            else:
                axs[metric_id].fill_between(
                                    data['vxs_des'], 
                                    metric_mean  - metric_std,
                                    metric_mean  + metric_std, 
                                    alpha=0.5,
                                    color=exp_plot_colours[exp_id]
                                    
                                    )

                axs[metric_id].plot(
                            data['vxs_des'], 
                            metric_mean, 
                            '-', 
                            color=exp_plot_colours[exp_id],
                            label= exp_name if exp_name != 'recur_rand' else exp_name+' (baseline)'

                            )
            
                # training limits
                axs[metric_id].text(exp_conf['vx_d_min'], data[metric_key].min(), str(exp_conf['vx_d_min']))
                axs[metric_id].text(exp_conf['vx_d_max'], data[metric_key].min(),str(exp_conf['vx_d_max']))
                axs[metric_id].vlines(
                                x=[exp_conf['vx_d_min'],exp_conf['vx_d_max']],
                                ymin=data[metric_key].min(),
                                ymax=data[metric_key].max(),
                                linestyles='dashed',
                                color=exp_plot_colours[exp_id],
                                linewidth=2
                            )
            axs[metric_id].set_ylim(metric_2_labels[metric_key]['ylim'][0],metric_2_labels[metric_key]['ylim'][1],)
            axs[metric_id].set_xlim(metric_2_labels[metric_key]['xlim'][0],metric_2_labels[metric_key]['xlim'][1],)

            # labels

            axs[metric_id].set_xlabel(metric_2_labels[metric_key]['x'])
            axs[metric_id].set_ylabel(metric_2_labels[metric_key]['y'])
            
            if exp_id == 0:
                axs[metric_id].grid()
            axs[metric_id].legend(loc='upper left')
            plt.tight_layout()
            if metric_id == (len(metric_2_labels.keys())-1):
                plt.savefig('./'+metric_key+'_'+str(exp_id))




    plt.show()