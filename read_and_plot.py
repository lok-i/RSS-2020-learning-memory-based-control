import numpy as np
import os
import matplotlib.pyplot as plt
from cassie.trajectory import CassieTrajectory



plot_name = 'perf_metrics'

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
    log_path = "./logs/recur_rand/"

    data = np.load(log_path+'vx_d_exps'+'.npz',allow_pickle=True)

    [
        'vxs_des',
        'vxs_0',
        'dist_travelled',
        'ep_returns',
    ]

    print(data['ep_returns'].shape)



    for metric_key in [
                        'dist_travelled',
                        # 'ep_returns'
                        ]:

        # metric_key = 'ep_returns'

        metric_mean = np.mean(data[metric_key],axis=1)
        metric_std = np.std(data[metric_key],axis=1)


        plt.fill_between(
                            data['vxs_des'], 
                            metric_mean  - metric_std,
                            metric_mean  + metric_std, 
                            alpha=0.5,
                            # color=all_terrains[terrain_type]['color']
                            )

        plt.plot(
                    data['vxs_des'], 
                    metric_mean, 
                    '--', 
                    # color=all_terrains[terrain_type]['color'],
                    label= metric_key
                    )
    plt.vlines(
                    x=[-0.15,0.8],
                    ymin=data[metric_key].min(),
                    ymax=data[metric_key].max(),
                    linestyles='dashed',
                    colors='red'
                )
    plt.text(-0.15, data[metric_key].min(), '-0.15')
    plt.text(0.8, data[metric_key].min(), '0.8')

    plt.grid()
    plt.legend()
    plt.show()
    # print(dis_mean,dis_std)

    # max_dis_mean = np.amax(dis_mean)


    # for key in data.keys():
    #     print(key)
