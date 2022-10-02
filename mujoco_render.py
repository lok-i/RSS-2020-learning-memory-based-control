
# external dm_control-python based renderer for viewing
# policy rollouts i.e. replaying collected by qpos trajectotries 

from dm_control import mujoco
import mujoco_viewer
import numpy as np

model_path = "./cassie/cassiemujoco/cassie.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)


log_path = "./logs/recur_rand/"
qpos_trajs = np.load(log_path+"five_epi_eval.npy",allow_pickle=True)

print("log_path:",log_path)

viewer._paused = True
viewer.cam.distance = 3
cam_pos = [0.0, 0.0, 0.75]

for i in range(3):        
    viewer.cam.lookat[i]= cam_pos[i] 
viewer.cam.elevation = -15
viewer.cam.azimuth = 180


for qpos_traj in qpos_trajs:
    mujoco.mj_resetData(model,data)
    for qpos in qpos_traj:

        data.qpos[:] = qpos[:]
        mujoco.mj_step(model,data)
        viewer.render()  
