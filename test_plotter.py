

import wandb
import numpy as np


wandb.init(
    # Set the project where this run will be logged
    project="test", 
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    # name=f"experiment_{run}", 
    # Track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,
    # }
    
    )

# wandb.log({"my_custom_id" : wandb.plot.line_series(
#                        xs=[0, 1, 2, 3, 4], 
#                        ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
#                        keys=["metric Y", "metric Z"],
#                        title="Two Random Metrics",
#                        xname="x units")})

for i in range(10):
    rand = np.random.random()
    wandb.log({'sub1/metric_1':rand,'sub2/metric_2':1-rand})


wandb.finish()
xyz = 0

print(f'{xyz=}'.split('=')[0])