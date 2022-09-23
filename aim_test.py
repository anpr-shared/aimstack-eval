from aim import Run
from time import sleep

# Initialize a new run
run = Run() # repo="https://35.200.171.144:53800"

# Log run parameters
run["hparams"] = {
    "learning_rate": 0.001,
    "batch_size": 32,
}

# Log metrics
for i in range(1, 200):
    sleep(1)
    run.track(i, name="loss", step=i, context={"subset": "train"})
    run.track(i, name="acc", step=i, context={"subset": "train"})
