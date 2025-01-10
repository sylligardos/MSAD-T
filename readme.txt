To run jupyter notebook on the server:

1. Connect interactively to a node:
salloc -c 8 --gres=gpu:v100:1 --hint=multithread -p gpu

2. Activate conda environment and run browserless notebook
jupyter-notebook --no-browser --port 8001 --ip $(hostname)

3. Create tunnel from PC, gpu001 should match current machine:
ssh -N -L 8001:gpu001:8001 esylliga@cleps

4. Open the link with the localhost and the token produced by the server, e.g.:
http://127.0.0.1:8001/tree?token=8728660a088188ed7ce18dca507b667b48b76dc977e5bc29
