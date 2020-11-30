#stage1 test

from typing import Any, Union
import os, sys

config = configparser.RawConfigParser()
config.read('configuration.txt')
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup') 

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

result_dir = name_experiment
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

if nohup:
    print ("\n2. Run the prediction on GPU  with nohup")
    os.system(run_GPU +' nohup python -u ./predict.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
    print ("\n2. Run the prediction on GPU (no nohup)")
    os.system(run_GPU +' python ./predict.py')
