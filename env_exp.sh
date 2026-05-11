export CAP4D_PATH=/data1/zjh/cap4d_exp_backhead
export PYTHONPATH=$CAP4D_PATH:$PYTHONPATH
export PIXEL3DMM_PATH=/data1/zjh/pixel3dmm
export CUDA_VISIBLE_DEVICES=0,1

export CUDA_HOME=/data1/zjh/cap4d/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
