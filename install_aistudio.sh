#!/bin/bash
paddle_dev="paddlepaddle_gpu-2.1.0.dev0.post101-cp37-cp37m-linux_x86_64.whl"
pip install /home/aistudio/paddorch
pip install dgl-cu101==0.4.3 
pip install pytest
pip install nnabla==1.18.0
if [ ! -f "$paddle_dev" ]; then
  wget https://paddle-wheel.bj.bcebos.com/develop-gpu-cuda10.1-cudnn7-mkl_gcc8.2/paddlepaddle_gpu-2.1.0.dev0.post101-cp37-cp37m-linux_x86_64.whl
fi
pip install paddlepaddle_gpu-2.1.0.dev0.post101-cp37-cp37m-linux_x86_64.whl

#由于GPU环境的问题，RDkit的安装如下完成
pip install rdkit-pypi
#source activate python35-paddle120-env
#cp -r -f /home/aistudio/dgl/pkgs/* /opt/conda/pkgs/
#count=0
#cat /home/aistudio/dgl/urls.txt | while read LINE; do
#echo $LINE
#conda install --use-local /opt/conda/pkgs/$LINE
#count=$(($count+1))
#echo "("$count'/46) finshed'
#done

#删除原来的backend folder
rm -rf /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/backend
rm -rf /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/nn
rm -rf /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/model_zoo
#复制我们的backend 目录过去
cp -r dgl/python/dgl/backend   /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/
cp -r dgl/python/dgl/nn   /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/
cp -r dgl/python/dgl/model_zoo   /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/
##copy library to home folder
cp /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/libdgl.so ~/
## 修改默认的backend 为paddorch
mkdir ~/.dgl
echo "{\"backend\":\"paddorch\"}" > ~/.dgl/config.json
pytest /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/dgl/backend/paddorch_unittest.py