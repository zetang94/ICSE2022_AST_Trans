# ast_trans_icse
This is the implementation for AST-Trans: Code Summarization with Efficient Tree-Structured Attention.
## 1. install environments
   pip install -r requirements.
   
   pytorch-geometric can be installed by
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
## 2.pre process data
this step attends to generate pot, sbt, paths and relationship matrices of ASTs.
The [dataset](https://drive.google.com/drive/folders/1dc42GFE6nx4x5_2_6H-qelTSI1KmPEJe) we proposed have been pre-processed. If you want to pre-process the dataset of your own, you can save ASTs as a json object of format like {'type':'';'value':'','children':[]}, and then run:

python pre_process/process.py --data_dir your_data_set_path --max_ast_len 250 --process --make_vocab

> PS: Due to the limited storage of the cloud drive, I accidentally deleted the data and code. Fortunately, in the work of [1, 2], they use tree-sitter to process and extract the data required for ast-trans. That is really cool! So if you are interested in the pre-process process, you can find the detail in their [repositor1](https://github.com/saeyoon17/Code-Structure-Aware-Transformer), [repository2](https://github.com/wssun/AST4PLU/tree/main).
>
>[1] Oh, Saeyoon, and Shin Yoo. "CSA-Trans: Code Structure Aware Transformer for AST." arXiv preprint arXiv:2404.05767 (2024).
> 
>[2] Sun W, Fang C, Miao Y, et al. Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?[J]. arXiv preprint arXiv:2312.00413, 2023.

## 3. run
we use py_config_runner.
The config of model is saved at ./config.
you need to change 'data_dir' to your own data set path.
clearml is also supported in our implementation.
if you do not want to use it, just set:

use_clearml = False

For run AST-Trans:

single-gpu:

python main.py --config ./config/ast_trans.py --g 0

multi-gpu:

python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --config=./config/ast_trans.py --g 0,1

## 4. test
For test, you can set the parameter 'is_test' to True.
and set the checkpoint file path.
The program will default find the checkpoint based on the hype-parameters.
And we apply a trained AST-Trans checkpoint file for python in https://box.nju.edu.cn/f/ebdb250c46524268be41/.
If you want to load model on your specific path, just change the 'load_epoch_path' of the test function in script/train.py 

load_epoch_path = './checkpoint/'

