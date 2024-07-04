# MolAE: Auto-Encoder Based Molecular Representation Learning With 3D Cloze Test Objective

Official implementation of paper: MolAE: Auto-Encoder Based Molecular Representation Learning With 3D Cloze Test Objective (ICML 2024) [[paper](https://openreview.net/forum?id=inEuvSg0y1)]


The Mol-AE code is primarily built upon the UniMol codebase. Therefore, for setting up the relevant environment, please refer to the [[UniMol](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol)] documentation. We highly recommend users, after successfully configuring UniMol, to directly replace the `unimol/` folder in the UniMol repository with the `unimol/` folder from our repository. This way, you will be able to use both the UniMol and Mol-AE models.

## Pre-train

There are two ways to use Mol-AE: you can either directly use our pre-trained weights, like picking up a ready-made tool, or you can pre-train Mol-AE from scratch, akin to crafting your own tool from raw materials.

### 1.Directly use the pre-trained model.
The pre-trained checkpoint can be downloaded from [[google drive](https://drive.google.com/file/d/1NKObZCfE80GCLS9yJ7hqMGzjfGol4LLo/view?usp=drive_link)].

### 2. Pre-train Mol-AE from scratch.

We used the exact same pre-training data as UniMol. After downloading the pre-train [[data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/pretrain/ligands.tar.gz)], you can use the following script to pre-train Mol-AE.

```
data_path= # replace to your data path
save_dir= # replace to your save path
logfile=${save_dir}/train.log
n_gpu=1
MASTER_PORT=$RANDOM
lr=1e-4
wd=1e-4
batch_size=128
update_freq=1
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

mkdir -p ${save_dir}
cp $0 ${save_dir}

export CUDA_VISIBLE_DEVICES=2
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task unimol --loss unimol_MAE --arch unimol_MAE_padding  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 100 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --decoder-x-norm-loss $x_norm_loss --decoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --encoder-x-norm-loss $x_norm_loss --encoder-delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --encoder-unmasked-tokens-only \
       --decoder-layers 5 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 64 \
       --save-dir $save_dir  --only-polar $only_polar > ${logfile} 2>&1
```


## Downstream Tasks

We used the exact same downstream tasks as UniMol, which mainly include two types of tasks: classification and regression. The downstream data can be downloaded from       [[data](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz)]. Please use following script to finetune Mol-AE:

```
data_path= # replace to your data path
save_dir=  # replace to your save path
n_gpu=4
MASTER_PORT=10086
dict_name="dict.txt"
weight_path= # replace to your ckpt path
task_name="qm9dft"  # molecular property prediction task name 
task_num=3
loss_func="finetune_smooth_mae"
lr=1e-4
batch_size=32
epoch=40
dropout=0
warmup=0.06
local_batch_size=32
only_polar=0
conf_size=11
seed=0

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

rm -rf ${save_dir}
mkdir -p ${save_dir}
mkdir -p ${save_dir}/tmp

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --conf-size $conf_size \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task mol_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 20 \
       --save-dir $save_dir --only-polar $only_polar > ${logfile} 2>&1
```

For the selection of hyperparameters, please refer to the following table:

Classification:

|Dataset      | BBBP | BACE | ClinTox | Tox21 | ToxCast | SIDER | HIV | PCBA | MUV |
|--------|----|----|----|----|----|-----|-----|----|-----|       
| task_num |  2 | 2 | 2 | 12 | 617 | 27 | 2 | 128 | 17 |
| lr         |  4e-4 | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 5e-4 | 5e-5 | 1e-4 | 2e-5 |
| batch_size |  128 | 64 | 256 | 128 | 64 | 32 | 256 | 128 | 128 |
| epoch      |  40 | 20 | 80 | 80 | 160 | 40 | 5 | 20 | 20 |
| pooler-dropout    |  0.1 | 0.2 | 0.7 | 0.1 | 0.2 | 0 | 0.2 | 0.1 | 0.1 |
| dropout    |  0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| warmup     |  0.18 | 0.36 | 0.25 | 0.06 | 0.06 | 0.5 | 0.1 | 0.06 | 0.3 |

Regression:

| Dataset | ESOL | FreeSolv | Lipo | QM7 | QM8 | QM9 |
|----- | ---- | ---- | ---- | ---- | --- | --- |
| task_num | 1 | 1 |  1 | 1  | 12 | 3 |
| lr         | 5e-4 | 8e-5 |  1e-4 | 3e-4  | 1e-4 | 1e-4 |
| batch_size | 256 | 64 |  32 | 32  | 32 | 128 |
| epoch      | 200 | 160 |  100 | 200  | 120 | 40 |
| pooler-dropout    | 0.4 | 0.4 |  0.1 | 0.1  | 0 | 0 |
| dropout    | 0.1 | 0.1 |  0.1 | 0.1  | 0 | 0.1 |
| warmup     | 0.06 | 0.1 | 0.24 | 0.06  | 0.02 | 0.06 |


TODO
--------
- [ ] More detailed README.

Citation
--------

Please kindly cite this paper if you use the data/code/model.
```
@article{yang2024mol,
  title={MOL-AE: Auto-Encoder Based Molecular Representation Learning With 3D Cloze Test Objective},
  author={Yang, Junwei and Zheng, Kangjie and Long, Siyu and Nie, Zaiqing and Zhang, Ming and Dai, Xinyu and Ma, Wei-Ying and Zhou, Hao},
  journal={bioRxiv},
  pages={2024--04},
  year={2024},
}
```

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/yjwtheonly/MolAE/blob/master/LICENSE) for additional details.