# Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering

PyTorch implementation of the paper "Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering".


# Get started.

```
cd DyFSS
python train.py --use_ckpt=0 --dataset='cora' --pretrain_epochs=250 --w_ssl_stage_one=0.25 --st_per_p=0.5 --lr_train_fusion=0.001 --labels_epochs=250 --w_ssl_stage_two=0.1
```

# Citation

```
@inproceedings{dyfss,
  author       = {Pengfei Zhu and
                  Qian Wang and
                  Yu Wang and
                  Jialu Li and
                  Qinghua Hu},
  title        = {Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering},
  publisher    = {{AAAI} Press},
  year         = {2024},
}
```
