# **_Dodekathlon_**


![GitHub](https://img.shields.io/github/license/mtroym/Dodekathlon?logo=apache)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/pytorch/pytorch?logo=pytorch)
![GitHub top language](https://img.shields.io/github/languages/top/mtroym/Dodekathlon)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/94fc5e9d75354b48b2ff09cc0d9527aa)](https://www.codacy.com/manual/mtroym/Dodekathlon?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mtroym/Dodekathlon&amp;utm_campaign=Badge_Grade)
![GitHub code size in byte![GitHub top language](https://img.shields.io/github/languages/top/mtroym/Dodekathlon)s](https://img.shields.io/github/languages/code-size/mtroym/dodekathlon?)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/mtroym/Dodekathlon)
![GitHub last commit](https://img.shields.io/github/last-commit/mtroym/Dodekathlon)
![Liberapay receiving](https://img.shields.io/liberapay/receives/troymao)
[![Donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/troymao/donate)


## Updates: 

- 2019-11-28 Hydra version under construction.
- 2019-11-12 Name of [my projects](https://mtroym.github.io/) will be changed to the Myths from Ancient Greek.


## Hydra Checklist.
![PATN.png](https://i.loli.net/2019/12/05/qO8u56aR73mWGvD.png)
- [x] Checkpoints support.
- [x] kp dataset/ kpparse dataset
- [x] model for PATN
- [ ] training
- [ ] losses
- [ ] evaluations.

## Canary Plans.

- [ ] Example of PATN and Controllable Human Pose Transfer `branch:hydra`
- [ ] Distributed training process.
- [ ] Visualization with `TensorBoard` embedded.
- [ ] Deploying using `oonx` and `TensorRT`


## Citation
if you find this code benefits your research, please take the citation at ease:


```
@inproceedings{zhu2019progressive,
  title={Progressive Pose Attention Transfer for Person Image Generation},
  author={Zhu, Zhen and Huang, Tengteng and Shi, Baoguang and Yu, Miao and Wang, Bofei and Bai, Xiang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2347--2356},
  year={2019}
}
```

## Structure of `Dodekathlon`

```shell
├── configures
│   ├── train.yaml // The configures. (Trail Level)
│   └── ....    // store the yaml configure.
├── criterions  // define the criterions train and test.
│   ├── loss    // define the loss.
│   └── metrics // define the metrics for evaluation.
├── datasets    // define the dataset. dataloader.
├── deployments // deploy to the cloud using oonx.
├── main.py     // great dodekathlon main file.
├── models      // define the model here.
├── options     // define the options. DIFFS From configures.(Task level)
├── requirements.txt // try 'pip install -Ur requirements.txt'
├── tools       // some useful tools(object oriented).
└── utils       // some useful functions.
```


## Reference of Hercules `Dodekathlon`

Before I chose this name "Dodekathlon", I supposed to use Hercules, who was considered as
a demi-god with strong labors. "Dodekathlon", considered as 12 Labors of Hercules, are 
how Hercules gained much of his mythological fame as a demi-god, listing as follows:

![Alt Twelve_Labours_Altemps](https://s2.ax1x.com/2019/12/06/QGoZ5t.md.jpg) (picture credit to wikipedia)

- Slay the `Nemean` Lion.
- Slay the nine-headed `Lernaean Hydra`.
- Capture the Golden Hind of `Artemis`.
- Capture the `Erymanthian Boar`.
- Clean the `Augean stables` in a single day.
- Slay the `Stymphalian` Birds.
- Capture the `Cretan Bull`.
- Steal the Mares of `Diomedes`.
- Obtain the girdle of `Hippolyta`, Queen of the Amazons.
- Obtain the cattle of the monster `Geryon`.
- Steal the apples of the `Hesperides`
- Capture and bring back `Cerberus`.

As PyTorch is from Torch, whose meaning is a sticker with flames and fires. 
This project combines Training, Testing, Deploying, Visualization, Evaluation, Checkpoints
and Resuming, etc., each of them is challenging and in great modularization. 
Therefore, I chose `Dodekathlon`. 