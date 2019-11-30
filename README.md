# `Dodekathlon`
![GitHub](https://img.shields.io/github/license/mtroym/Dodekathlon?logo=apache)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/pytorch/pytorch?logo=pytorch)
![GitHub top language](https://img.shields.io/github/languages/top/mtroym/Dodekathlon)
![GitHub code size in byte![GitHub top language](https://img.shields.io/github/languages/top/mtroym/Dodekathlon)s](https://img.shields.io/github/languages/code-size/mtroym/dodekathlon?)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/mtroym/Dodekathlon)
![GitHub last commit](https://img.shields.io/github/last-commit/mtroym/Dodekathlon)
![Liberapay receiving](https://img.shields.io/liberapay/receives/troymao)
[![Donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/troymao/donate)
## Updates: 
- 2019-11-28 `Hydra` version under construction.
- 2019-11-12 Name of [my projects](https://mtroym.github.io/) will be changed to the Myths from Ancient Greek.

### `Hydra` Checklist.
- [x] Checkpoints support.
- [x] kp dataset/ kpparse dataset
- [ ] models
- [ ] training
- [ ] losses
- [ ] evaluations.
- [ ] Example of PATN and Controllable Human Pose Transfer

## Reference of `Dodekathlon`
Before I chose this name `"Dodekathlon"`, I supposed to use Hercules, who was considered as
a demi-god with strong labors. "Dodekathlon", considered as 12 Labors of Hercules, are 
how Hercules gained much of his mythological fame as a demi-god, listing as follows:

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

## Structure of `Dodekathlon`

### Tree.
```
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
├── requirements.txt // try 'pip install -U requirements.txt'
├── tools       // some useful tools(object oriented).
└── utils       // some useful functions.
```


### `Canary` Plans.
- [x] Checkpoints support.
- [ ] Example of PATN and Controllable Human Pose Transfer `branch:hydra`
- [ ] Distributed training process.
- [ ] Visualization with `TensorBoard` embedded.
- [ ] Deploying using `oonx` and `TensorRT`

