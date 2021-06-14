# Viewpoint adaptation in a synthesized environment

This project is based on KP2D and Hypersim. For general information regarding(e.g. environment building, dataset downloading), please refer to their official repositories:
[**[KP2D]**](https://github.com/TRI-ML/KP2D)
[**[Hypersim]**](https://github.com/apple/ml-hypersim)

## Dataset configuration

1. Download necessary csvs for Hypersim usage [**[here]**](https://drive.google.com/file/d/1lazqnI10m64tHbqdo1cgkQ2_MGQ_z8d_/view?usp=sharing).

2. Unzip and put ```true_frames_2_0.2_val.csv```, ```true_frames_2_0.2_test.csv```, ```true_frames.csv``` and ```true_frames_8_2.0_train.csv``` under the same directory as Hypersim data.

3. Config dataset path in ```kp2d/configs/base_config.py```.

## Train
```
python train_keypoint_net.py --wandb_name RUN_NAME --file kp2d/configs/v4.yaml --training_mode MODE --pretrained_model PATH/TO/PRETRAINED/MODEL

positional arguments:
  wandb_name            Wandb run name, will overwrite the one in config file.
  file                  Basic KP2D pipeline setup file.
  training_mode         How to use hypersim transformations. Can be:
                                                                    HA: use Homography Adaptation and treat images as without correspondence.
                                                                    HA_wo_sp: same as above but additionally without spatial augmentation.
                                                                    scene: generate frame pairs by choosing every two frame of the same scene.
                                                                    scene+HA: above combined with Homography Adaptation.
                                                                    cam: generate frame pairs by choosing every two frame of the same camera trajectory.
                                                                    cam+HA: above combined with Homography Adaptation.
                                                                    con: generate frame pairs by choosing every consecutive frame pairs within a camera trajectory.
                                                                    con+HA: above combined with Homography Adaptation.
  pretrained_model      Path to a pretrained model.
  non_spatial_aug       Include to activate non spatial pre-processing augmentation.
  partition             Which partition of Hypersim to use. Can be val+test, weakly_filtered or strongly_filtered.
  interval              Interval between consecutive frames. Only takes effect if training_mode is con or con+HA.
```

To reproduce the experiments shown in the paper:

baseline<sub>w</sub>:

```
python train_keypoint_net.py --wandb_name baseline_w --file kp2d/configs/v4.yaml --training_mode HA --non_spatial_aug --pretrained_model /PATH/to/pretrained_models/v4.ckpt --partition weakly_filtered
```

F2F<sub>w</sub>:

```
python train_keypoint_net.py --wandb_name F2F_w --interval 1 --file kp2d/configs/v4.yaml --training_mode con --pretrained_model /PATH/to/pretrained_models/v4.ckpt --partition weakly_filtered
```

baseline<sub>s</sub>:

```
python train_keypoint_net.py --wandb_name baseline_s --file kp2d/configs/v4.yaml --training_mode HA --non_spatial_aug --pretrained_model /PATH/to/pretrained_models/v4.ckpt --partition strongly_filtered
```

F2F<sub>s</sub>:

```
python train_keypoint_net.py --wandb_name F2F_s --interval 1 --file kp2d/configs/v4.yaml --training_mode con --pretrained_model /PATH/to/pretrained_models/v4.ckpt --partition strongly_filtered
```

## Evaluation
### On HPatches
```
python eval_keypoint_net.py --pretrained_model /PATH/TO/model.ckpt --input /PATH/TO/HPatches/
```
### ON Hypersim
```
python eval_keypoint_net_hypersim.py --pretrained_model /PATH/TO/model.ckpt --input /PATH/TO/HPatches/
```