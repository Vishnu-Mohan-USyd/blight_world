# Current shape classification and motion network

Updated 22 October 2026, Australia/Sydney.

The current network learns object category and object velocity together from scratch. Each input is one continuous 48-frame stereo recording of a moving shape while the camera sweeps through 60 degrees. The images are 128 × 128 pixels per eye with a coarse wall grid. The object pathway has a 128-unit vector RNN; the motion pathway branches after the third convolution and has a 128-channel spatial RNN.

The current dataset has **12 classes and 7,200 generated objects**, with one recording per object. The selected model classifies **1,071/1,200 TEST videos correctly, or 89.25%**, with **7.03° mean direction error** and **0.96 cm/s mean speed error** relative to the room. These results come from the seed-0 run selected at epoch 71 of 128.

The expansion doubles the shape families from six to twelve and the examples per class from 200/50/50 to 400/100/100 for TRAIN/VAL/TEST. Recording settings and training settings remain unchanged. The classification output increases from six to twelve logits; both pathways are trained together from scratch.

## Recorded results

The current run uses twelve classes; the three historical runs use the earlier six-class dataset. All use 128-pixel images, the coarse wall grid and seed 0, with training from scratch.

| Recorded run | Selected epoch | TEST classification | Direction MAE | Speed MAE | Velocity vector MAE |
|---|---:|---:|---:|---:|---:|
| [Current: 12 classes, joint motion 128](results/joint_motion128_12classes_seed0/test_report.json) | 71 | 1,071/1,200, 89.25% | 7.0306° | 0.9633 cm/s | 1.4434 cm/s |
| [Historical: 6 classes, classification only](results/classification_only_seed0/test_report.json) | 87 | 281/300, 93.67% | — | — | — |
| [Historical: 6 classes, joint motion 64](results/joint_motion64_seed0/test_report.json) | 93 | 281/300, 93.67% | 8.6870° | 1.1320 cm/s | 1.7229 cm/s |
| [Historical: 6 classes, joint motion 128](results/joint_motion128_seed0/test_report.json) | 94 | 284/300, 94.67% | 7.9971° | 1.0290 cm/s | 1.5791 cm/s |

Within the historical six-class dataset, the 64-to-128 comparison changes only the motion recurrent width. It improved classification by three videos, direction error by 0.69°, and speed error by 0.103 cm/s in this run. The classification-only baseline uses a different checkpoint criterion, described below. These are individual seed-0 results.

At the selected current checkpoint:

| Split | Classification | Direction MAE | Speed MAE |
|---|---:|---:|---:|
| TRAIN | 4,640/4,800, 96.67% | 6.5383° | 0.9148 cm/s |
| VAL | 1,086/1,200, 90.50% | 7.0736° | 0.9420 cm/s |
| TEST | 1,071/1,200, 89.25% | 7.0306° | 0.9633 cm/s |

Each current VAL/TEST motion result averages 56,400 intervals: 1,200 recordings × 47 intervals. The recorded training duration was 6,023.8 seconds, about 100.4 minutes. The scores above all use the selected epoch-71 checkpoint; later epochs in the training history are not the evaluated TEST model.

The [training report](results/joint_motion128_12classes_seed0/report.json) contains selected-checkpoint TRAIN/VAL results and final-epoch VAL results. The [TEST report](results/joint_motion128_12classes_seed0/test_report.json) contains the separate TEST evaluation. Reported model scores use `best.pt`, selected by validation, rather than `last.pt`.

This repository includes the code, configuration and result reports. Recordings, retinal caches, checkpoints, training histories and preview files are not included. Generate the dataset and train with the commands below, or point the commands at an existing compatible dataset cache. Absolute paths in the archived JSON reports record the original experiment locations; the runnable commands use your own storage paths.

## Directory and code map

The entry point is `networks/simple_shapes` within the repository.

```text
networks/simple_shapes/
  README.md
  requirements.txt             Python dependencies
  code/
    train_joint_motion.py       joint model, training and evaluation
    train_expanded_rnn.py       object model, frame loading and object-only training
    motion_targets.py          interval velocity labels
    render_dataset.py          current 128-pixel coarse-grid renderer
    prepare_retina.py          current 128-pixel retinal preprocessing
    gpu_filters.py             parvo and magno filter implementation
  results/
    joint_motion128_12classes_seed0/ current config, training and TEST reports
    classification_only_seed0/       historical six-class reports
    joint_motion64_seed0/            historical six-class reports
    joint_motion128_seed0/           historical six-class reports
```

| Code | Responsibility |
|---|---|
| [train_joint_motion.py](code/train_joint_motion.py) | `JointClassifier`, motion data loader, joint losses, model selection and physical motion metrics |
| [train_expanded_rnn.py](code/train_expanded_rnn.py) | `SequenceClassifier`, `FrameSource`, object recurrent projection and classification-only command |
| [motion_targets.py](code/motion_targets.py) | Builds room X/Y velocity labels from recorded shape positions |
| [render_dataset.py](code/render_dataset.py) | Meshes, random poses, bounded motion, stereo camera arc, wall grid, RGB arrays and sample videos |
| [prepare_retina.py](code/prepare_retina.py) | Applies retinal filters to all 48 frames and writes the training cache |
| [gpu_filters.py](code/gpu_filters.py) | `GPUParvo` and `GPUMagno`, loaded locally by preprocessing |

The commands below create `data/`, `cache/` and separate run directories under `SHAPES_STORAGE`, which can be on an external drive. A completed training run and its separate TEST evaluation produce:

| File | Contents |
|---|---|
| `config.json` | Recorded architecture, data path and training settings |
| `best.pt` | Validation-selected model state, configuration, epoch and validation metrics |
| `last.pt` | Final-epoch model with the same checkpoint fields |
| `history.json` | Per-epoch training/validation metrics and selection decisions |
| `report.json` | Completed training summary |
| `test_report.json` | Separate TEST evaluation |

Checkpoints store model weights, not optimizer/scheduler state. The training scripts currently have no resume command.

## Dataset and recordings

| Property | Current value |
|---|---|
| Classes in label order | 0 box, 1 sphere, 2 cylinder, 3 cone, 4 pyramid, 5 torus, 6 capsule, 7 ellipsoid, 8 triangular_prism, 9 hexagonal_prism, 10 tetrahedron, 11 truncated_cone |
| TRAIN | 4,800 videos; 400 per class; generation seed 19701 |
| VAL | 1,200 videos; 100 per class; generation seed 19702 |
| TEST | 1,200 videos; 100 per class; generation seed 19703 |
| Recording | 48 consecutive frames at 12 fps; four-second playback |
| Stereo images | 128 × 128 per eye; 4.9 cm eye separation; parallel stereo axes |
| Camera | Constant-speed 60° horizontal arc, radius 1.05 m, fixed look-at point (0, 0, 0.20 m), vertical field of view 60° |
| Arc direction | Independently seeded left-to-right or right-to-left per recording |
| Room | One-metre cube, coordinates −0.5 to +0.5 m; open front |
| Wall grid | 25 cm spacing, 2 cm line width, grid colour 0.44 × wall colour |
| Object appearance | Shared orange colour; randomized shape proportions, size and 3D orientation |
| Object size | Maximum extent after rotation between 32 and 40 cm |
| Object motion plane | Room X/Y, with Y up; Z fixed at 20 cm |
| Object centre bounds | X and Y each remain within ±22 cm; complete paths outside the bounds are rejected |
| Motion changes | New random direction/speed every two seconds; 0.25-second smooth transition |

The four-second recording has two motion periods, with the transition from the first velocity to the second over 1.75–2.0 seconds. Nominal speeds are approximately 5.68, 11.37 and 17.05 cm/s. Speed selection and whole-path rejection mean that accepted recordings need not contain equally many examples of each speed. Smooth transitions can also have intermediate speeds.

Object orientation stays fixed within a recording while the camera changes viewpoint. New orientations, mesh proportions, trajectories and camera directions are generated across recordings and splits. Evaluation therefore measures category recognition on new generated examples of these twelve shape families. The present task has planar object motion and a 60° camera arc.

Frames are rendered with twofold spatial supersampling and downsampled to 128 pixels. World-space trajectories retain the original 64-pixel motion reference, so increasing image resolution did not change physical speeds. The metadata field `reference_speeds_pixels_per_frame` refers to that original fixed-camera reference.

Generated `data/index.json` records labels, generation seeds, mesh parameters, orientations, all 48 world positions and camera angles. Raw arrays have shape `[videos, 48, 2, 3, 128, 128]`, dtype `uint8`. Generation also creates `data/sample_shapes.png` and twelve `sample_<class>.mp4` videos in `data/`, showing one training example per class. Full recordings are stored in the RGB arrays.

## Retinal inputs

Each eye supplies six channels in this order: `p_l_on`, `p_l_off`, `p_m_on`, `p_m_off`, `m_on`, `m_off`. Left-eye channels precede right-eye channels, giving 12 inputs per frame.

`GPUParvo` computes four L/M centre-surround channels, with centre side 2 and surround width 5. `GPUMagno` computes two temporal ON/OFF channels, with centre side 6 and surround width 8. Its temporal state is carried through a recording; the filters reset between recordings. The first frame's magno output is zero. Preprocessing uses the filters' float64 internal computation and saves unscaled float16 responses at native image resolution.

Each cached split is `[videos × 48, 12, 128, 128]`, dtype `float16`. Training reshapes a video batch to `[batch, 48, 12, 128, 128]` and casts gathered frames to float32 on the selected device. All 48 frames are used in chronological order on every epoch. The cache has no additional spatial pooling or display normalization.

## Network architecture

```text
Stereo retinal frame: 12 × 128 × 128
  Conv1 → Conv2 → Conv3: shared features, 64 × 32 × 32
    Object: Conv4 → Conv5 → Conv6 → pool → linear128
            → ReLU RNN128 → twelve class logits
    Motion: adapter128 → input convolution
            → spatial tanh RNN128 → pool → two velocity components
```

Every numbered convolution has GroupNorm with eight groups and ReLU after it; these convolution layers have no bias.

| Layer | Channels | Kernel / stride / padding | Output per frame |
|---|---|---|---|
| Conv1, shared | 12 → 32 | 5 / 2 / 2 | 32 × 64 × 64 |
| Conv2, shared | 32 → 32 | 3 / 1 / 1 | 32 × 64 × 64 |
| Conv3, shared | 32 → 64 | 3 / 2 / 1 | 64 × 32 × 32 |
| Conv4, object | 64 → 64 | 3 / 1 / 1 | 64 × 32 × 32 |
| Conv5, object | 64 → 128 | 3 / 2 / 1 | 128 × 16 × 16 |
| Conv6, object | 128 → 128 | 3 / 1 / 1 | 128 × 16 × 16 |

The object branch applies adaptive average pooling to 4 × 5, flattens to 2,560 values, then applies Linear(2560, 128) and ReLU. A one-layer, ungated `nn.RNN(128, 128, nonlinearity='relu')` carries object state across frames. Linear(128, 12) gives class logits. The last frame's logits determine the video classification.

The motion branch taps Conv3 after its normalization and ReLU. A bias-free 1 × 1 adapter maps 64 to 128 channels. A 3 × 3 input convolution with bias supplies the current drive; a bias-free 3 × 3 recurrent convolution supplies the previous-state drive. Both use padding 1. The update is:

```text
motion_state[t] = tanh(input_conv(adapter(features[t]))
                       + recurrent_conv(motion_state[t-1]))
```

The state is `[batch, 128, 32, 32]`: **128 spatial channels**, compared with the object RNN's 128-element vector. Adaptive average pooling to 4 × 5 and Linear(2560, 2) produce normalized velocity components at every frame. Both recurrent states start at zero for each video and carry through all 48 frames.

The current model has 966,926 trainable parameters: 658,572 in the object model and 308,354 added by the motion branch. Both objectives update the shared first three convolution blocks. The historical six-class motion-128 model has 966,152 parameters; its motion-64 counterpart has 738,248.

### Motion recurrence rationale

The motivation is to retain spatial memory across frames. Each location in the motion state receives the current local features and a learned combination of neighbouring locations in the previous state. The same recurrent convolution is applied across the grid. This gives the network a structure for learning movement of object edges and background features before pooling them into a velocity estimate.

A regular vector RNN could also learn motion. **A regular RNN has not been tested as the motion branch**, so the experiments do not establish that the ConvRNN is better. The existing regular 128-unit RNN serves classification. Earlier motion experiments varied convolutional recurrent width and branching point; they did not compare convolutional and vector recurrence.

## Motion reference and objectives

The two motion outputs represent actual object velocity in fixed room X/Y coordinates, in cm/s after scaling. Positive Y is upward. Camera movement affects the images, while the targets come from object displacement in the room.

For frame index `t = 1..47`:

```text
dt = 1 / 12 seconds
target_velocity[t] = 100 * (position_m[t, :2] - position_m[t-1, :2]) / dt
scale = 6.8926159357457255 cm/s
predicted_velocity_cm_s = model_velocity_output * scale
```

The scale is the original six-class training-set RMS across velocity components, retained unchanged for the twelve-class run. Recorded positions construct supervision labels; the network receives the retinal images as its input. Frame 0 has no preceding interval, so its motion output is excluded from loss and metrics.

The training objective is:

```text
loss = mean class cross-entropy across all 48 recurrent outputs
     + mean squared error of the two normalized velocity components
       across frames 1..47
```

The loss coefficients are both 1. The physical weighting of motion depends on the normalization scale; equal coefficients do not impose equal gradient contributions from the tasks. The same class label supervises every carried object state. Both pathways train jointly through the complete sequence with backpropagation through time.

Classification accuracy counts videos whose final-state class is correct. Speed MAE is the mean absolute difference between predicted and true vector magnitudes. Direction MAE is the unsigned angle between predicted and true vectors, ranging from 0° to 180°; zero target speeds are excluded and a zero predicted vector is assigned 90°. Velocity vector MAE averages the Euclidean length of the vector error, combining speed and direction errors.

## Training settings and model selection

| Setting | Current joint run |
|---|---|
| Initialization | All weights trained from scratch; seed 0 |
| Epochs | 128; all training videos once each epoch in seeded shuffled order |
| Sequence | All 48 chronological frames; full sequence gradients |
| Effective batch | 65 videos per update, with a smaller last batch |
| Physical batch | At most 32 videos; gradients accumulated to the effective batch |
| Optimizer | AdamW, initial learning rate 0.001 |
| Weight decay | 0.01 for encoder, classifier and motion; 0.0001 for object RNN |
| Learning rate schedule | Cosine annealing over 128 epochs |
| Gradient clipping | Global norm cap 5.783946721808504 |
| Object recurrence | Identity initialization, then every recurrent singular value set to 1 − 1e−6 initially and after each optimizer step |
| Motion recurrence | Learned convolutional recurrence with tanh |
| Precision | CUDA BF16 for CNN and motion recurrence; FP32 object RNN, readouts, losses and master weights; FP64 SVD for object recurrent projection |
| Augmentation | Disabled |
| Checkpoint selection | Lowest VAL final-state object CE + normalized motion component MSE; higher VAL classification accuracy breaks ties |

The selected current checkpoint is epoch 71. Training completed all 128 epochs; TEST was evaluated separately using the selected checkpoint. The historical six-class classification-only baseline instead selects by highest VAL final-state accuracy, breaking ties with lower CE, and uses physical batch 65. Its other recorded object training settings match the values above.

The recurrent layers are rate-based engineering approximations. Their implemented learning rule is supervised backpropagation through time with AdamW; the object recurrent projection is part of the current training procedure.

## Environment and commands

Run all commands **from the repository root**. The supported setup is Linux with an NVIDIA GPU and a CUDA-enabled PyTorch installation. Rendering uses Linux's `fork` multiprocessing; the existing retinal preprocessing requires CUDA. Install FFmpeg with the `libx264` encoder and make `ffmpeg` available on `PATH`.

The recorded environment was Python 3.13.7, PyTorch 2.10.0+cu130, NumPy 2.3.5, Numba 0.64.0, Pillow 12.0.0 and trimesh 4.12.2, using an RTX 5090. [requirements.txt](requirements.txt) pins the Python package versions. Training from a prepared cache needs NumPy and PyTorch; rendering additionally needs Numba, Pillow, trimesh and FFmpeg.

### Set up the environment and storage

Create a Python 3.13 environment, or use an existing compatible environment. Change the `SHAPES_STORAGE` value in this block to use your chosen storage directory or external drive:

```bash
python3.13 -m venv .venv-shapes
source .venv-shapes/bin/activate
python -m pip install -r networks/simple_shapes/requirements.txt

export SHAPES_PYTHON=python
export SHAPES_CODE="$PWD/networks/simple_shapes/code"
export SHAPES_STORAGE="$PWD/artifacts/simple_shapes_12classes_128"
export SHAPES_RUN="$SHAPES_STORAGE/joint_motion128_12classes_seed0"
```

The remaining commands use these variables in the same shell. With an existing cache, use its parent directory for `SHAPES_STORAGE` in the setup block and skip generation. A compatible cache contains `index.json` and `train.npy`, `val.npy`, `test.npy` in the format described above, including each recording's `positions_m` for motion targets.

Use the explicit arguments below. The training script retains generic defaults of motion width 16, 256 epochs and motion scale 1; the renderer retains optional fixed-camera and one-second motion defaults.

### Generate the recordings and retinal cache

A fresh checkout needs this step before training. These flags reproduce the 60° sweep, two-second motion periods and halved speed setting used by the recorded runs.

```bash
"$SHAPES_PYTHON" "$SHAPES_CODE/render_dataset.py" \
  --out "$SHAPES_STORAGE/data" --workers 8 \
  --camera-sweep-degrees 60 --motion-interval-seconds 2 --motion-speed-scale 0.5

"$SHAPES_PYTHON" "$SHAPES_CODE/prepare_retina.py" \
  --data "$SHAPES_STORAGE/data" --cache "$SHAPES_STORAGE/cache" --device cuda:0
```

Both scripts require new output directories. Resolution, coarse grid spacing and split seeds are fixed in this renderer. Preprocessing writes the cache index, including the RGB source and retinal channel details.

### Train the current joint setup

This trains both pathways from scratch. Use a new output directory for each new run.

```bash
"$SHAPES_PYTHON" "$SHAPES_CODE/train_joint_motion.py" \
  --cache "$SHAPES_STORAGE/cache" --output "$SHAPES_RUN" \
  --branch-after 3 --motion-channels 128 \
  --motion-scale-cm-s 6.8926159357457255 \
  --device cuda:0 --seed 0 --epochs 128 --physical-batch 32
```

### Evaluate a newly completed run on TEST

Continue in the same shell after training:

```bash
"$SHAPES_PYTHON" "$SHAPES_CODE/train_joint_motion.py" \
  --cache "$SHAPES_STORAGE/cache" --output "$SHAPES_RUN" \
  --branch-after 3 --motion-channels 128 \
  --device cuda:0 --physical-batch 32 --evaluate-test
```

Evaluation loads `best.pt` and its saved motion scale. Match the branch and motion width to the checkpoint. Training refuses to overwrite a nonempty output directory; TEST evaluation refuses to overwrite an existing `test_report.json`. Once a run has been evaluated, read that report for its scores.

### Historical six-class setup

The current renderer generates twelve classes and 7,200 recordings. The six-class generator, baseline commands and guide are preserved on [branch `codex/simple-shapes-motion128-20261020`](https://github.com/Vishnu-Mohan-USyd/blight_world/tree/codex/simple-shapes-motion128-20261020/networks/simple_shapes). Use that version and a separate storage directory to reproduce the historical six-class experiments.

### Load your trained joint model

After joint training, this example loads your run's `best.pt`, classifies one cached validation video and obtains its 47 physical velocity predictions. Run in Python from the repository root, with `SHAPES_STORAGE` and `SHAPES_RUN` exported as above. Set `SHAPES_RUN` to another completed joint run to load it instead.

```python
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

network = Path.cwd() / 'networks/simple_shapes'
sys.path.insert(0, str(network / 'code'))
from train_joint_motion import JointClassifier

device = torch.device('cuda:0')
run = Path(os.environ['SHAPES_RUN'])
checkpoint = torch.load(run / 'best.pt',
                        map_location='cpu', weights_only=True)
config = checkpoint['config']
model = JointClassifier(len(config['classes']), config['branch_after'],
                        config['seed'], config['motion_channels']).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

cache = Path(os.environ['SHAPES_STORAGE']) / 'cache'
index = json.loads((cache / 'index.json').read_text())
frames = np.load(cache / 'val.npy', mmap_mode='r')
video_index = 0
video = np.array(frames[video_index * 48:(video_index + 1) * 48])
video = torch.from_numpy(video).unsqueeze(0).to(device=device, dtype=torch.float32)

with torch.inference_mode():
    logits, normalized_velocity = model(video)

label = logits[0, -1].argmax().item()
velocity_cm_s = normalized_velocity[0, 1:] * config['motion_scale_cm_s']
speed_cm_s = velocity_cm_s.norm(dim=-1)
direction_deg = torch.rad2deg(torch.atan2(velocity_cm_s[:, 1], velocity_cm_s[:, 0]))

print('Predicted:', config['classes'][label])
print('True:', config['classes'][index['splits']['val'][video_index]['label']])
print('Interval speed in cm/s:', speed_cm_s.cpu().tolist())
print('Room direction in degrees:', direction_deg.cpu().tolist())
```

Here direction is signed `atan2(v_y, v_x)`: 0° along room +X and +90° along room +Y. This is a direction estimate; the reported direction MAE is the unsigned angle between the prediction and target. Every forward call consumes a complete video and initializes both recurrent states to zero.
