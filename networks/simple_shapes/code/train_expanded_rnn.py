"""Six-convolution plain ReLU RNN with carried-prefix training and full-video scoring."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FRAMES = 48
PHYSICAL_BATCH = 3
VIDEO_BATCH = 65
GRADIENT_CAP = 5.783946721808504
SPECTRAL_CAP = 1.0 - 1e-6


class SequenceClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # Construct the frame encoder and head before the original plain RNN.
        static = nn.Sequential(
            nn.Conv2d(12, 32, 5, 2, 2, bias=False), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(), nn.Linear(2560, 128), nn.ReLU(),
            nn.Linear(128, classes),
        )
        self.frame_encoder = nn.Sequential(*list(static.children())[:-1])
        self.classifier = static[-1]
        self.rnn = nn.RNN(128, 128, nonlinearity='relu', num_layers=1, batch_first=True)

    def forward(self, video, prefixes=False):
        with torch.autocast(video.device.type, dtype=torch.bfloat16, enabled=video.device.type == 'cuda'):
            features = self.frame_encoder(video.flatten(0, 1))
        with torch.autocast(video.device.type, enabled=False):
            features = features.float().reshape(video.shape[0], video.shape[1], 128)
            states, hidden = self.rnn(features)  # Native zero initial state; no resets within a video.
            return self.classifier(states if prefixes else hidden[-1])


@torch.no_grad()
def project_recurrent_weight(model):
    weight = model.rnn.weight_hh_l0
    u, _, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    weight.copy_(((u * SPECTRAL_CAP) @ vh).to(weight.dtype))


def augment_videos(video, generator):
    """One mirror/eye swap and affine transform shared by every selected frame of a video."""
    batch, frames, channels, height, width = video.shape
    draws = torch.rand((batch, 5), generator=generator, device=video.device)
    mirrored = torch.cat((video[:, :, 6:], video[:, :, :6]), dim=2).flip(-1)
    video = torch.where((draws[:, 0] < 0.5)[:, None, None, None, None], mirrored, video)
    zoom = 0.8 + 0.45 * draws[:, 1]
    angle = (20 * draws[:, 2] - 10) * (math.pi / 180)
    dx, dy = (16 * draws[:, 3:5] - 8).unbind(1)
    cosine, sine = angle.cos() / zoom, angle.sin() / zoom
    theta = torch.zeros((batch, 2, 3), dtype=video.dtype, device=video.device)
    # Inverse pixel-space affine map, including rectangular-image aspect ratio.
    theta[:, 0, 0], theta[:, 0, 1] = cosine, sine * height / width
    theta[:, 1, 0], theta[:, 1, 1] = -sine * width / height, cosine
    theta[:, 0, 2] = -2 * (cosine * dx + sine * dy) / width
    theta[:, 1, 2] = -2 * (-sine * dx + cosine * dy) / height
    grid = F.affine_grid(theta, (batch, channels, height, width), align_corners=False)
    grid = grid[:, None].expand(-1, frames, -1, -1, -1).reshape(batch * frames, height, width, 2)
    return F.grid_sample(video.flatten(0, 1), grid, mode='bilinear', padding_mode='zeros',
                         align_corners=False).reshape(batch, frames, channels, height, width)


class FrameSource:
    def __init__(self, cache, split, entries, device):
        self.frames = np.load(cache / f'{split}.npy', mmap_mode='r', allow_pickle=False)
        self.labels = np.asarray([entry['label'] for entry in entries], dtype=np.int64)
        assert self.frames.dtype == np.float16
        assert self.frames.ndim == 4 and self.frames.shape[:2] == (len(entries) * FRAMES, 12)
        self.height, self.width = self.frames.shape[-2:]
        self.device = device
        self.buffer = torch.empty((PHYSICAL_BATCH * FRAMES, 12, self.height, self.width), dtype=torch.float16,
                                  pin_memory=device.type == 'cuda')

    def batch(self, videos, positions=None):
        if positions is None:
            positions = np.broadcast_to(np.arange(FRAMES), (len(videos), FRAMES))
        rows = (videos[:, None] * FRAMES + positions).reshape(-1)
        host = self.buffer[:len(rows)]
        np.take(self.frames, rows, axis=0, out=host.numpy(), mode='clip')
        # Blocking FP16 copy permits safe buffer reuse; cast on the chosen device.
        x = host.to(self.device).float().reshape(len(videos), positions.shape[1], 12, self.height, self.width)
        y = torch.from_numpy(self.labels[videos]).to(self.device)
        return x, y


@torch.inference_mode()
def evaluate(model, source, predictions=None):
    model.eval()
    correct, ce_sum = 0, 0.0
    for first in range(0, len(source.labels), PHYSICAL_BATCH):
        videos = np.arange(first, min(first + PHYSICAL_BATCH, len(source.labels)))
        x, y = source.batch(videos)
        logits = model(x)
        predicted = logits.argmax(-1)
        correct += int((predicted == y).sum())
        ce_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        if predictions is not None:
            predictions.extend(zip(y.cpu().tolist(), predicted.cpu().tolist()))
    count = len(source.labels)
    return {'correct': correct, 'count': count, 'accuracy': correct / count, 'ce': ce_sum / count}


def main():
    global PHYSICAL_BATCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=128, help='Training passes and cosine schedule length; default128')
    parser.add_argument('--train-frames', type=int, choices=(2, 48), default=48)
    parser.add_argument('--alternate-train-frames', action='store_true', help='Odd epochs use2-frame pairs, even epochs use full48; overrides --train-frames')
    parser.add_argument('--videos-per-update', type=int, default=VIDEO_BATCH, help='Videos per update; physical training chunks contain at most144 selected frames')
    parser.add_argument('--physical-batch', type=int, default=PHYSICAL_BATCH,
                        help='Full-length videos per physical batch; changes memory usage, preserving videos-per-update')
    parser.add_argument('--augment', action='store_true', help='TRAIN-only video-consistent mirror/eye swap, zoom, rotation and translation')
    parser.add_argument('--encoder-head-weight-decay', type=float, default=0.0001,
                        help='AdamW weight decay for frame_encoder and classifier; RNN remains0.0001')
    parser.add_argument('--evaluate-test', action='store_true', help='Only evaluate completed training\'s VAL-selected best.pt on TEST')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.videos_per_update < 1:
        parser.error('--videos-per-update must be positive')
    if args.physical_batch < 1:
        parser.error('--physical-batch must be positive')
    PHYSICAL_BATCH = args.physical_batch
    args.cache, args.output = args.cache.resolve(), args.output.resolve()
    index = json.loads((args.cache / 'index.json').read_text())
    classes, splits = index['classes'], index['splits']
    assert classes and len(set(classes)) == len(classes)
    assert len(index['frame_indices']) == FRAMES and (np.diff(index['frame_indices']) > 0).all()
    for entries in splits.values():
        assert entries and all(0 <= entry['label'] < len(classes) for entry in entries)
    identity_sets = {split: {entry['asset_id'] for entry in splits[split]} for split in ('train', 'val', 'test')}
    assert not (identity_sets['train'] & identity_sets['val'] or identity_sets['train'] & identity_sets['test']
                or identity_sets['val'] & identity_sets['test']), 'Objects must be disjoint across TRAIN/VAL/TEST'
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('--device must select CPU or CUDA')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if args.evaluate_test:
        if (args.output / 'test_report.json').exists():
            raise FileExistsError('TEST was already evaluated for this output')
        completed = json.loads((args.output / 'report.json').read_text())
        checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['config']['cache'] == str(args.cache) and checkpoint['config']['classes'] == classes
        assert completed['epochs_completed'] == checkpoint['config']['epochs']
        model = SequenceClassifier(len(classes)).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        predictions = []
        metrics = evaluate(model, FrameSource(args.cache, 'test', splits['test'], device), predictions)
        result = {'checkpoint': str(args.output / 'best.pt'), 'epoch': checkpoint['epoch'],
                  'selection': 'VAL accuracy, then lower VAL CE; no TEST selection',
                  'readout': 'Native final carried RNN state after all48 chronological frames', 'test': metrics,
                  'objects': [{'asset_id': entry['asset_id'], 'true_label': true, 'predicted_label': predicted}
                              for entry, (true, predicted) in zip(splits['test'], predictions)]}
        (args.output / 'test_report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result), flush=True)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    phase_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    augmentation_rng = torch.Generator(device=device).manual_seed(args.seed)
    train_lengths = (2, 48) if args.alternate_train_frames else (args.train_frames,)
    physical_batches = {length: min(args.videos_per_update, PHYSICAL_BATCH * FRAMES // length)
                        for length in train_lengths}
    train = FrameSource(args.cache, 'train', splits['train'], device)
    val = FrameSource(args.cache, 'val', splits['val'], device)
    model = SequenceClassifier(len(classes)).to(device)
    with torch.no_grad():
        model.rnn.weight_hh_l0.copy_(torch.eye(128, device=device))
    project_recurrent_weight(model)
    optimizer = torch.optim.AdamW([
        {'params': model.frame_encoder.parameters(), 'weight_decay': args.encoder_head_weight_decay},
        {'params': model.classifier.parameters(), 'weight_decay': args.encoder_head_weight_decay},
        {'params': model.rnn.parameters(), 'weight_decay': 0.0001},
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {'cache': str(args.cache), 'seed': args.seed, 'device': str(device), 'epochs': args.epochs,
              'train_frames': list(train_lengths) if args.alternate_train_frames else args.train_frames,
              'alternate_train_frames': args.alternate_train_frames,
              'train_length_schedule': 'Odd epochs2, even epochs48' if args.alternate_train_frames else f'Every epoch{args.train_frames}',
              'evaluation_frames': FRAMES,
              'classes': classes, 'frame_indices': index['frame_indices'],
              'input_shape': [12, train.height, train.width],
              'split_videos': {split: len(splits[split]) for split in ('train', 'val', 'test')},
              'architecture': '6Conv/GN8/ReLU:32k5s2,32k3s1,64k3s2,64k3s1,128k3s2,128k3s1; pool4x5; Linear2560->128/ReLU; native one-layer ReLU RNN128; Linear128->K',
              'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'initialization': 'All weights from scratch; frame encoder and head drawn before RNN; identity W_hh then scaled-orthogonal projection',
              'objective': 'Mean CE over every selected genuine carried-prefix output; full selected-sequence BPTT; no auxiliary classifier',
              'readout': 'Evaluation uses the native final carried state after all48 chronological frames',
              'reset_semantics': 'TRAIN: zero at the first selected frame, carry through every selected frame; evaluation: zero at cache frame0, carry through all48; no within-sequence reset',
              'input': 'Unscaled FP16 cache; gather only selected chronological TRAIN frames; evaluation always all48; pinned CPU streaming',
              'precision': 'CUDA BF16 encoder; CPU FP32 encoder; FP32 RNN, head, master weights and CE',
              'sampling': 'Seeded uniform permutation of all videos each epoch; every video once, no oversampling',
              'train_frame_sampling': 'Two-frame epochs: independent p uniform0..23 per video, positions[p,p+24], separate NumPy SeedSequence([seed,1]); full48 epochs: all cache positions0..47, no phase draws' if 2 in train_lengths else 'All cache positions0..47 in order; no phase draws',
              'physical_video_batch': physical_batches if args.alternate_train_frames else physical_batches[args.train_frames],
              'physical_frame_capacity': PHYSICAL_BATCH * FRAMES,
              'evaluation_physical_video_batch': PHYSICAL_BATCH, 'effective_video_batch': args.videos_per_update,
              'weight_decay': {'frame_encoder': args.encoder_head_weight_decay,
                               'classifier': args.encoder_head_weight_decay, 'rnn': 0.0001},
              'augmentation': {'enabled': args.augment, 'training_only': True, 'shared_across_selected_frames': True,
                               'mirror_probability': 0.5, 'mirror_swaps_eye_channels': True,
                               'zoom_uniform': [0.8, 1.25], 'rotation_uniform_degrees': [-10, 10],
                               'translation_uniform_pixels_each_axis': [-8, 8], 'interpolation': 'bilinear',
                               'padding': 'zeros', 'align_corners': False,
                               'randomness': 'Separate device Torch generator seeded with the run seed'},
              'optimizer': f'AdamW lr0.001 frame_encoder/classifier weight_decay{args.encoder_head_weight_decay}; rnn weight_decay0.0001; cosine T_max{args.epochs} epochs',
              'gradient_clip': GRADIENT_CAP,
              'projection': 'FP64 SVD; every W_hh singular value set to1-1e-6 initially and after each AdamW step; FP32 write',
              'selection': 'Best VAL final-state accuracy, lower CE breaks ties; TEST only separate --evaluate-test'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    history, updates, frame_presentations = [], 0, 0
    best_correct, best_ce, best_epoch = -1, float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        epoch_began = time.monotonic()
        updates_before_epoch = updates
        model.train()
        train_frames = (2 if epoch % 2 else 48) if args.alternate_train_frames else args.train_frames
        physical_train_batch = physical_batches[train_frames]
        order = rng.permutation(len(train.labels))
        phases = phase_rng.integers(0, 24, size=len(train.labels)) if train_frames == 2 else None
        ce_sum, correct = 0.0, 0
        for first in range(0, len(order), args.videos_per_update):
            batch = order[first:first + args.videos_per_update]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), physical_train_batch):
                videos = batch[offset:offset + physical_train_batch]
                positions = phases[videos, None] + np.array([0, 24]) if phases is not None else None
                x, y = train.batch(videos, positions)
                if args.augment:
                    x = augment_videos(x, augmentation_rng)
                logits = model(x, prefixes=True)
                loss = F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(train_frames))
                (loss * (len(videos) / len(batch))).backward()
                ce_sum += float(loss.detach()) * len(videos)
                correct += int((logits.detach()[:, -1].argmax(-1) == y).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CAP)
            optimizer.step()
            project_recurrent_weight(model)
            updates += 1
        validation = evaluate(model, val)
        improved = (validation['correct'], -validation['ce']) > (best_correct, -best_ce)
        if improved:
            best_correct, best_ce, best_epoch = validation['correct'], validation['ce'], epoch
        frame_presentations += len(order) * train_frames
        row = {'epoch': epoch, 'optimizer_updates': updates, 'learning_rate': optimizer.param_groups[0]['lr'],
               'updates_this_epoch': updates - updates_before_epoch,
               'train_prefix_ce': ce_sum / len(order), 'train_final_correct': correct, 'train_count': len(order),
               'train_frames': train_frames, 'physical_video_batch': physical_train_batch,
               'frame_presentations': len(order) * train_frames, 'total_frame_presentations': frame_presentations,
               'val': validation, 'selected': improved,
               'seconds': time.monotonic() - epoch_began}
        history.append(row)
        scheduler.step()
        checkpoint = {'model_state': model.state_dict(), 'config': config, 'epoch': epoch,
                      'optimizer_updates': updates, 'validation': validation}
        if improved:
            torch.save(checkpoint, args.output / 'best.pt')
        torch.save(checkpoint, args.output / 'last.pt')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
    assert updates == args.epochs * math.ceil(len(train.labels) / args.videos_per_update)
    report = {'config': config, 'epochs_completed': args.epochs, 'optimizer_updates': updates,
              'train_video_presentations': args.epochs * len(train.labels),
              'train_frames': config['train_frames'], 'evaluation_frames': FRAMES,
              'epochs_by_train_length': {str(length): sum(row['train_frames'] == length for row in history) for length in train_lengths},
              'train_frame_presentations': frame_presentations,
              'best_epoch': best_epoch, 'best_val': history[best_epoch - 1]['val'], 'last_val': validation,
              'test_evaluated': False, 'training_seconds': time.monotonic() - began}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
