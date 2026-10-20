"""Joint object/motion learning, comparing an early branch after Conv1, 2, or 3."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from motion_targets import build_motion_targets
from train_expanded_rnn import (
    FRAMES, GRADIENT_CAP, FrameSource, SequenceClassifier, project_recurrent_weight,
)


class JointClassifier(SequenceClassifier):
    def __init__(self, classes, branch_after, seed=0, motion_channels=16):
        super().__init__(classes)
        self.branch_after = branch_after
        self.motion_channels = motion_channels
        # The adapter gives each tap the requested recurrent width.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 10)
            self.motion_adapter = nn.Conv2d(64 if branch_after == 3 else 32, motion_channels, 1, bias=False)
            torch.manual_seed(seed + 11)
            self.motion_input = nn.Conv2d(motion_channels, motion_channels, 3, padding=1)
            torch.manual_seed(seed + 12)
            self.motion_recurrent = nn.Conv2d(motion_channels, motion_channels, 3, padding=1, bias=False)
            torch.manual_seed(seed + 13)
            self.motion_readout = nn.Linear(motion_channels * 4 * 5, 2)

    def forward(self, video):
        batch, frames = video.shape[:2]
        with torch.autocast(video.device.type, dtype=torch.bfloat16, enabled=video.is_cuda):
            x = video.flatten(0, 1)
            for layer_index, layer in enumerate(self.frame_encoder):
                x = layer(x)
                if layer_index + 1 == 3 * self.branch_after:
                    motion_features = x
            drive = self.motion_input(self.motion_adapter(motion_features))
            drive = drive.reshape(batch, frames, self.motion_channels, *drive.shape[-2:])
            state = torch.zeros_like(drive[:, 0])
            pooled = []
            for frame in range(frames):
                state = torch.tanh(drive[:, frame] + self.motion_recurrent(state))
                pooled.append(F.adaptive_avg_pool2d(state, (4, 5)).flatten(1))
        with torch.autocast(video.device.type, enabled=False):
            object_states, _ = self.rnn(x.float().reshape(batch, frames, 128))
            logits = self.classifier(object_states)
            velocity = self.motion_readout(torch.stack(pooled, dim=1).float())
        return logits, velocity

    def motion_parameters(self):
        for module in (self.motion_adapter, self.motion_input, self.motion_recurrent, self.motion_readout):
            yield from module.parameters()


class JointSource(FrameSource):
    def __init__(self, cache, split, entries, targets, device, physical_batch):
        super().__init__(cache, split, entries, device)
        self.buffer = torch.empty((physical_batch * FRAMES, 12, self.height, self.width), dtype=torch.float16,
                                  pin_memory=device.type == 'cuda')
        self.targets = torch.from_numpy(targets).to(device)

    def batch(self, videos, positions=None):
        x, y = super().batch(videos, positions)
        return x, y, self.targets[videos]


@torch.inference_mode()
def evaluate_joint(model, source, physical_batch, motion_scale_cm_s=1.0):
    model.eval()
    count, correct, ce_sum, intervals = 0, 0, 0.0, 0
    squared_error, vector_error, speed_error, angle_error, direction_count = 0.0, 0.0, 0.0, 0.0, 0
    for first in range(0, len(source.labels), physical_batch):
        videos = np.arange(first, min(first + physical_batch, len(source.labels)))
        x, y, target = source.batch(videos)
        logits, predicted = model(x)
        correct += int((logits[:, -1].argmax(-1) == y).sum())
        ce_sum += float(F.cross_entropy(logits[:, -1], y, reduction='sum'))
        count += len(videos)
        predicted, target = predicted[:, 1:] * motion_scale_cm_s, target[:, 1:]
        difference = predicted - target
        pred_speed, true_speed = predicted.norm(dim=-1), target.norm(dim=-1)
        squared_error += float(difference.square().sum())
        vector_error += float(difference.norm(dim=-1).sum())
        speed_error += float((pred_speed - true_speed).abs().sum())
        valid = true_speed > 1e-6
        dot = (predicted * target).sum(-1)
        cross = predicted[..., 0] * target[..., 1] - predicted[..., 1] * target[..., 0]
        angle = torch.rad2deg(torch.atan2(cross.abs(), dot))
        angle = torch.where(pred_speed > 1e-6, angle, 90.0)
        angle_error += float(angle[valid].sum())
        direction_count += int(valid.sum())
        intervals += target.shape[0] * target.shape[1]
    physical_mse = squared_error / (2 * intervals)
    mse = physical_mse / motion_scale_cm_s ** 2
    return {'correct': correct, 'count': count, 'accuracy': correct / count,
            'object_ce': ce_sum / count, 'motion_component_mse': mse,
            'velocity_component_mse_cm2_s2': physical_mse,
            'velocity_vector_mae_cm_s': vector_error / intervals,
            'speed_mae_cm_s': speed_error / intervals,
            'direction_mae_deg': angle_error / direction_count,
            'motion_intervals': intervals, 'direction_intervals': direction_count,
            'selection_loss': ce_sum / count + mse}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--branch-after', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--motion-channels', type=int, default=16)
    parser.add_argument('--motion-scale-cm-s', type=float, default=1.0,
                        help='Physical velocity represented by one motion-output unit')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--physical-batch', type=int, default=65)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    if args.motion_channels < 1:
        parser.error('--motion-channels must be positive')
    if not np.isfinite(args.motion_scale_cm_s) or args.motion_scale_cm_s <= 0:
        parser.error('--motion-scale-cm-s must be finite and positive')
    cache, output = args.cache.resolve(), args.output.resolve()
    index = json.loads((cache / 'index.json').read_text())
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    targets = build_motion_targets(cache, index)
    torch.manual_seed(args.seed)
    model = JointClassifier(len(index['classes']), args.branch_after, args.seed, args.motion_channels).to(device)

    def source(split):
        return JointSource(cache, split, index['splits'][split], targets[split], device, args.physical_batch)

    if args.evaluate_test:
        checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['config']['branch_after'] != args.branch_after:
            raise ValueError('Requested branch differs from checkpoint')
        if checkpoint['config'].get('motion_channels', 16) != args.motion_channels:
            raise ValueError('Requested motion width differs from checkpoint')
        if (output / 'test_report.json').exists():
            raise FileExistsError('TEST already evaluated for this run')
        model.load_state_dict(checkpoint['model_state'])
        result = {'epoch': checkpoint['epoch'], 'branch_after': args.branch_after,
                  'motion_channels': args.motion_channels,
                  'motion_scale_cm_s': checkpoint['config'].get('motion_scale_cm_s', 1.0),
                  'test': evaluate_joint(model, source('test'), args.physical_batch,
                                         checkpoint['config'].get('motion_scale_cm_s', 1.0))}
        write_json(output / 'test_report.json', result)
        print(json.dumps(result), flush=True)
        return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.rnn.weight_hh_l0.copy_(torch.eye(128, device=device))
    project_recurrent_weight(model)
    optimizer = torch.optim.AdamW([
        {'params': model.frame_encoder.parameters(), 'weight_decay': 0.01},
        {'params': model.classifier.parameters(), 'weight_decay': 0.01},
        {'params': model.rnn.parameters(), 'weight_decay': 0.0001},
        {'params': model.motion_parameters(), 'weight_decay': 0.01},
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {'cache': str(cache), 'branch_after': args.branch_after, 'seed': args.seed,
              'motion_channels': args.motion_channels,
              'motion_scale_cm_s': args.motion_scale_cm_s,
              'device': str(device), 'epochs': args.epochs, 'frames': FRAMES,
              'classes': index['classes'], 'physical_batch': args.physical_batch, 'effective_batch': 65,
              'parameters': sum(p.numel() for p in model.parameters()),
              'motion_parameters': sum(p.numel() for p in model.motion_parameters()),
              'initialization': 'From scratch, same object weights across branches; original identity/projected object recurrence; independently seeded motion adapter/input/recurrent/readout',
              'motion_architecture': f'1x1 adapter to{args.motion_channels}; Conv3x3 input + Conv3x3 recurrent; tanh; pool4x5; Linear{args.motion_channels * 4 * 5}->2; native tap resolution',
              'motion_reference': ('Actual object interval velocity in fixed room X/Y axes (Y up), cm/s'
                                   if 'positions_m' in index['splits']['train'][0] else
                                   'World object interval velocity expressed in current camera-heading horizontal right/forward axes, cm/s'),
              'motion_targets': 'Actual displacement across consecutive cache samples / dt; first frame excluded',
              'objective': f'Mean object CE across48 carried prefixes + mean squared motion component error across47 intervals, velocity unit{args.motion_scale_cm_s}cm/s, weight1',
              'selection': 'Lowest VAL final-object CE + motion component MSE; higher object accuracy breaks ties; TEST unused',
              'optimizer': f'AdamW lr0.001; decay0.01 except object RNN0.0001; cosine{args.epochs}; clip5.783946721808504; original object recurrent projection',
              'augmentation': False, 'sequence': 'All48 chronological frames, zero initial states, full BPTT, both losses train shared layers',
              'precision': 'BF16 CNN/motion recurrence, FP32 object RNN/readouts/losses/master weights'}
    write_json(output / 'config.json', config)
    train, val = source('train'), source('val')
    rng = np.random.default_rng(args.seed)
    history, best_key, best_epoch = [], (float('inf'), 0), 0
    began = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_began = time.monotonic()
        model.train()
        order = rng.permutation(len(train.labels))
        object_sum, motion_sum, correct = 0.0, 0.0, 0
        for first in range(0, len(order), 65):
            batch = order[first:first + 65]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), args.physical_batch):
                videos = batch[offset:offset + args.physical_batch]
                x, y, target = train.batch(videos)
                logits, velocity = model(x)
                object_loss = F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(FRAMES))
                motion_loss = F.mse_loss(velocity[:, 1:], target[:, 1:] / args.motion_scale_cm_s)
                loss = object_loss + motion_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch{epoch}')
                (loss * (len(videos) / len(batch))).backward()
                object_sum += float(object_loss.detach()) * len(videos)
                motion_sum += float(motion_loss.detach()) * len(videos)
                correct += int((logits.detach()[:, -1].argmax(-1) == y).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CAP, error_if_nonfinite=True)
            optimizer.step()
            project_recurrent_weight(model)
        validation = evaluate_joint(model, val, args.physical_batch, args.motion_scale_cm_s)
        key = (validation['selection_loss'], -validation['correct'])
        improved = key < best_key
        if improved:
            best_key, best_epoch = key, epoch
        row = {'epoch': epoch, 'train_object_prefix_ce': object_sum / len(order),
               'train_motion_component_mse': motion_sum / len(order),
               'train_final_accuracy': correct / len(order), 'val': validation,
               'selected': improved, 'seconds': time.monotonic() - epoch_began}
        history.append(row)
        scheduler.step()
        checkpoint = {'model_state': model.state_dict(), 'config': config, 'epoch': epoch,
                      'validation': validation}
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        torch.save(checkpoint, output / 'last.pt')
        write_json(output / 'history.json', history)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True)['model_state'])
    report = {'config': config, 'epochs_completed': args.epochs, 'best_epoch': best_epoch,
              'best_val': history[best_epoch - 1]['val'], 'last_val': history[-1]['val'],
              'train': evaluate_joint(model, train, args.physical_batch, args.motion_scale_cm_s),
              'training_seconds': time.monotonic() - began}
    write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
