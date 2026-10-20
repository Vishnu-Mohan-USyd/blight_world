"""Apply the existing retinal filters to every native-resolution shape-video frame."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    source_file = Path(__file__).resolve().with_name('gpu_filters.py')
    spec = importlib.util.spec_from_file_location('shape_retinal_filters', source_file)
    filters = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(filters)
    index = json.loads((args.data / 'index.json').read_text())
    args.cache.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    parvo = filters.GPUParvo(device=device)
    magno = filters.GPUMagno(device=device)
    began = time.monotonic()
    with torch.inference_mode():
        for split in ('train', 'val', 'test'):
            rgb = np.load(args.data / f'{split}.npy', mmap_mode='r')
            count, frames, eyes, channels, height, width = rgb.shape
            assert (frames, eyes, channels, height, width) == (48, 2, 3, 128, 128)
            assert rgb.dtype == np.uint8 and count == len(index['splits'][split])
            output = np.lib.format.open_memmap(args.cache / f'{split}.npy', mode='w+',
                dtype=np.float16, shape=(count * frames, 12, height, width))
            block = torch.empty((frames, 12, height, width), dtype=torch.float16, device=device)
            for video in range(count):
                images = torch.from_numpy(np.array(rgb[video])).to(device).permute(0, 1, 3, 4, 2)
                parvo.reset()
                magno.reset()
                for frame in range(frames):
                    responses = torch.cat((parvo(images[frame]), magno(images[frame])), dim=1)
                    block[frame] = responses.reshape(12, height, width)
                output[video * frames:(video + 1) * frames] = block.cpu().numpy()
                if (video + 1) % 25 == 0 or video + 1 == count:
                    print(json.dumps({'split': split, 'videos': video + 1, 'count': count,
                                      'seconds': time.monotonic() - began}), flush=True)
            output.flush()
            del output
    index['rgb_source'] = str(args.data.resolve())
    index['dtype'] = 'float16'
    index['array_axes'] = ['frame', 'retinal_channel', 'y', 'x']
    index['retinal_input'] = {'implementation': str(source_file), 'dtype': 'float64',
        'spatial_size': [128, 128], 'spatial_pooling': False, 'frames': 'all 48 consecutive frames',
        'reset': 'parvo and magno reset per recording',
        'channels_per_eye': list(parvo.channels + magno.channels)}
    (args.cache / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'complete': True, 'seconds': time.monotonic() - began}), flush=True)


if __name__ == '__main__':
    main()
