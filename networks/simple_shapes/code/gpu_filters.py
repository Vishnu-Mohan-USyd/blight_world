"""CUDA ports of the accepted parvo and magno filters with persistent device state.

These reproduce :class:`tools.retina.parvo_filter.ParvoFilter` and
:class:`tools.retina.magno_filter.WindowMagnoFilter` on a torch device. The CPU
modules stay the reference implementation; nothing here changes their semantics.

The API is streaming rather than batched: one device frame in, one device
tensor out, with kernels, centre/surround normalizers and magno state held on
the device across calls. No call allocates a kernel or copies to the host, so
the same objects serve a realtime embodied loop and this collection pipeline.

``cv2.filter2D`` and ``cv2.boxFilter`` are correlations and so is ``conv2d``,
so no kernel is flipped. ``BORDER_CONSTANT`` becomes explicit zero padding that
reproduces each filter's anchor, and dividing by the filter's own response to a
ones image reproduces the CPU normalization at the border.
"""

import numpy as np
import torch
import torch.nn.functional as F


_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                 [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])
_HPE_LM = np.array([[0.4002, 0.7076, -0.0808],
                    [-0.2263, 1.1653, 0.0457]])


def _cone_rgb():
    """White-normalized HPE L/M proxies from linear sRGB, as both CPU filters build."""
    cone_rgb = _HPE_LM @ _XYZ
    return cone_rgb / cone_rgb.sum(axis=1, keepdims=True)


def _srgb_to_linear(srgb):
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


class _DeviceFilter:
    """Shared device/dtype handling and the CPU filters' input contract."""

    def __init__(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.device = torch.device(device)
        self.dtype = dtype
        if self.device.type == "cuda":
            # conv2d reaches cuDNN, which allows TF32 by default and would drop a
            # float32 mantissa to 10 bits. The accepted filters are IEEE float, so
            # keep full precision; float64 is unaffected either way.
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self._cone = torch.as_tensor(_cone_rgb(), dtype=dtype, device=self.device)
        self._shape = None

    def _linear(self, images):
        """Validate shape, then reproduce the CPU uint8/float sRGB -> linear step.

        Element values are not inspected: that would force a device
        synchronization every frame. The caller owns the ``[0, 1]`` contract,
        exactly as the staged float recordings already guarantee it.
        """
        if (not isinstance(images, torch.Tensor) or images.dim() != 4
                or images.shape[0] != 2 or images.shape[-1] != 3
                or min(images.shape[1:3]) < 1):
            raise ValueError("images must be a device tensor with shape [2,H,W,3]")
        if images.device != self.device:
            raise ValueError(f"images must already live on {self.device}")
        if images.dtype == torch.uint8:
            srgb = images.to(self.dtype) / 255.0
        elif images.dtype.is_floating_point:
            srgb = images.to(self.dtype)
        else:
            raise ValueError("images must be uint8 or floating-point sRGB in [0,1]")
        shape = tuple(images.shape[1:3])
        if self._shape != shape:
            self._prepare(shape)
            self._shape = shape
        return _srgb_to_linear(srgb)

    def _prepare(self, shape):
        raise NotImplementedError

    def reset(self):
        """Drop per-recording state. Cached kernels and normalizers are kept."""


class GPUParvo(_DeviceFilter):
    """``ParvoFilter`` on a device: four L/M centre-surround maps, no temporal state."""

    channels = ("p_l_on", "p_l_off", "p_m_on", "p_m_off")

    def __init__(self, center_side=2, surround_width=5, *, device="cuda:0",
                 dtype=torch.float64):
        if any(not isinstance(v, (int, np.integer)) or v < 1
               for v in (center_side, surround_width)):
            raise ValueError("center_side and surround_width must be positive integers")
        super().__init__(device, dtype)
        side = center_side + 2 * surround_width
        mid = (side + 1) // 2
        y, x = np.mgrid[1:side + 1, 1:side + 1]
        central = ((x > surround_width) & (x <= side - surround_width)
                   & (y > surround_width) & (y <= side - surround_width))
        cx = np.where(x > mid, center_side + surround_width + 1 - x, x - surround_width)
        cy = np.where(y > mid, center_side + surround_width + 1 - y, y - surround_width)
        sx = np.where(x > mid, side + 1 - x, x)
        sy = np.where(y > mid, side + 1 - y, y)
        center = ((cx + cy) * central).astype(np.float64)
        surround = ((sx + sy) * ~central).astype(np.float64)
        anchor = mid - 1
        # filter2D anchor index a leaves a taps before and side-1-a after it.
        self._pad = (anchor, side - 1 - anchor, anchor, side - 1 - anchor)
        self._kernels = torch.as_tensor(np.stack((center, surround))[:, None],
                                        dtype=self.dtype, device=self.device)
        self._normalizers = None

    def _sum(self, planes):
        """planes: [N,1,H,W] -> [N,2,H,W] holding the centre and surround sums."""
        return F.conv2d(F.pad(planes, self._pad), self._kernels)

    def _prepare(self, shape):
        ones = torch.ones((1, 1, *shape), dtype=self.dtype, device=self.device)
        normalizers = self._sum(ones)
        if bool((normalizers == 0).any()):
            raise ValueError("The image leaves a window without in-bounds centre or surround support")
        self._normalizers = (normalizers[:, 0:1], normalizers[:, 1:2])

    def __call__(self, images):
        """images: device [2,H,W,3] uint8 or float sRGB -> device float32 [2,4,H,W]."""
        linear = self._linear(images)
        height, width = linear.shape[1:3]
        cones = linear @ self._cone.T                               # [2,H,W,2]
        center_ideal, surround_ideal = self._normalizers
        planes = cones.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        sums = self._sum(planes)                                    # [2*2,2,H,W]
        center = (sums[:, 0:1] / center_ideal).reshape(2, 2, height, width)
        surround = (sums[:, 1:2] / surround_ideal).reshape(2, 2, height, width)
        difference = center - surround.mean(dim=1, keepdim=True)
        output = torch.empty((2, 4, height, width), dtype=torch.float32,
                             device=self.device)
        output[:, 0::2] = difference.clamp(0, 1).to(torch.float32)
        output[:, 1::2] = (-difference).clamp(0, 1).to(torch.float32)
        return output


class GPUMagno(_DeviceFilter):
    """``WindowMagnoFilter`` on a device, carrying its drive and weights between frames."""

    channels = ("m_on", "m_off")

    def __init__(self, centre_side=6, surround_width=8, *, device="cuda:0",
                 dtype=torch.float64):
        if any(not isinstance(v, (int, np.integer)) or v < 1
               for v in (centre_side, surround_width)):
            raise ValueError("centre_side and surround_width must be positive integers")
        super().__init__(device, dtype)
        self._sizes = (int(centre_side), int(centre_side + 2 * surround_width))
        # boxFilter anchor (side-1)//2 with normalize=False is a ones-kernel correlation.
        self._pads, self._boxes = [], []
        for side in self._sizes:
            anchor = (side - 1) // 2
            self._pads.append((anchor, side - 1 - anchor, anchor, side - 1 - anchor))
            self._boxes.append(torch.ones((2, 1, side, side), dtype=self.dtype,
                                          device=self.device))
        self._one = torch.ones((), dtype=self.dtype, device=self.device)
        self._three = torch.tensor(3, dtype=torch.int16, device=self.device)
        self.reset()

    def reset(self):
        self._previous = self._weights = None

    def _box(self, statistics, index):
        """statistics: [N,2,H,W] -> unnormalized box sums over the same layout."""
        padded = F.pad(statistics, self._pads[index])
        return F.conv2d(padded, self._boxes[index], groups=2)

    def _prepare(self, shape):
        ones = torch.ones((1, 2, *shape), dtype=self.dtype, device=self.device)
        center = self._box(ones, 0)
        ring = self._box(ones, 1) - center
        # Every gain is 1 or 1 + w/10 with w in [-3,3], so it is strictly positive:
        # the CPU support check depends only on how many taps land in bounds.
        if bool((center <= 0).any()) or bool((ring <= 0).any()):
            raise ValueError("The image leaves a window without centre or surround support")

    def __call__(self, images):
        """images: device [2,H,W,3] uint8 or float sRGB -> device float32 [2,2,H,W]."""
        linear = self._linear(images)
        height, width = linear.shape[1:3]
        drive = (linear @ self._cone.T).mean(dim=-1).to(torch.float32)   # [2,H,W]
        if self._previous is None:
            self._previous = drive
            self._weights = torch.zeros((2, 2, height, width), dtype=torch.int8,
                                        device=self.device)
            return torch.zeros((2, 2, height, width), dtype=torch.float32,
                               device=self.device)
        if drive.shape != self._previous.shape:
            raise ValueError("Image dimensions changed; call reset() before a new shape")
        current = torch.stack((drive, -drive), dim=1)                    # [2,2,H,W]
        previous = torch.stack((self._previous, -self._previous), dim=1)
        delta = current - previous
        positive = delta.clamp(min=0)
        gain = torch.where(delta > 0, 1 + self._weights.to(self.dtype) / 10.0, self._one)
        statistics = torch.stack((positive * gain, gain), dim=-1)        # [2,2,H,W,2]
        statistics = statistics.permute(0, 1, 4, 2, 3).reshape(4, 2, height, width)
        center = self._box(statistics, 0)
        ring = self._box(statistics, 1) - center
        output = (center[:, 0] / center[:, 1] - ring[:, 0] / ring[:, 1]).clamp(0, 1)

        boost = torch.floor(3 * positive).to(torch.int16)
        increments = torch.zeros_like(self._weights, dtype=torch.int16)
        # Compare every neighbour against the old weights, then apply all boosts.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys, ye = max(0, -dy), height - max(0, dy)
                xs, xe = max(0, -dx), width - max(0, dx)
                ty, tye = max(0, dy), height - max(0, -dy)
                tx, txe = max(0, dx), width - max(0, -dx)
                shifted = boost[..., ys:ye, xs:xe]
                eligible = ((self._weights[..., ty:tye, tx:txe] < shifted)
                            & (previous[..., ty:tye, tx:txe] < current[..., ys:ye, xs:xe]))
                increments[..., ty:tye, tx:txe] += torch.where(
                    eligible, shifted, torch.zeros_like(shifted))
        propagated = torch.minimum(self._three, self._weights.to(torch.int16) + increments)
        update = torch.where(delta <= 0, torch.ones_like(boost),
                             -torch.floor(7 * positive).to(torch.int16))
        self._weights = (propagated + update).clamp(-3, 3).to(torch.int8)
        self._previous = drive
        return output.reshape(2, 2, height, width).to(torch.float32)
