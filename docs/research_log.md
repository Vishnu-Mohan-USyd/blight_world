# Audition research and engineering log

## Scope and timing

The auditory output models blocked-ear-canal pressure in pascals. It is an
engineering simulation for embodied learning, not a claim of exact infant
hearing. MuJoCo advances at 5 ms and MIMo applies two substeps per action, so
the sensor returns 10 ms of stereo audio: `(2, 480)` at 48 kHz or `(2, 240)` at
24 kHz. Audio has its own deterministically seeded random stream and cannot
change the MuJoCo state.

## Binaural filtering

The bundled bank is the paired 48 kHz SOFA measurement for adult subject H10
from SADIE II v2.2. Runtime conversion preserves all 2,114 measured direction
pairs and 256 taps per ear. The stale free-text SOFA comment reports 2,818
positions, so the converter deliberately trusts the validated array dimensions.
`Data.Delay` is `(1, 2)` zeros; interaural timing remains in the HRIR taps.
Conversion removes one 113-sample common acquisition delay and applies one
global frontal 500–2,000 Hz calibration scalar. It never applies independent
per-ear geometric delays, which would duplicate the HRIR's ITD.

The acoustic mapping is grounded by:

- Wightman and Kistler's measured free-field-to-eardrum filtering experiments,
  [JASA 85, 858–867 (1989), DOI 10.1121/1.397557](https://doi.org/10.1121/1.397557).
- Armstrong et al.'s measurement and perceptual evaluation of individual and
  non-individual SADIE II HRTFs,
  [Applied Sciences 8, 2029 (2018), DOI 10.3390/app8112029](https://doi.org/10.3390/app8112029).

Nearest measured direction lookup always selects the left/right pair together.
A persistent 5 ms crossfade limits discontinuities when direction changes. The
head-local virtual-ear offsets are `(0.01, ±0.0625, 0.0575)` m; the listener
position for common propagation delay is their midpoint. This is an adult-HRTF
surrogate on the fixed MIMo v2 head and is not anatomically individualized.

## Propagation, obstruction, and sample-rate conversion

For every source or image path, common propagation uses `delay = distance / 343`
seconds and pressure gain `1 / max(distance, 0.05 m)`. A 32-tap,
1,024-phase Kaiser-windowed fractional delay is the streaming implementation;
a 127-tap implementation is retained as the numerical reference. Source and
listener poses are updated on each 5 ms physics interval.

At 24 kHz, both HRIRs and dry assets derive from 48 kHz through a 127-tap
Kaiser-8.6 low-pass at 10.8 kHz, explicit group-delay compensation, and
decimation by two. Direct occlusion ray-tests MuJoCo geoms while excluding the
head/eyes and the source body. The blocked path approaches -9 dB through a
1.5 kHz first-order low-pass with 20 ms attack and 50 ms release.

Fixed sound speed, inverse-distance pressure, a binary ray decision, and the
single-pole obstruction filter are declared engineering approximations. The
model omits air absorption, diffraction, Doppler shift, pinna deformation, and
frequency-dependent material transmission.

## Room response

The early response uses the six first-order image sources of the 5 m × 5 m ×
2.5 m shoebox. Pressure reflection is `sqrt(1 - alpha)` with fixed absorption
coefficients 0.10 (floor), 0.20 (walls), and 0.30 (ceiling). Contact midpoint
sources up to 10 mm beyond a room plane are projected back to that plane;
larger excursions fail closed.

The late response is an eight-line feedback-delay network (FDN) with a 10 ms
predelay and orthonormal Sylvester-Hadamard feedback. Low-band RT60 follows
`clamp(0.161 V / sum(alpha_i S_i), 0.2, 1.2)`; the high-band target is
`max(0.15, 0.55 * RT60_low)`. Each line's DC and Nyquist gains are derived from
the corresponding 60 dB decay target.

Primary foundations are:

- Allen and Berkley's rectangular-room image-source method,
  [JASA 65, 943–950 (1979), DOI 10.1121/1.382599](https://doi.org/10.1121/1.382599).
- Jot and Chaigne's unitary digital delay networks for controlled decay,
  [AES Convention Paper 3030 (1991)](https://aes2.org/publications/elibrary-page/?id=5663).
- Schroeder's listening-tested artificial reverberation structures,
  [JAES 10, 219–223 (1962)](https://aes2.org/publications/elibrary-page/?id=849).

Only first-order specular images and a compact statistical tail are rendered;
this is not a wave solver and does not model room modes or scattering.

## Contact sound

New MuJoCo contacts are eligible when normal impulse is at least `5e-4 N s`
and approach speed is at least `0.05 m/s`. Resting contacts are seeded silent,
rearm time is 10 ms, and onsets within 20 mm are clustered. Static-static and
self contacts are silent. Material labels map room/MIMo geoms to hard, wood,
rubber, plastic, or soft. Estimated impact energy is
`E = 0.5 * normal_impulse * approach_speed`; pressure scale is
`clamp(sqrt(E / 0.01 J), 0.25, 4)`.

The three variants for each of 11 reachable material pairs and the four named
effects are deterministic synthesized dry assets. They are useful varied
excitation, not recordings of calibrated physical specimens or a validated
material-identification dataset. Asset provenance and regeneration details are
stored beside the WAV manifest.

## Reproducibility and asset identity

- SADIE II v2.2 H10 download archive SHA-256:
  `9fa9b2043af37c97fa36907d8d4e96f52d527027a15a84320e03f7360a3de0a4`.
- Extracted official H10 SOFA SHA-256:
  `b48cf93b1b3919fa61a787242c75763fad2a4b0ee3464e53e165ba12a9720fde`.
- Packaged runtime HRTF SHA-256:
  `afa86aa0edaf16067b27ff197bd23c25bd4311787195731acc4e2b011527e8c6`.
- Audio manifest SHA-256:
  `80235d09e72bc505a9b8976fcb442bb62e35c46716d579440b279b9a7dd81cdd`.

Loaders verify required metadata and every packaged WAV checksum, and fail
closed rather than silently substituting synthetic HRTFs or missing clips.
