# SADIE II H10 HRTF provenance

`sadie_h10_v2_2_48k.npz` is a runtime representation of the University of
York SADIE II human subject H10, downloaded from the official v2-2 Zenodo
release record 12092466 as `H10_HRIR_SOFA.zip`.

- Archive SHA-256: `9fa9b2043af37c97fa36907d8d4e96f52d527027a15a84320e03f7360a3de0a4`
- Source member: `H10_48K_24bit_256tap_FIR_SOFA.sofa`
- Source member SHA-256: `b48cf93b1b3919fa61a787242c75763fad2a4b0ee3464e53e165ba12a9720fde`
- Source arrays: 2,114 paired directions, two receiver channels, 256 taps,
  48,000 Hz. The stale SOFA global comment says 2,818 positions; array
  dimensions are authoritative.
- `Data.Delay` is `[0, 0]`; interaural delay is embedded in the paired FIRs.
- License: Apache License 2.0, Copyright 2018 University of York.

The conversion normalizes the global frontal 500--2,000 Hz magnitude with one
scalar. It does not normalize directions or ears independently. One common
linear-phase delay is recorded in metadata for runtime geometric-delay
compensation; the paired FIR taps and their ITD remain intact.

Dataset page: <https://www.york.ac.uk/sadie-project/database.html>

Associated paper: Armstrong et al., *A Perceptual Evaluation of Individual and
Non-Individual HRTFs: A Case Study of the SADIE II Database*, Applied Sciences
8(11), 2029 (2018), <https://doi.org/10.3390/app8112029>.
