# Project lessons

Read [docs/HANDOVER.md](docs/HANDOVER.md) first for the current setup, commands,
module map, accepted examples, timing contracts and detailed failure analysis.
The notes below preserve lessons from earlier work; they are not an instruction
to broaden a user's task or run every historical validation workflow.

Current priorities from the conversation:

- Deliver the requested visible sequence. A build, harness or passing array check
  cannot substitute for inspecting the actual eyes/body over the complete video.
- Reuse `.venv-unreal`, UE 5.5.4 and the pinned stack. Keep Python-owned physics,
  refreshed observation copies and simulation-clocked audio. Do not rebuild
  infrastructure just to produce another example.
- Use `CorrectedMuscleModel` when selecting muscles: setting activation is not a
  timestep. Restore direct-flight delay to both Steam Audio branches exactly once.
- The corrected street reference is `artifacts/unreal_street_run_redo`, not the
  earlier `unreal_street_run` video. Its gait is assisted and uses adjusted joint
  damping; realistic-looking movement does not establish autonomous running.
- The road redo fixed unstable pelvis damping, infeasible ankle targets, height
  tracking and missing acceleration torques. Compare actual moving-state data
  with the reference; do not hide bad motion by leveling the eye cameras.
- Use appropriate recorded footstep assets and physical landing events. Waveform
  checks establish timing/format, not perceived sound quality when listening is
  unavailable.
- Keep accepted media intact. Generated artifacts are ignored but remain local;
  the guide explains recreating required manifests after a fresh checkout.

## Earlier lessons

- Hold an approved frozen specification and keep advancing until the user
  explicitly changes it. Do not idle or flip between incompatible designs.
- Prioritize a coherent functioning auditory framework over infant-anatomy
  statistics, exhaustive provenance, native optimization, or perfection-level
  DSP. Adult SADIE metrics and clearly labeled generated impact assets are
  acceptable for this project's practical core.
- On any unexpected failure, preserve the exact command and output, stop
  speculative edits, and resume only from evidence-backed diagnosis.
- Make tests independently selectable: every test must create or register its
  own prerequisites rather than relying on module execution order.
- Treat measured array shape and validated metadata as truth when stale free
  text disagrees; retain a regression for the discrepancy.
- Copy caller-owned numerical arrays before marking internal state read-only.
- Account for streaming propagation and FIR transients before measuring a
  steady-state acoustic ratio.
- For user-facing embodiment demos, capture the sensor streams the user asked
  to inspect (for example, the two public eye observations), not a convenient
  external camera unless it is explicitly requested and labelled.
- Never present a kinematically clamped validation fixture as evidence of
  natural physics or autonomous interaction. After one-time staging, physical
  demos must drive objects through MuJoCo forces, actuators, constraints, and
  contacts, and must validate velocity, rebound, deceleration, and bounds from
  recorded trajectories.
- Distinguish short regression fixtures from user-facing showcase media. When
  the user wants to judge visual, auditory, and physical quality, provide
  enough duration and varied representative interactions to make that judgment
  possible, while declaring trial cuts and unsupported capabilities honestly.
- Treat straightforward extensions as extensions: reuse validated components
  and parameters, vary only what the request needs, and avoid reopening the
  architecture or launching broad research when a small composition change is
  sufficient.
- Do not substitute repeated passive object trials when the user asks to see
  embodied behavior. Separate simulator/body capability from controller
  capability explicitly: walking, crouching, grasping, and throwing require a
  motor controller or policy, and must never be implied by teleports, hidden
  clamps, or a showcase of unrelated falling objects.
- Never equate one narrow regression fixture with extensive system testing.
  Derive a coverage matrix from the user's requested behaviors and modalities,
  validate each included primitive independently, and make showcase media long
  and varied enough to expose failures rather than cherry-picking one event.
- When the user explicitly requires persistence until results, do not turn a
  difficult but authorized scope expansion into a permission question or go
  idle. Make the narrowest reasonable assumptions, keep executing through the
  specialist diagnose/implement/validate loop, and surface only genuine
  external blockers after exhausting in-scope alternatives.
- Never ask whether to begin or continue the next already-authorized phase.
  Proceed autonomously through implementation, validation, checkpointing, and
  artifact production; ask only when a genuinely new authority or material
  user choice is required.
- For a requested showcase, make the user's visible storyboard the completion
  checklist and inspect those event frames early. Do not substitute hashes,
  infrastructure work, or a narrower validated behavior for a missing action;
  the artifact is unfinished until every requested action is plainly visible.
- When correcting a showcase artifact, patch the visible failure directly and
  regenerate it. Time-box diagnostics to representative frames/audio windows;
  do not build new harnesses or launch broad validation unless the direct fix
  actually exposes a deeper problem.
- A claimed pickup/throw demo must prove that the object remains stationary
  until hand contact, attaches without a positional snap or levitation, moves
  through limb kinematics, and inherits release velocity. Disclose all mocap
  world-space routes; they cannot substitute for limb action. Before declaring
  impact audio correct, verify that detected events correlate with the emitted
  waveform in time and content.
- Quantify embodied motion in a named body-local frame before delivery, then
  inspect the corresponding visual evidence. World-frame speed alone can hide
  a sideways or downward action that reads incorrectly in the agent's frame.
- Direction thresholds cannot substitute for action semantics. Validate the
  visible limb/object displacement and released-object flight energy as well
  as the instantaneous body-local velocity before calling an action a throw.
- A showcase claimed as an eye view must use the public eye observations.
  Camera pose, aim, up-vector, or frustum overrides are external views and
  cannot be presented as the agent's eye feeds.
