# Evidence and Module Record for the Training and Assembly Plan

Evidence cutoff: 12 September 2026. This is a targeted multi-pass synthesis of primary experiments, original computational studies and official simulator documentation. It is not a preregistered systematic review or a claim to have examined every full paper and supplement. Some sources were available as publisher/author abstracts or indexed primary excerpts. The source-by-source access field records that limitation. The report proposes a training schedule; no agent was trained to validate it.

## Findings that determine the schedule

| Decision | Converging or contrasting evidence | What follows, and what does not |
|---|---|---|
| Begin sensory and body branches in parallel | [Ackman et al. (2012)](https://www.nature.com/articles/nature11529); [Ge et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34437090/); [Khazipov et al. (2004)](https://www.nature.com/articles/nature03132) | Early retinal and motor/somatosensory activity supports structured bootstrap before mature skills. It does not establish the exact artificial objective or a complete human timetable. |
| Separate passive pretraining from active calibration | [Li and DiCarlo (2008)](https://doi.org/10.1126/science.1160028); [Li and DiCarlo (2010)](https://doi.org/10.1016/j.neuron.2010.08.029); [Attinger, Wang and Keller (2017)](https://pubmed.ncbi.nlm.nih.gov/28602353/); [Sommer and Wurtz (2006)](https://www.nature.com/articles/nature05279) | Passive experience changes IT representations; active coupling changes particular V1 responses; command-copy circuitry supports remapping. Useful offline pretraining does not imply permanent completion of every sensory computation. |
| Distinguish acquisition dependencies from adult execution | [Kawai et al. (2015)](https://doi.org/10.1016/j.neuron.2015.03.024); [Riddle, Edgley and Baker (2009)](https://pubmed.ncbi.nlm.nih.gov/19369568/); [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371) | A learned subcortical behavior can have required cortical participation during acquisition. Brainstem primitives precede skilled cortical control, but the whole lower repertoire should not be completed in permanent isolation. |
| Avoid needless recognition prerequisites for grasping | [Buneo et al. (2002)](https://www.nature.com/articles/416632a); [Goodale et al. (1991)](https://www.nature.com/articles/349154a0); [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371) | Action geometry can support grasping without complete explicit recognition. Preserve geometry and somatic input instead of making an invariant IT class code the only motor input. |
| Start memory before mature grids and semantics | [Langston et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20558721/); [Nakazawa et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/12040087/); [Bittner et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28883072/); [Kitamura et al. (2017)](https://doi.org/10.1126/science.aam6808) | Early heading/place-related representations and rapid associative mechanisms support a coarse first memory release. The exact pretraining of DG/CA3/CA1 in an ANN remains a design proposal. |
| Use real closed-loop tests for memory and routing | [Guo et al. (2017)](https://www.nature.com/articles/nature22324); [Schmitt et al. (2017)](https://www.nature.com/articles/nature22073); [Bolkan et al. (2017)](https://www.nature.com/articles/nn.4568); [Hosford et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42214337/); [Saalmann et al. (2012)](https://pubmed.ncbi.nlm.nih.gov/22879517/) | Cortical-thalamic interactions contribute to persistence and updating. Coexecution is a functional requirement in these tasks; simultaneous optimization of every partner is not experimentally established. |
| Constrain specific anatomical return routes | [Kohara et al. (2014)](https://www.nature.com/articles/nn.3614); [Andrianova et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/41082527/); [Vega-Zuniga et al. (2025)](https://www.nature.com/articles/s41593-025-01874-w) | CA2 is a parallel route; reuniens targets require cell-type specificity; mouse vLGN is distinct from dLGN and pulvinar. Do not copy an undifferentiated arrow diagram into a human model. |
| Calibrate crossmodal interfaces using experience | [Wallace and Stein (2000)](https://doi.org/10.1152/jn.2000.83.6.3578); [Hyde and Knudsen (2002)](https://www.nature.com/articles/415073a); [Linkenhoker and Knudsen (2002)](https://www.nature.com/articles/nature01002) | Developing SC and auditory spatial maps depend on partner inputs; gradual perturbation can increase recalibration. Bird/cat findings constrain a training principle, not an exact primate nucleus mapping. |
| Start replay early and maintain differentiated learning signals | [Girardeau et al. (2009)](https://www.nature.com/articles/nn.2384); [Kitamura et al. (2017)](https://doi.org/10.1126/science.aam6808); [Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/); [Bittner et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28883072/); [Bellec et al. (2020)](https://www.nature.com/articles/s41467-020-17236-y); [Turrigiano et al. (1998)](https://www.nature.com/articles/36103) | Replay, cortical maturation, eligibility, fast plasticity and homeostasis have different evidence bases. One end-to-end loss is not a biological conclusion; e-prop is an original model rather than a proven brain rule. |
| Keep body needs and modulation present throughout | [Livneh et al. (2017)](https://www.nature.com/articles/nature22375); [Ran et al. (2022)](https://www.nature.com/articles/s41586-022-05139-5); [Adamantidis et al. (2007)](https://www.nature.com/articles/nature06310); [Uematsu et al. (2017)](https://www.nature.com/articles/nn.4642); [Pinto et al. (2013)](https://www.nature.com/articles/nn.3552); [Miyazaki et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25155504/); [Campos et al. (2018)](https://www.nature.com/articles/nature25511) | Early physiological state is part of the input-generating system; context-dependent modulation learns later. Simulator physiology must be declared, not hidden task reward renamed as interoception. |
| Separate acoustic learning, social grounding and mature instruction use | [Saffran, Aslin and Newport (1996)](https://pubmed.ncbi.nlm.nih.gov/8943209/); [Kuhl, Tsao and Liu (2003)](https://pubmed.ncbi.nlm.nih.gov/12861072/); [DiNicola, Braga and Buckner (2020)](https://doi.org/10.1152/jn.00529.2019); [Hitti and Siegelbaum (2014)](https://www.nature.com/articles/nature13028) | These capacities have different prerequisites. Full mentalizing is not a necessary first stage for learning a grounded word or a simple instruction. |
| Evaluate integration without assuming consciousness | [Redinbaugh et al. (2020)](https://pubmed.ncbi.nlm.nih.gov/32053769/); [Cogitate Consortium et al. (2025)](https://www.nature.com/articles/s41586-025-08888-1); [Fleming et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25100039/); [Kiani and Shadlen (2009)](https://pubmed.ncbi.nlm.nih.gov/19423820/) | State, content use, decision confidence and metacognition provide distinct constraints. No cited study validates a necessary-and-sufficient consciousness test for this artificial architecture. |

## Research and verification record

1. Re-read the existing circuit report and evidence record, retaining previous primary sources as leads rather than treating a generated report as independent scientific evidence.
2. Investigated developmental bootstrap, reticulospinal/rubrospinal circuits, spinal coordination, active sensing, visual and auditory calibration, and head-direction/retrosplenial dependencies.
3. Checked plasticity mechanisms, hippocampal input/return routes, thalamic maintenance/updating, current reuniens and vLGN evidence, bodily regulation, and social/language ordering.
4. Cross-checked consequential acquisition-versus-execution, V1-to-MT, recognition-versus-grasp and modern anatomical claims against accessible original records.
5. Separated recommended bootstrap edges from reciprocal execution partners. Removed unnecessary MT prerequisites for basic reach geometry and complete IT recognition prerequisites for basic grasping. Preserved early memory and working-state versions.
6. Validated all 49 manifest IDs, source references, release ordering and bootstrap acyclicity. Reciprocal runtime partners are deliberately represented separately.
7. Stopped broad discovery when the major ordering decisions had support or an explicit uncertainty, and additional papers were unlikely to change the dependency plan. No trained-model performance is claimed.

## Confidence and remaining experimental questions

Confidence is strongest for major anatomical input requirements, the need for real sensorimotor experience, specific reciprocal functional dependencies, and the distinction between acquisition and execution. Confidence is lower for a unique artificial objective, an exact local learning rule, and the earliest release at which an engineering implementation will pass a task. Whether the complete implementation supports subjective experience is not established.

The primary unresolved engineering questions are whether local learning reaches each proposed representation without privileged labels; whether released sources preserve enough information; how much interface recalibration is needed after new behaviors change the data distribution; and whether the assembled system generalizes beyond the training tasks. These are acceptance experiments, not assumptions to hide in the design.

## Module catalog

Each first release is a proposed implementation milestone, not a developmental age. Bootstrap prerequisites are requirements of this selected curriculum, not proof of universal necessity. Live partners can remain parameter-frozen while their online state evolves. Optional branches do not gate the nonsocial core. The JSON companion provides the same catalog in structured form.

### ENV: Body, receptors and causal timing

**First release:** R0. **Branch:** core.
**Anatomical scope:** Musculoskeletal plant; receptor models; anatomical connection masks.
**Bootstrap prerequisites:** none.
**Inputs:** Physical simulator outputs transformed into noisy, delayed receptor-equivalent signals; independent eye poses and action commands.
**Outputs:** Binocular streams, binaural waveforms, touch/proprioception/vestibular signals and issued-command copies.
**Training experience:** Passive perturbations, controlled camera/head movement, contacts and actuator impulse tests.
**Learning signal:** No neural task training. Calibrate transducers, causality and stable body dynamics.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** No future sample, world-pose field, identity label, auto-fixation or privileged critic enters the learner; passive and active traces remain distinguishable.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [MuJoCo documentation (2026)](https://mujoco.readthedocs.io/en/latest/XMLreference.html); [Mattern et al. (2023)](https://arxiv.org/abs/2312.04318).

### STATE: Operating-state and coarse homeostatic scaffold

**First release:** R0. **Branch:** core.
**Anatomical scope:** Hypothalamic/brainstem/basal-forebrain and central-thalamic state circuits.
**Bootstrap prerequisites:** ENV.
**Inputs:** Declared simulated bodily variables and sensory activity.
**Outputs:** Baseline excitability, state transitions and primary physiological consequences.
**Training experience:** Safe supported-body conditions with controlled rest, activation and bodily perturbations.
**Learning signal:** Start with specified baseline physiology and connectivity; later learn context-dependent state regulation.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Reproducible stable baseline; state interventions change responsiveness without supplying task answers.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Ran et al. (2022)](https://www.nature.com/articles/s41586-022-05139-5); [Adamantidis et al. (2007)](https://www.nature.com/articles/nature06310); [Redinbaugh et al. (2020)](https://pubmed.ncbi.nlm.nih.gov/32053769/).

### VIS: Retina-LGN-V1 basic spatiotemporal vision

**First release:** R1. **Branch:** core.
**Anatomical scope:** Retinal channels, dorsal LGN, local V1 populations.
**Bootstrap prerequisites:** ENV, STATE.
**Inputs:** Separate left/right luminance and color streams with local temporal history.
**Outputs:** Retinotopic orientation, spatial/temporal frequency, contrast and binocular population activity.
**Training experience:** Structured spontaneous activity, varied unlabeled natural views, gratings, moving textures and binocular input.
**Learning signal:** Proposed local competition/homeostasis and sensory reconstruction or prediction; angles and depth remain evaluator-only.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Receptive-field and tuning tests generalize to held-out contrasts, positions, eyes and temporal rates; activity is bounded.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Ackman et al. (2012)](https://www.nature.com/articles/nature11529); [Ge et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34437090/); [Olshausen and Field (1996)](https://www.nature.com/articles/381607a0); [Turrigiano et al. (1998)](https://www.nature.com/articles/36103).

### SOM: Somatic and proprioceptive front end

**First release:** R1. **Branch:** core.
**Anatomical scope:** Peripheral afferents; dorsal-column/trigeminal and thalamic routes; S1.
**Bootstrap prerequisites:** ENV, STATE.
**Inputs:** Cutaneous contact streams and muscle/joint-receptor proxies.
**Outputs:** Body-indexed contact and movement-related population states.
**Training experience:** Passive joint movement, gentle contact, spontaneous twitches and self-contact.
**Learning signal:** Learn sensory covariance, body topology and temporal prediction from actual consequences.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Localize contact in body coordinates; distinguish passive from commanded movement; predict withheld sensory channels.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Khazipov et al. (2004)](https://www.nature.com/articles/nature03132); [Mattern et al. (2023)](https://arxiv.org/abs/2312.04318).

### VES: Vestibular front end

**First release:** R1. **Branch:** core.
**Anatomical scope:** Canal/otolith receptor models and vestibular nuclei.
**Bootstrap prerequisites:** ENV, STATE.
**Inputs:** Head-centered angular-motion and specific-force signals with receptor dynamics.
**Outputs:** Head-motion and gravity-related evidence with latency and ambiguity retained.
**Training experience:** Passive rotations/translations and later self-generated head motion.
**Learning signal:** Calibrate local dynamics; learn relationships with subsequent sensory evidence rather than true world orientation labels.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Correct axis/sign/time response; distinguish different motion histories; no direct gravity-vector oracle.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [MuJoCo documentation (2026)](https://mujoco.readthedocs.io/en/latest/XMLreference.html); [Mattern et al. (2023)](https://arxiv.org/abs/2312.04318).

### AUD: Auditory front end

**First release:** R1. **Branch:** core.
**Anatomical scope:** Cochlear channels; cochlear nuclei; superior olive; inferior colliculus; MGB; A1.
**Bootstrap prerequisites:** ENV, STATE.
**Inputs:** Separate left/right acoustic waveforms.
**Outputs:** Tonotopic temporal features and binaural timing/level evidence.
**Training experience:** Isolated and mixed sounds, moving head and sources, varying reverberation.
**Learning signal:** Acoustic prediction and local covariance/timing learning without source identity or direction labels.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Frequency, temporal and binaural sensitivity on held-out sounds; distinguish timing cues from amplitude shortcuts.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Saffran, Aslin and Newport (1996)](https://pubmed.ncbi.nlm.nih.gov/8943209/); [Hyde and Knudsen (2002)](https://www.nature.com/articles/415073a).

### SPIN: Spinal primitives and motor output

**First release:** R1. **Branch:** core.
**Anatomical scope:** Motor pools; inhibitory/excitatory interneurons; reflex and rhythm-generating circuits.
**Bootstrap prerequisites:** ENV, STATE, SOM.
**Inputs:** Somatic/proprioceptive signals and bounded primitive motor drive.
**Outputs:** Muscle activation or explicitly labeled torque-baseline commands; local internal copy.
**Training experience:** Supported twitches, passive stretches, contact and short low-amplitude movement bouts.
**Learning signal:** Anatomically seeded sign/connection constraints; local sensory-motor calibration and homeostasis.
**Live execution partners:** ENV, SOM.
**Acceptance test:** Stable muscle/joint response, reciprocal coordination, bounded excursions, reproducible internal-copy timing.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Khazipov et al. (2004)](https://www.nature.com/articles/nature03132); [Azim et al. (2014)](https://www.nature.com/articles/nature13021); [Talpalar et al. (2013)](https://www.nature.com/articles/nature12286).

### INTERO: Visceral, danger and need representations

**First release:** R1. **Branch:** core.
**Anatomical scope:** NTS; parabrachial populations; hypothalamus; early insular/amygdalar targets.
**Bootstrap prerequisites:** ENV, STATE.
**Inputs:** Declared simulated visceral and damage-risk receptor streams.
**Outputs:** Bodily need and primary danger-related signals.
**Training experience:** Benign controlled changes in modeled satiety, fatigue, load and contact limits.
**Learning signal:** Sensory prediction and native homeostatic restoration; no hidden task-correctness signal.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Responses track physically modeled need and recovery, not a task-ID or scene shortcut.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Ran et al. (2022)](https://www.nature.com/articles/s41586-022-05139-5); [Campos et al. (2018)](https://www.nature.com/articles/nature25511); [Livneh et al. (2017)](https://www.nature.com/articles/nature22375).

### CHEM: Olfactory/gustatory transduction extension

**First release:** R1. **Branch:** optional/conditional.
**Anatomical scope:** Olfactory bulb/piriform and taste-brainstem/insular branches.
**Bootstrap prerequisites:** ENV, STATE, INTERO.
**Inputs:** Explicit odor concentration mixtures and taste receptor-equivalent signals, only if the environment implements them.
**Outputs:** Chemical identity/intensity evidence and its relation to bodily outcome.
**Training experience:** Controlled sources, mixtures and ingestion consequences.
**Learning signal:** Sensory and outcome association, with no odor-name labels.
**Live execution partners:** local circuit and its physical inputs.
**Acceptance test:** Concentration changes are separable from identity changes; no training on nonexistent simulator channels.
**Interpretation:** Optional plant extension. Included in coverage; no claim that MuJoCo supplies chemical sensing or that the cited sources specify the entire olfactory circuit.
**Evidence:** [Livneh et al. (2017)](https://www.nature.com/articles/nature22375); [Campos et al. (2018)](https://www.nature.com/articles/nature25511).

### RF: Reticular formation and reticulospinal control

**First release:** R2. **Branch:** core.
**Anatomical scope:** Pontomedullary reticular formation; bilateral reticulospinal projections; spinal targets.
**Bootstrap prerequisites:** SPIN, VES, INTERO.
**Inputs:** Vestibular/body/contact activity; coarse descending drive; issued-command copy.
**Outputs:** Distributed postural and limb drive, including appropriate distal contributions.
**Training experience:** Supported stance, small perturbations, head/trunk stabilization and supported reaching.
**Learning signal:** Sensory-based stability and consequence learning; coarse structure initialized anatomically.
**Live execution partners:** ENV, SPIN, VES, SOM.
**Acceptance test:** Recover from held-out perturbations; coordinate posture and limb movement; do not assume this pathway controls only axial muscles.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Riddle, Edgley and Baker (2009)](https://pubmed.ncbi.nlm.nih.gov/19369568/); [Schepens and Drew (2006)](https://journals.physiology.org/doi/full/10.1152/jn.00342.2006); [Kawai et al. (2015)](https://doi.org/10.1016/j.neuron.2015.03.024).

### EYE: Gaze stabilization and elementary orienting

**First release:** R2. **Branch:** core.
**Anatomical scope:** SC; pretectal/accessory-optic pathways; vestibular/ocular premotor integrators; III/IV/VI nuclei; cerebellar partners.
**Bootstrap prerequisites:** VIS, VES, SPIN.
**Inputs:** Retinal slip, visual population activity, vestibular input and eye-command copy.
**Outputs:** Eye/head commands and a causally timed corollary-discharge stream.
**Training experience:** Passive head rotation, stationary targets, reflexive orienting and later self-generated gaze shifts.
**Learning signal:** Retinal-slip and later sensory-consistency learning; no automatic target fixation or target coordinates.
**Live execution partners:** ENV, VIS, VES, CB0.
**Acceptance test:** Stable fixation during head motion and appropriate responses to target motion; command-copy ablation changes remapping.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Sommer and Wurtz (2006)](https://www.nature.com/articles/nature05279); [Attinger, Wang and Keller (2017)](https://pubmed.ncbi.nlm.nih.gov/28602353/).

### CB0: Early cerebellar predictive calibration

**First release:** R2. **Branch:** core.
**Anatomical scope:** Mossy-fiber/granule/Purkinje/deep-nuclear and climbing-fiber microcircuits.
**Bootstrap prerequisites:** SPIN, SOM, VES.
**Inputs:** Sensorimotor context, primitive commands and their sensed consequences.
**Outputs:** Predictive correction and a learned mapping from command/context to consequences.
**Training experience:** Repeated simple movement, gaze and postural perturbations.
**Learning signal:** Observed sensory discrepancy plus local eligibility/instructive signals; no perfect inverse-dynamics labels.
**Live execution partners:** ENV, SPIN.
**Acceptance test:** Reduce predictable errors; preserve response to unexpected perturbations; separate immediate feedback from across-trial adaptation.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Azim et al. (2014)](https://www.nature.com/articles/nature13021); [Herzfeld et al. (2018)](https://www.nature.com/articles/s41593-018-0136-y).

### GAIT: Locomotor drive and coordinated gait

**First release:** R2. **Branch:** core.
**Anatomical scope:** CnF/PPN mesencephalic locomotor circuits; reticular formation; spinal CPG partners.
**Bootstrap prerequisites:** SPIN, RF, CB0.
**Inputs:** Body/contact state and graded locomotor drive.
**Outputs:** Rhythmic, speed-dependent descending coordination.
**Training experience:** Supported stepping, varied loads and speeds, gradual removal of physical support.
**Learning signal:** Body-based stability and effort outcomes with local sensorimotor adaptation.
**Live execution partners:** ENV, RF, SPIN, CB0.
**Acceptance test:** Speed/gait modulation and left-right coordination across untrained loads; no vision dependency for basic supported stepping.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Caggiano et al. (2018)](https://www.nature.com/articles/nature25448); [Talpalar et al. (2013)](https://www.nature.com/articles/nature12286).

### BG0: Basic action regulation and outcome association

**First release:** R2. **Branch:** core.
**Anatomical scope:** Striatum; GPe; GPi/SNr; STN; dopamine-related teaching pathways.
**Bootstrap prerequisites:** SPIN, INTERO, STATE.
**Inputs:** Available movement/activity states, sensory cues and physically delivered outcomes.
**Outputs:** Selective modulation of downstream action expression, vigor and interruption.
**Training experience:** Small repertoire of already executable actions with different sensed consequences.
**Learning signal:** Outcome-contingent three-factor plasticity as a candidate; retain sign-constrained inhibitory output routes.
**Live execution partners:** ENV, SPIN.
**Acceptance test:** Change preference with outcome reversal, modulate vigor, and interrupt an ongoing action without destroying its trajectory generator.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Steinberg et al. (2013)](https://www.nature.com/articles/nn.3413); [Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/); [Fife et al. (2017)](https://elifesciences.org/articles/27689).

### HD0: Initial head-direction/path-integration state

**First release:** R2. **Branch:** core.
**Anatomical scope:** Vestibular brainstem; dorsal tegmental/lateral mammillary/anterior-thalamic heading circuit.
**Bootstrap prerequisites:** VES, RF.
**Inputs:** Angular self-motion evidence and occasional landmark features.
**Outputs:** A recurrent estimate of relative heading with drift and reset behavior.
**Training experience:** Supported rotations and short movements, darkness intervals, controlled landmark rotations.
**Learning signal:** Recurrent continuity and cross-sensory calibration; no compass-angle supervision.
**Live execution partners:** ENV, VES.
**Acceptance test:** Relative-direction continuity in darkness and appropriate landmark correction, quantified by an observer.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Langston et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20558721/); [Waye et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/31647251/).

### RUB: Rubrospinal comparative branch

**First release:** R2. **Branch:** optional/conditional.
**Anatomical scope:** Red nucleus and rubrospinal projections.
**Bootstrap prerequisites:** SPIN, CB0.
**Inputs:** Motor and cerebellar state.
**Outputs:** Additional descending limb-control influence.
**Training experience:** Species-matched reaching and hand/digit tasks.
**Learning signal:** Same sensed motor consequences as other controllers, with anatomy appropriate to the chosen species.
**Live execution partners:** ENV, SPIN, CB0.
**Acceptance test:** Contribution to skilled movement is tested separately from reticulospinal output; optional for the proposed primate-oriented core.
**Interpretation:** RST can also abbreviate rubrospinal tract. Rodent causal evidence is not sufficient to assign the same importance to the human pathway.
**Evidence:** [Morris et al. (2011)](https://journals.sagepub.com/doi/abs/10.1089/neu.2010.1708).

### V2V4: Intermediate form and surface processing

**First release:** R3. **Branch:** core.
**Anatomical scope:** V2/V4 and recurrent early-visual connections.
**Bootstrap prerequisites:** VIS.
**Inputs:** Actual spatially organized V1 activity and sensory history.
**Outputs:** Grouped contour/surface/shape-related activity with retinotopy preserved.
**Training experience:** Natural views, partial occlusion, changing contrast and repeated viewpoints.
**Learning signal:** Proposed local prediction/reconstruction and continuity constraints; no segmentation oracle.
**Live execution partners:** VIS.
**Acceptance test:** Transfer to unseen contours, surfaces and occlusion; later feedback can improve ambiguous inputs without dominating clear evidence.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Kar et al. (2019)](https://www.nature.com/articles/s41593-019-0392-5); [Olshausen and Field (1996)](https://www.nature.com/articles/381607a0).

### MT: Motion population processing

**First release:** R3. **Branch:** core.
**Anatomical scope:** MT and motion-related extrastriate circuitry.
**Bootstrap prerequisites:** VIS.
**Inputs:** Actual temporal V1 populations, including spatial/phase information.
**Outputs:** Component and pattern motion-related activity; later self-motion context input.
**Training experience:** Moving gratings, plaids, textures and independent object motion.
**Learning signal:** Temporal sensory prediction and structured pooling/competition; angle/velocity labels used only by the evaluator.
**Live execution partners:** VIS.
**Acceptance test:** Held-out component versus pattern-motion tuning, speed generalization and preserved motion uncertainty.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Rust et al. (2006)](https://www.nature.com/articles/nn1786).

### IT: Object continuity and identity representation

**First release:** R3. **Branch:** core.
**Anatomical scope:** Ventral visual cortex and inferotemporal/perirhinal partners.
**Bootstrap prerequisites:** V2V4.
**Inputs:** Actual intermediate visual populations over time.
**Outputs:** Object-related activity stable across selected view changes while retaining useful distinctions.
**Training experience:** Continuous views of multiple objects with appearance, occlusion and viewpoint changes.
**Learning signal:** Temporal-contiguity and predictive association; no object-ID positives or ground-truth instance labels.
**Live execution partners:** VIS, V2V4.
**Acceptance test:** Generalization to new views and discrimination between similar objects; avoid collapsing all temporally adjacent objects.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Li and DiCarlo (2008)](https://doi.org/10.1126/science.1160028); [Li and DiCarlo (2010)](https://doi.org/10.1016/j.neuron.2010.08.029); [Kar et al. (2019)](https://www.nature.com/articles/s41593-019-0392-5).

### AUD1: Auditory objects and sequence retention

**First release:** R3. **Branch:** core.
**Anatomical scope:** Auditory belt/parabelt and temporal association cortex.
**Bootstrap prerequisites:** AUD.
**Inputs:** Acoustic/binaural population trajectories.
**Outputs:** Sound-source and phonological-sequence features.
**Training experience:** Repeating sound events, mixtures and structured acoustic sequences.
**Learning signal:** Predict temporal regularity and recover missing acoustic structure; ground semantics later.
**Live execution partners:** AUD.
**Acceptance test:** Segmentation and retention generalize beyond memorized waveforms; natural acoustic distractors reveal interference.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Saffran, Aslin and Newport (1996)](https://pubmed.ncbi.nlm.nih.gov/8943209/); [Hyde and Knudsen (2002)](https://www.nature.com/articles/415073a).

### SOM1: Haptic object and contact sequence processing

**First release:** R3. **Branch:** core.
**Anatomical scope:** S2/parietal somatosensory association circuits.
**Bootstrap prerequisites:** SOM, SPIN.
**Inputs:** Body-indexed tactile/proprioceptive trajectories and exploratory commands.
**Outputs:** Texture/shape/contact-event representations.
**Training experience:** Active exploration of varied surfaces and objects with safe forces.
**Learning signal:** Cross-time/cross-contact sensory prediction and association without object identity labels.
**Live execution partners:** ENV, SOM, SPIN.
**Acceptance test:** Generalize across fingers and exploration trajectories; distinguish expected contact from sensed contact.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Khazipov et al. (2004)](https://www.nature.com/articles/nature03132); [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371).

### WM0: Short-delay selected retention

**First release:** R3. **Branch:** core.
**Anatomical scope:** Content-specific cortex with primitive frontal/thalamic recurrent partners.
**Bootstrap prerequisites:** VIS.
**Inputs:** Actual sensory population cues, variable delays and subsequent probes.
**Outputs:** Retained task-relevant content available for comparison.
**Training experience:** One-item delayed comparisons, then interference and two-modality tasks.
**Learning signal:** Reconstruction of a previously observed sensory code and sensed task outcome; latent and active states remain distinct implementation choices.
**Live execution partners:** VIS, SOM.
**Acceptance test:** Retention improves above transient sensory persistence; report delay/load/interference curves instead of enforcing a fixed slot number.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Guo et al. (2017)](https://www.nature.com/articles/nature22324); [Bolkan et al. (2017)](https://www.nature.com/articles/nn.4568); [Luck and Vogel (1997)](https://www.nature.com/articles/36846).

### HP0: Early rapid associative memory

**First release:** R3. **Branch:** core.
**Anatomical scope:** EC-DG-CA3-CA1/subicular circuit scaffold.
**Bootstrap prerequisites:** VIS.
**Inputs:** Coarse sensory codes and temporal context, without semantic labels or adult grid codes.
**Outputs:** Recoverable associations between previously co-occurring population patterns.
**Training experience:** Repeated and overlapping sequences with partial cues.
**Learning signal:** Sparse separation, rapid association and reconstruction of experienced codes; explicit fast episodic plasticity.
**Live execution partners:** VIS, SOM.
**Acceptance test:** One/few-exposure retrieval and separation of similar sequences; no mature grid or IT prerequisite.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Langston et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20558721/); [Nakazawa et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/12040087/); [Horner et al. (2015)](https://www.nature.com/articles/ncomms8462); [Bittner et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28883072/).

### ATT0: Elementary selection and target-template feedback

**First release:** R3. **Branch:** core.
**Anatomical scope:** FEF/parietal and feature-selective frontal populations; primitive thalamic routing.
**Bootstrap prerequisites:** VIS, WM0, EYE.
**Inputs:** Retained sensory target, current visual population and gaze state.
**Outputs:** Feature/location bias and sampling priority.
**Training experience:** Cue-to-target matching among few distractors; empty or invalid cue trials.
**Learning signal:** Learn effects of a candidate bias through real downstream performance or sensory evidence; fixture biases are only unit tests.
**Live execution partners:** VIS, WM0, EYE.
**Acceptance test:** Improve relevant selection without fabricating targets; valid/invalid/neutral cue effects; no privileged target coordinates.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Moore and Armstrong (2003)](https://www.nature.com/articles/nature01341); [Bichot et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/26526392/).

### REPLAY: Offline cortical-memory interaction

**First release:** R3. **Branch:** core.
**Anatomical scope:** Hippocampal replay and cortical recipient circuits.
**Bootstrap prerequisites:** HP0.
**Inputs:** Stored episodes and their original source representations.
**Outputs:** Reinstated sequences and slow cortical updates.
**Training experience:** Interleaved rest and online experience from the first memory-capable checkpoint.
**Learning signal:** Reconstruction and consolidation using retained observations; do not supply unobserved truth.
**Live execution partners:** HP0, VIS, SOM.
**Acceptance test:** Improved later recall with reduced interference; compare to matched no-replay and shuffled-replay controls.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Girardeau et al. (2009)](https://www.nature.com/articles/nn.2384); [Kitamura et al. (2017)](https://doi.org/10.1126/science.aam6808).

### PPC: Body/world coordinate and action-geometry processing

**First release:** R4. **Branch:** core.
**Anatomical scope:** Posterior parietal VIP/MIP/PRR/AIP-related populations.
**Bootstrap prerequisites:** V2V4, SOM1, EYE, RF.
**Inputs:** Retinotopic visual geometry, gaze/command copies, proprioception and touch.
**Outputs:** Target-body relations and candidate reach/grasp geometry.
**Training experience:** Look-reach-touch sequences, passive body displacement and occluded targets.
**Learning signal:** Crossmodal consequence prediction and learned transformations from sensed contact.
**Live execution partners:** ENV, EYE, RF, SOM1.
**Acceptance test:** Accurate relationships across eye/head/hand configurations; no access to true object pose.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Buneo et al. (2002)](https://www.nature.com/articles/416632a); [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371).

### RSC: Landmark, route and perspective integration

**First release:** R4. **Branch:** core.
**Anatomical scope:** Retrosplenial cortex with anterior-thalamic, parietal and medial-temporal partners.
**Bootstrap prerequisites:** HD0, V2V4, HP0.
**Inputs:** Heading activity, landmark appearance, movement history and coarse retrieved context.
**Outputs:** Context-dependent relations between current view, route and remembered environment.
**Training experience:** Routes with repeated landmarks, reversed approaches, dark segments and cue conflicts.
**Learning signal:** Predict view/route continuity and reconcile remembered with currently sampled relationships.
**Live execution partners:** ENV, HD0, HP0, EYE.
**Acceptance test:** Transfer across approach directions and local/global cue changes; differentiate route position from a single image match.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [van der Goes et al. (2024)](https://elifesciences.org/articles/82952); [Park et al. (2024)](https://www.nature.com/articles/s41467-024-51391-w); [Alexander and Nitz (2015)](https://www.nature.com/articles/nn.4058).

### NAV: Place/path and entorhinal spatial refinement

**First release:** R4. **Branch:** core.
**Anatomical scope:** MEC/presubicular/subicular and hippocampal spatial circuits.
**Bootstrap prerequisites:** HD0, RSC, HP0.
**Inputs:** Estimated self-motion, boundary/landmark codes and existing episodic activity.
**Outputs:** Place/path relationships; potentially grid-like organization as an observed result.
**Training experience:** Unfamiliar spaces, changed geometry and partial landmark removal.
**Learning signal:** Sequence prediction, integration and episode association without position labels.
**Live execution partners:** ENV, RSC, HD0, HP0.
**Acceptance test:** Localize relative to experience; quantify drift/reanchoring/remapping. Do not require ideal hexagonal grids before any spatial competence.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Langston et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20558721/); [Waye et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/31647251/); [Alexander and Nitz (2015)](https://www.nature.com/articles/nn.4058).

### REACH: Cortical reach acquisition

**First release:** R4. **Branch:** core.
**Anatomical scope:** Posterior parietal/dorsal-premotor/M1 circuits with brainstem/spinal/cerebellar partners.
**Bootstrap prerequisites:** PPC, SPIN, RF, CB0, BG0.
**Inputs:** Action geometry, somatic state, context and regulated movement drive.
**Outputs:** Evolving descending reach commands and their copies.
**Training experience:** Supported reaches, then varied positions, dynamics and disturbances.
**Learning signal:** Sensed contact/proprioceptive consequences and observed reinforcement; no inverse-kinematics or pose target labels.
**Live execution partners:** ENV, PPC, RF, SPIN, CB0, BG0.
**Acceptance test:** Generalization across body configurations and dynamics; retain upstream primitive functions after training.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Buneo et al. (2002)](https://www.nature.com/articles/416632a); [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371); [Kawai et al. (2015)](https://doi.org/10.1016/j.neuron.2015.03.024).

### GRASP: Hand preshaping and dexterous contact

**First release:** R4. **Branch:** core.
**Anatomical scope:** AIP/ventral-premotor/F5 and M1/corticospinal-hand circuits.
**Bootstrap prerequisites:** REACH, V2V4, SOM1.
**Inputs:** Object shape, hand configuration and cutaneous feedback.
**Outputs:** Preshape, contact-sensitive finger commands and force adjustment.
**Training experience:** Objects differing in width, orientation, texture, friction and compliance.
**Learning signal:** Predicted versus sensed contact plus physically delivered success consequences.
**Live execution partners:** ENV, REACH, PPC, SOM1, SPIN.
**Acceptance test:** Grasp novel shape/friction combinations; distinguish loss of visual preshape from loss of motor execution using pathway perturbations.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Fogassi et al. (2001)](https://academic.oup.com/brain/article/124/3/571/334371); [Riddle, Edgley and Baker (2009)](https://pubmed.ncbi.nlm.nih.gov/19369568/); [Goodale et al. (1991)](https://www.nature.com/articles/349154a0).

### MULTI: Crossmodal causal binding and calibration

**First release:** R4. **Branch:** core.
**Anatomical scope:** Multisensory parietal/premotor/temporal populations and SC with cortical influences.
**Bootstrap prerequisites:** IT, AUD1, SOM1, PPC, EYE.
**Inputs:** Actual visual, auditory and haptic population trajectories with their natural delays.
**Outputs:** Compatible event representations and reliability-dependent influence.
**Training experience:** Matched and mismatched audiovisual events, synchronous/asynchronous contact and degraded modalities.
**Learning signal:** Crossmodal prediction and consequences of successful orienting/contact; source identities and matching labels remain private.
**Live execution partners:** ENV, EYE, PPC, AUD1, SOM1.
**Acceptance test:** Fuse compatible evidence and segregate incompatible evidence; change weighting with reliability; preserve modality-specific detail.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Wallace and Stein (2000)](https://doi.org/10.1152/jn.2000.83.6.3578); [Hyde and Knudsen (2002)](https://www.nature.com/articles/415073a); [Linkenhoker and Knudsen (2002)](https://www.nature.com/articles/nature01002).

### VISACTIVE: Closed-loop perceptual calibration

**First release:** R4. **Branch:** core.
**Anatomical scope:** V1/extrastriate recurrence; MT/MST; SC-thalamic-FEF and vestibular partners.
**Bootstrap prerequisites:** IT, MT, EYE, PPC, ATT0.
**Inputs:** Actual sensory streams plus their issued actions and contextual feedback.
**Outputs:** Action-calibrated sensory populations and improved self/external-motion discrimination.
**Training experience:** Interleaved active and passive movement, altered visuomotor coupling and unexpected perturbations.
**Learning signal:** Predicted-versus-observed sensory consequences; train interfaces first, then selected earlier plasticity.
**Live execution partners:** ENV, VIS, MT, EYE, PPC, ATT0.
**Acceptance test:** Active/passive and coupled/uncoupled conditions produce appropriate differences; pretraining sensory tests remain passed.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Attinger, Wang and Keller (2017)](https://pubmed.ncbi.nlm.nih.gov/28602353/); [Sommer and Wurtz (2006)](https://www.nature.com/articles/nature05279); [Kar et al. (2019)](https://www.nature.com/articles/s41593-019-0392-5).

### VLGN: Comparative ventral-LGN action coordination

**First release:** R4. **Branch:** optional/conditional.
**Anatomical scope:** Mouse ventral LGN inhibitory visuomotor hub and its SC/motor partners.
**Bootstrap prerequisites:** VISACTIVE.
**Inputs:** Optic-flow activity and relevant motor/state copies.
**Outputs:** Action-dependent inhibitory influences on visuomotor processing.
**Training experience:** Action-conflicted visual tasks and depth-related movement.
**Learning signal:** Sensory consequence and motor consistency learning under mouse-specific connectivity constraints.
**Live execution partners:** ENV, VIS, EYE, MT.
**Acceptance test:** Selective interaction deficits under disruption; retain distinction from dLGN and pulvinar.
**Interpretation:** Comparative mouse module; do not silently assign this exact architecture to human vLGN homologs.
**Evidence:** [Vega-Zuniga et al. (2025)](https://www.nature.com/articles/s41593-025-01874-w).

### EPI: Context-rich episodic binding and cortical reinstatement

**First release:** R5. **Branch:** core.
**Anatomical scope:** Perirhinal/parahippocampal-EC-DG-CA3-CA1-subiculum/deep-EC loops.
**Bootstrap prerequisites:** HP0, IT, RSC, NAV, MULTI.
**Inputs:** Actual object, place, body and event-sequence codes.
**Outputs:** Partial-cue reconstruction of experienced object-context relationships and cortical reactivation.
**Training experience:** One-exposure object-location episodes, highly overlapping episodes, moved objects and incomplete cues.
**Learning signal:** Rapid local association and cortical-code reconstruction; preserve direct EC-CA1 and EC-CA3 paths.
**Live execution partners:** HP0, IT, RSC, REPLAY.
**Acceptance test:** Correct completion without confusing similar episodes; successful reinstatement through actual cortical recipients; knowledge from unobserved episodes remains absent.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Nakazawa et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/12040087/); [Kohara et al. (2014)](https://www.nature.com/articles/nn.3614); [Horner et al. (2015)](https://www.nature.com/articles/ncomms8462); [Bittner et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28883072/).

### RETENTION: Flexible maintenance, updating and reactivation

**First release:** R5. **Branch:** core.
**Anatomical scope:** PFC-MD and cortical-thalamic loops; PPC-pulvinar/LP-TRN; content-specific cortex.
**Bootstrap prerequisites:** WM0, PPC, ATT0.
**Inputs:** Selected sensory/memory content, contextual update cues and distractors.
**Outputs:** Separately controlled maintenance, overwrite, reactivation and readout.
**Training experience:** Protect a target while revising search history; modality-switching and variable-delay tasks.
**Learning signal:** Predict/reconstruct retained source activity and learn which updates improve later sensed outcomes.
**Live execution partners:** WM0, PPC, ATT0.
**Acceptance test:** Dissociate maintenance failures from update failures; measure load/precision/interference; stage-specific loop perturbations.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Guo et al. (2017)](https://www.nature.com/articles/nature22324); [Schmitt et al. (2017)](https://www.nature.com/articles/nature22073); [Bolkan et al. (2017)](https://www.nature.com/articles/nn.4568); [Hosford et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42214337/); [Luck and Vogel (1997)](https://www.nature.com/articles/36846).

### REU: Reuniens-mediated context coordination

**First release:** R5. **Branch:** core.
**Anatomical scope:** Nucleus reuniens with PFC/EC and hippocampal interneuron-targeted projections.
**Bootstrap prerequisites:** EPI, RETENTION.
**Inputs:** Task/context activity and hippocampal/cortical states.
**Outputs:** Context-dependent modulation of memory-related circuit interactions.
**Training experience:** Context switches during encoding/retrieval and interference.
**Learning signal:** Learn target-specific influence rather than a generic excitatory copy of PFC into CA1 pyramidal cells.
**Live execution partners:** EPI, RETENTION.
**Acceptance test:** Perturb modulation and compare context use while preserving simpler retrieval; connectivity audit enforces known cell-type constraints.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Andrianova et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/41082527/).

### CA2: Social-recognition memory branch

**First release:** R5. **Branch:** optional/conditional.
**Anatomical scope:** EC/DG-CA2-deep-CA1 with relevant social sensory inputs.
**Bootstrap prerequisites:** EPI, IT, AUD1.
**Inputs:** Repeated individual-related appearance/voice/interaction patterns.
**Outputs:** Familiarity and encounter-specific social associations.
**Training experience:** Repeated encounters, similar-looking agents and changed behavioral context.
**Learning signal:** Episode association using sensory identity continuity, not simulator agent IDs.
**Live execution partners:** EPI, IT, AUD1.
**Acceptance test:** Distinguish familiar individuals across changed appearance and context; compare CA2-branch disruption to nonsocial memory tests.
**Interpretation:** Required for the proposed detailed social branch, not for the non-social core.
**Evidence:** [Kohara et al. (2014)](https://www.nature.com/articles/nn.3614); [Hitti and Siegelbaum (2014)](https://www.nature.com/articles/nature13028).

### ATT1: Distributed task-dependent routing

**First release:** R5. **Branch:** core.
**Anatomical scope:** Dorsal/ventral attention, feature-frontal, pulvinar and reticular-thalamic partners.
**Bootstrap prerequisites:** ATT0, RETENTION, VISACTIVE, MULTI.
**Inputs:** Maintained goals, salient events, uncertainty and cortical population content.
**Outputs:** Selective sensory gain, routing and orienting/readout priorities.
**Training experience:** Distraction, informative interruptions, hidden targets and variable cue reliability.
**Learning signal:** Consequences of sampling and selective processing with partners live; fixtures do not establish competence.
**Live execution partners:** VISACTIVE, RETENTION, MULTI, EYE.
**Acceptance test:** Improve target processing, recover from relevant interruption and resist irrelevant distraction without indiscriminate suppression.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Moore and Armstrong (2003)](https://www.nature.com/articles/nature01341); [Bichot et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/26526392/); [Saalmann et al. (2012)](https://pubmed.ncbi.nlm.nih.gov/22879517/); [Hosford et al. (2026)](https://pubmed.ncbi.nlm.nih.gov/42214337/).

### MOD: Learned bodily valuation and target-specific modulation

**First release:** R5. **Branch:** core.
**Anatomical scope:** Amygdala/insula/hypothalamus; LC; basal forebrain; raphe; dopamine/habenula-related channels.
**Bootstrap prerequisites:** INTERO, BG0, RETENTION, IT.
**Inputs:** Bodily need, cue/outcome history and ongoing task/circuit activity.
**Outputs:** Target-dependent gain, learning signals, persistence and action-value influences.
**Training experience:** Outcome reversal, satiation/devaluation, waiting and variable bodily effort.
**Learning signal:** Physically modeled consequences; projection-specific candidate plasticity and eligibility traces.
**Live execution partners:** INTERO, STATE, BG0, RETENTION.
**Acceptance test:** Need changes cue influence; outcome reversals are learned; waiting and motor slowing are distinguished; no single global gain is assumed.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Steinberg et al. (2013)](https://www.nature.com/articles/nn.3413); [Livneh et al. (2017)](https://www.nature.com/articles/nature22375); [Matsumoto and Hikosaka (2007)](https://www.nature.com/articles/nature05860); [Uematsu et al. (2017)](https://www.nature.com/articles/nn.4642); [Pinto et al. (2013)](https://www.nature.com/articles/nn.3552); [Miyazaki et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25155504/); [Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/).

### SEM: Grounded conceptual knowledge

**First release:** R6. **Branch:** core.
**Anatomical scope:** Anterior/lateral temporal association and modality-specific semantic partners.
**Bootstrap prerequisites:** IT, SOM1, MULTI, EPI, REPLAY.
**Inputs:** Sensory identity, recurrent affordances, interaction consequences and episodes.
**Outputs:** Reusable relations and concepts transferable across instances and modalities.
**Training experience:** Multiple instances sharing uses/properties, and exceptions preventing superficial grouping.
**Learning signal:** Slow prediction and association across real episodes; no language labels needed for the first concepts.
**Live execution partners:** IT, MULTI, EPI, REPLAY.
**Acceptance test:** Generalize a learned use/relation to new instances; distinguish identity, category and specific episode.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Li and DiCarlo (2008)](https://doi.org/10.1126/science.1160028); [Li and DiCarlo (2010)](https://doi.org/10.1016/j.neuron.2010.08.029); [Kitamura et al. (2017)](https://doi.org/10.1126/science.aam6808); [Baldassano et al. (2017)](https://doi.org/10.1016/j.neuron.2017.06.041).

### EVENT: Long-timescale event and prospective context

**First release:** R6. **Branch:** core.
**Anatomical scope:** Posterior-medial/PCC-RSC, angular/lateral-parietal and medial-prefrontal default-associated subnetworks.
**Bootstrap prerequisites:** EPI, RSC, RETENTION, SEM.
**Inputs:** Current observations, remembered relationships and evolving task context.
**Outputs:** Event continuity, boundary-sensitive updating and reconstructed prospective situations.
**Training experience:** Continuous activities, interrupted tasks, reordered events and route changes.
**Learning signal:** Predict/reconstruct event relations at several timescales; memory-grounded prospective sampling.
**Live execution partners:** EPI, RETENTION, RSC, SEM.
**Acceptance test:** Maintain situation across viewpoint changes, detect meaningful boundaries and resume after interruption.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Baldassano et al. (2017)](https://doi.org/10.1016/j.neuron.2017.06.041); [DiNicola, Braga and Buckner (2020)](https://doi.org/10.1152/jn.00529.2019); [Alexander and Nitz (2015)](https://www.nature.com/articles/nn.4058).

### TASKSTATE: Hidden task-state and outcome organization

**First release:** R6. **Branch:** core.
**Anatomical scope:** Orbitofrontal and frontal/cingulate task-related circuits.
**Bootstrap prerequisites:** EVENT, MOD, RETENTION, BG0.
**Inputs:** Observed cues, history and sensed outcomes.
**Outputs:** Inferred current contingency and action-relevant contextual state.
**Training experience:** Same visible object under different rules or outcome contingencies; hidden switches.
**Learning signal:** Predict sensed consequences from inferred history-dependent state.
**Live execution partners:** EVENT, MOD, RETENTION, BG0.
**Acceptance test:** Same stimulus produces different justified choices under different histories; recover after contingency change.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Schuck et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27657452/); [Steinberg et al. (2013)](https://www.nature.com/articles/nn.3413).

### EXEC: Hierarchical task control and action sequencing

**First release:** R6. **Branch:** core.
**Anatomical scope:** Frontoparietal control; cingulo-opercular/medial-frontal; basal-ganglia-thalamic loops.
**Bootstrap prerequisites:** TASKSTATE, ATT1, EPI, REACH, GRASP.
**Inputs:** Current goal, event model, remembered possibilities and available action-state information.
**Outputs:** Subgoals, readout/update decisions and action/sampling priorities.
**Training experience:** Multi-step tasks with interruptions, failed attempts, alternative routes and delayed outcomes.
**Learning signal:** Observed completion consequences and memory-guided prediction; preserve already learned action interfaces.
**Live execution partners:** TASKSTATE, ATT1, EPI, BG0, REACH, GRASP.
**Acceptance test:** Compose learned skills in new orders; revise a subgoal without erasing the overall goal; retain primitive performance.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Schmitt et al. (2017)](https://www.nature.com/articles/nature22073); [Schuck et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27657452/); [Baldassano et al. (2017)](https://doi.org/10.1016/j.neuron.2017.06.041); [Kawai et al. (2015)](https://doi.org/10.1016/j.neuron.2015.03.024).

### CLA: Distraction-resilience comparative module

**First release:** R6. **Branch:** optional/conditional.
**Anatomical scope:** Claustrum and appropriate cortical targets.
**Bootstrap prerequisites:** ATT1, RETENTION.
**Inputs:** Cortical task/distractor activity.
**Outputs:** Contextual influence on cortical processing during distraction.
**Training experience:** Sensory task with carefully matched auditory or crossmodal distractors.
**Learning signal:** Task-specific influence learning with target circuits running.
**Live execution partners:** ATT1, RETENTION.
**Acceptance test:** Perturbation selectively impairs distractor resistance more than simple undistracted discrimination.
**Interpretation:** A causally motivated auxiliary branch, not a consciousness generator.
**Evidence:** [Atlan et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0960982218308595).

### META: Confidence and information-seeking control

**First release:** R6. **Branch:** core.
**Anatomical scope:** Parietal decision populations and anterior-prefrontal/related evaluative circuits.
**Bootstrap prerequisites:** RETENTION, TASKSTATE, ATT1.
**Inputs:** Decision dynamics, evidence reliability and subsequent observed outcomes.
**Outputs:** Calibrated confidence-like state and requests for additional evidence.
**Training experience:** Opt-out, delayed verification, low visibility and conflicting modalities.
**Learning signal:** Calibration against later available outcomes rather than hidden truth labels; retain evaluator-only accuracy estimates.
**Live execution partners:** RETENTION, TASKSTATE, ATT1.
**Acceptance test:** Confidence predicts error on held-out tasks and affects sensible evidence seeking; separate accuracy from metacognitive calibration.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Fleming et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25100039/); [Kiani and Shadlen (2009)](https://pubmed.ncbi.nlm.nih.gov/19423820/).

### SOC: Grounded social inference

**First release:** R7. **Branch:** optional/conditional.
**Anatomical scope:** STS/TPJ and medial-prefrontal social-associated systems with episodic/CA2 partners.
**Bootstrap prerequisites:** CA2, EVENT, SEM, AUD1.
**Inputs:** Observed agents, gaze/actions, shared and unshared sensory histories.
**Outputs:** Other-agent expectations and socially relevant event relations.
**Training experience:** Joint attention, turn taking, occluded access and changed locations known to only one agent.
**Learning signal:** Predict another agent from what it could observe, using actual sensory access and interaction outcomes.
**Live execution partners:** EVENT, SEM, EPI, CA2.
**Acceptance test:** Generalize across agents and separate own knowledge from another agent's available evidence.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [DiNicola, Braga and Buckner (2020)](https://doi.org/10.1152/jn.00529.2019); [Kuhl, Tsao and Liu (2003)](https://pubmed.ncbi.nlm.nih.gov/12861072/); [Hitti and Siegelbaum (2014)](https://www.nature.com/articles/nature13028).

### LANG: Grounded language and instruction use

**First release:** R7. **Branch:** optional/conditional.
**Anatomical scope:** Auditory-temporal and inferior-frontal language-related circuits; phonological retention.
**Bootstrap prerequisites:** AUD1, SEM, RETENTION, EXEC.
**Inputs:** Actual speech waveforms and shared sensory/action context.
**Outputs:** Compositional relations mapped to available concepts, memory and control.
**Training experience:** Contingent spoken interaction, contrasting instructions, novel object-action combinations.
**Learning signal:** Acoustic statistical learning plus sensed referential/action consequences; no pretrained text-model oracle.
**Live execution partners:** AUD1, SEM, RETENTION, EXEC.
**Acceptance test:** Understand new combinations and retain instructions through intervening action; separate auditory failure from semantic or control failure.
**Interpretation:** Interactive social experience is part of the environment; a fully trained theory-of-mind module is not a mandatory prerequisite.
**Evidence:** [Saffran, Aslin and Newport (1996)](https://pubmed.ncbi.nlm.nih.gov/8943209/); [Kuhl, Tsao and Liu (2003)](https://pubmed.ncbi.nlm.nih.gov/12861072/); [DiNicola, Braga and Buckner (2020)](https://doi.org/10.1152/jn.00529.2019).

### VOICE: Speech/vocal motor extension

**First release:** R7. **Branch:** optional/conditional.
**Anatomical scope:** Orofacial/laryngeal motor and brainstem circuits with auditory/cerebellar feedback.
**Bootstrap prerequisites:** LANG, SPIN, CB0.
**Inputs:** Communicative intent and real auditory/somatic feedback from a modeled vocal tract.
**Outputs:** Articulator commands and generated sound.
**Training experience:** Vocal exploration, auditory imitation and contingent interaction.
**Learning signal:** Sensory prediction and communication outcomes through the body.
**Live execution partners:** ENV, LANG, AUD1, CB0.
**Acceptance test:** Generalize novel sequences; demonstrate altered feedback adaptation. A text-to-speech API is an engineering substitute, not biological speech motor learning.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Saffran, Aslin and Newport (1996)](https://pubmed.ncbi.nlm.nih.gov/8943209/); [Kuhl, Tsao and Liu (2003)](https://pubmed.ncbi.nlm.nih.gov/12861072/); [Herzfeld et al. (2018)](https://www.nature.com/articles/s41593-018-0136-y).

### INTEGRATION: Integrated perception, flexible access and state tests

**First release:** R8. **Branch:** core.
**Anatomical scope:** Already trained recurrent cortical-subcortical architecture; no additional consciousness nucleus.
**Bootstrap prerequisites:** EXEC, EVENT, META, EPI, VISACTIVE, MULTI, RF, RETENTION, REPLAY.
**Inputs:** All available actual sensory, memory, body and task states.
**Outputs:** Cross-task access, continuous embodied behavior, report or nonverbal choice, calibrated uncertainty.
**Training experience:** Unfamiliar combinations of tasks, missing modalities, interruptions and reversible circuit/state perturbations.
**Learning signal:** Continue selected local adaptation and replay; no consciousness label or globally privileged success signal.
**Live execution partners:** All selected core modules; social/language branches only when included.
**Acceptance test:** Transfer new information across memory/action/perception, maintain continuity, seek missing evidence, and show interpretable causal dissociations. These are functional tests, not a validated proof of subjective experience.
**Interpretation:** A proposed engineering unit and curriculum milestone constrained by the cited evidence, not a uniquely established biological partition.
**Evidence:** [Redinbaugh et al. (2020)](https://pubmed.ncbi.nlm.nih.gov/32053769/); [Cogitate Consortium et al. (2025)](https://www.nature.com/articles/s41586-025-08888-1); [Fleming et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25100039/); [Guo et al. (2017)](https://www.nature.com/articles/nature22324); [Baldassano et al. (2017)](https://doi.org/10.1016/j.neuron.2017.06.041).

## Source-by-source evidence ledger

Entries distinguish observations from the proposed training consequence. Original article links are preserved; publisher PDFs and supplements are not bundled.

### 1. Ackman et al. (2012)

**[Retinal waves coordinate patterned activity throughout the developing visual system](https://www.nature.com/articles/nature11529)**. Nature.
**Evidence type:** Mouse imaging of spontaneous developmental activity.
**Supported observation:** Retinal waves drive patterned activity through SC and primary visual cortex before vision.
**Training implication:** Begin sensory calibration with structured activity before sophisticated behavior.
**Boundary:** Does not establish that a particular artificial pretraining loss is biological.
**Access:** primary abstract/indexed article.

### 2. Ge et al. (2021)

**[Retinal waves prime visual motion detection by simulating future optic flow](https://pubmed.ncbi.nlm.nih.gov/34437090/)**. Science.
**Evidence type:** Mouse developmental recording and perturbation.
**Supported observation:** Directional spontaneous waves influence developing visual-motion responses.
**Training implication:** Activity statistics are an early training resource; motion bootstrap need not await navigation.
**Boundary:** Rodent previsual development is not a universal human training timetable.
**Access:** primary abstract/indexed article.

### 3. Khazipov et al. (2004)

**[Early motor activity drives spindle bursts in the developing somatosensory cortex](https://www.nature.com/articles/nature03132)**. Nature.
**Evidence type:** Newborn rat electrophysiology.
**Supported observation:** Spontaneous muscle twitches trigger spatially localized somatotopic S1 spindle bursts.
**Training implication:** Motor babbling and sensory-map calibration belong in the same early wave.
**Boundary:** Circuit organization inference is distinct from proof of a unique learning algorithm.
**Access:** publisher abstract and references inspected.

### 4. Attinger, Wang and Keller (2017)

**[Visuomotor Coupling Shapes the Functional Development of Mouse Visual Cortex](https://pubmed.ncbi.nlm.nih.gov/28602353/)**. Cell.
**Evidence type:** Mouse controlled VR rearing and cell-type-resolved recording.
**Supported observation:** Normal versus random visuomotor coupling changes L2/3 mismatch responses; restoring coupling restores normal processing.
**Training implication:** Pretrain vision, then recalibrate real action-to-vision interactions before declaring active vision mature.
**Boundary:** Does not show all orientation selectivity requires self-motion or all V1 computes one error.
**Access:** PubMed and indexed publisher summary; full publisher open denied.

### 5. Kawai et al. (2015)

**[Motor Cortex Is Required for Learning but Not for Executing a Motor Skill](https://doi.org/10.1016/j.neuron.2015.03.024)**. Neuron.
**Evidence type:** Rat lesions before versus after acquisition.
**Supported observation:** Motor cortex required to learn the tested non-dexterous sequence, but learned execution survived lesions.
**Training implication:** Keep learning-dependency edges distinct from execution-dependency edges.
**Boundary:** Not evidence that primate dexterous movements become cortex independent.
**Access:** Primary author-laboratory abstract, including before-versus-after-training lesion results.

### 6. Riddle, Edgley and Baker (2009)

**[Direct and indirect connections with upper limb motoneurons from the primate reticulospinal tract](https://pubmed.ncbi.nlm.nih.gov/19369568/)**. Journal of Neuroscience.
**Evidence type:** Monkey intracellular physiology.
**Supported observation:** Reticulospinal mono- and disynaptic connections reach distal and intrinsic hand motoneurons.
**Training implication:** Model reticular source populations and spinal projection targets, not a separate tract black box.
**Boundary:** Does not define reticulospinal output as a complete grasp controller.
**Access:** primary PubMed abstract; PMC full-text open blocked.

### 7. Schepens and Drew (2006)

**[Descending Signals From the Pontomedullary Reticular Formation Are Bilateral, Asymmetric, and Gated During Reaching Movements in the Cat](https://journals.physiology.org/doi/full/10.1152/jn.00342.2006)**. Journal of Neurophysiology.
**Evidence type:** Cat reticular recording and spike-triggered effects during reach.
**Supported observation:** Bilateral asymmetric reticular signals contribute to posture and reach, with state-dependent spinal expression.
**Training implication:** Train anticipatory and reactive body support with limb movement before unconstrained reaching.
**Boundary:** Cat quadrupedal posture does not supply a ready-made human biped controller.
**Access:** extensive indexed primary text.

### 8. Caggiano et al. (2018)

**[Midbrain circuits that set locomotor speed and gait selection](https://www.nature.com/articles/nature25448)**. Nature.
**Evidence type:** Mouse cell-type-specific perturbation.
**Supported observation:** CnF and PPN glutamatergic populations have differing contributions to speed and gait.
**Training implication:** Use separate descending drive/state channels interacting with spinal patterns.
**Boundary:** Mouse gait results are not direct proof of human locomotor cell-type homology.
**Access:** primary abstract/indexed article.

### 9. Morris et al. (2011)

**[Impaired Arpeggio Movement in Skilled Reaching by Rubrospinal Tract Lesions in the Rat: A Behavioral/Anatomical Fractionation](https://journals.sagepub.com/doi/abs/10.1089/neu.2010.1708)**. Journal of Neurotrauma.
**Evidence type:** Rat lesion and skilled reaching.
**Supported observation:** Rubrospinal tract contributes to components of skilled forelimb movement.
**Training implication:** Retain a species-dependent red-nucleus/rubrospinal route in the coverage map.
**Boundary:** Do not transfer rodent pathway dominance directly to humans.
**Access:** primary abstract/indexed article.

### 10. Vega-Zuniga et al. (2025)

**[A thalamic hub-and-spoke network enables visual perception during action by coordinating visuomotor dynamics](https://www.nature.com/articles/s41593-025-01874-w)**. Nature Neuroscience.
**Evidence type:** Mouse anatomy, physiology and behavioral perturbation.
**Supported observation:** vLGN combines optic flow and motor copies, influences SC processing and depth-related behavior.
**Training implication:** Early sensorimotor correction includes subcortical loops; do not put all feedback only in PFC.
**Boundary:** Specific mouse circuit; not interchangeable with primate dLGN or pulvinar.
**Access:** publisher full HTML inspected through main results introduction.

### 11. Langston et al. (2010)

**[Development of the spatial representation system in the rat](https://pubmed.ncbi.nlm.nih.gov/20558721/)**. Science.
**Evidence type:** Developing rat head-direction, place and grid recordings.
**Supported observation:** Head direction is adult-like early; place/grid refine more gradually and grid is slowest.
**Training implication:** Bootstrap heading before mature grid coding; allow place/episodic training before mature grids.
**Boundary:** Developmental precedence alone does not prove a mandatory causal training order.
**Access:** primary abstract/indexed article.

### 12. van der Goes et al. (2024)

**[Coordinated head direction representations in mouse anterodorsal thalamic nucleus and retrosplenial cortex](https://elifesciences.org/articles/82952)**. eLife.
**Evidence type:** Mouse simultaneous recordings and connectivity.
**Supported observation:** ADn and RSC update direction coherently during cue rotation and remain coordinated in darkness.
**Training implication:** Train heading-landmark reconciliation as a coupled interface, then integrate with navigation memory.
**Boundary:** Evidence favors strong ADn-to-RSC drive and restricted feedback, not a symmetric universal transform.
**Access:** primary abstract/indexed article.

### 13. Park et al. (2024)

**[Egocentric neural representation of geometric vertex in the retrosplenial cortex](https://www.nature.com/articles/s41467-024-51391-w)**. Nature Communications.
**Evidence type:** Mouse calcium imaging and environmental manipulation.
**Supported observation:** RSC neurons code vertices relative to the animal; boundaries and goal relevance alter coding.
**Training implication:** Preserve geometric egocentric inputs to RSC; do not reduce it to a scene-identity classifier.
**Boundary:** Rodent representation evidence does not establish one complete RSC algorithm.
**Access:** primary abstract/indexed article.

### 14. Olshausen and Field (1996)

**[Emergence of simple-cell receptive field properties by learning a sparse code for natural images](https://www.nature.com/articles/381607a0)**. Nature.
**Evidence type:** Computational model.
**Supported observation:** Sparse reconstruction of natural images produces localized oriented filters.
**Training implication:** Begin visual representation learning without orientation class labels.
**Boundary:** This is an existence proof in a model, not identification of the cortical optimizer.
**Access:** primary abstract/indexed article.

### 15. Li and DiCarlo (2008)

**[Unsupervised Natural Experience Rapidly Alters Invariant Object Representation in Visual Cortex](https://doi.org/10.1126/science.1160028)**. Science.
**Evidence type:** Macaque exposure manipulation and IT recordings.
**Supported observation:** Altered temporal image pairings reshape object tolerance.
**Training implication:** Use temporally coherent upstream activity for ventral training.
**Boundary:** An exposure effect does not identify one unique temporal learning rule.
**Access:** primary abstract/indexed article.

### 16. Li and DiCarlo (2010)

**[Unsupervised Natural Visual Experience Rapidly Reshapes Size-Invariant Object Representation in Inferior Temporal Cortex](https://doi.org/10.1016/j.neuron.2010.08.029)**. Neuron.
**Evidence type:** Macaque controlled exposure and IT recording.
**Supported observation:** Temporal experience changes size tolerance, including conditions not requiring an eye movement.
**Training implication:** Passive sequential pretraining is useful for some object computations.
**Boundary:** Do not generalize passive sufficiency to active self-motion prediction.
**Access:** primary abstract/indexed article.

### 17. Wallace and Stein (2000)

**[Onset of Cross-Modal Synthesis in the Neonatal Superior Colliculus is Gated by the Development of Cortical Influences](https://doi.org/10.1152/jn.2000.83.6.3578)**. Journal of Neurophysiology.
**Evidence type:** Developing cat recordings and cortical manipulation.
**Supported observation:** Mature SC multisensory integration depends on developing cortical influences.
**Training implication:** Separate primitive orienting from later cortical-SC multisensory calibration.
**Boundary:** Sensory convergence is not itself mature multisensory integration.
**Access:** primary abstract/indexed article.

### 18. Yagishita et al. (2014)

**[A critical time window for dopamine actions on the structural plasticity of dendritic spines](https://pubmed.ncbi.nlm.nih.gov/25258080/)**. Science.
**Evidence type:** Mouse striatal spine experiments.
**Supported observation:** Dopamine promotes structural potentiation in a restricted interval after excitatory input.
**Training implication:** Use synapse-local eligibility with delayed modulatory instruction.
**Boundary:** The measured 0.3-2 s window is not a universal neural credit horizon.
**Access:** primary abstract/indexed article.

### 19. Bittner et al. (2017)

**[Behavioral time scale synaptic plasticity underlies CA1 place fields](https://pubmed.ncbi.nlm.nih.gov/28883072/)**. Science.
**Evidence type:** In vivo CA1 and slice physiology.
**Supported observation:** Plateau-associated plasticity can form a place field after a single experience and spans seconds.
**Training implication:** Allow rapid episodic weight changes distinct from slow representation training.
**Boundary:** BTSP is not evidence for a universal hippocampal backpropagation rule.
**Access:** primary abstract/indexed article.

### 20. Bellec et al. (2020)

**[A solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y)**. Nature Communications.
**Evidence type:** Original computational learning method.
**Supported observation:** Eligibility traces plus learning signals permit online recurrent spiking-network training.
**Training implication:** Candidate local-learning implementation and comparison to a BPTT baseline.
**Boundary:** A proposed biologically motivated algorithm, not a demonstrated universal brain learning rule.
**Access:** primary abstract/indexed article.

### 21. Turrigiano et al. (1998)

**[Activity-dependent scaling of quantal amplitude in neocortical neurons](https://www.nature.com/articles/36103)**. Nature.
**Evidence type:** Cultured cortical-neuron physiology.
**Supported observation:** Activity perturbations induce compensatory changes in excitatory synaptic strength.
**Training implication:** Include homeostatic stabilization when connecting recurrent modules.
**Boundary:** Culture mechanisms do not uniquely specify a whole-brain stabilization algorithm.
**Access:** primary abstract/indexed article.

### 22. Guo et al. (2017)

**[Maintenance of persistent activity in a frontal thalamocortical loop](https://www.nature.com/articles/nature22324)**. Nature.
**Evidence type:** Mouse recordings and reciprocal photoinhibition.
**Supported observation:** Frontal and motor-thalamic activity depend on reciprocal interactions during a delay.
**Training implication:** Train delay competence as a coupled frontal-thalamic system.
**Boundary:** This is motor preparation in a specific task, not all human working memory.
**Access:** primary abstract/indexed article.

### 23. Schmitt et al. (2017)

**[Thalamic amplification of cortical connectivity sustains attentional control](https://www.nature.com/articles/nature22073)**. Nature.
**Evidence type:** Mouse circuit perturbation and recording.
**Supported observation:** MD supports cortical rule-related interactions rather than simply relaying rule identity.
**Training implication:** Give MD-type partners modulatory/interaction-support interfaces.
**Boundary:** A rule-maintenance circuit is not a global consciousness switch.
**Access:** primary abstract/indexed article.

### 24. Bolkan et al. (2017)

**[Thalamic projections sustain prefrontal activity during working memory maintenance](https://www.nature.com/articles/nn.4568)**. Nature Neuroscience.
**Evidence type:** Mouse projection-specific recording and perturbation.
**Supported observation:** Direction of MD-PFC communication matters differently for retention and subsequent choice.
**Training implication:** Include reciprocal ports and phase-specific tests.
**Boundary:** Anatomical projections should not be collapsed into one undirected memory node.
**Access:** primary abstract/indexed article.

### 25. Hosford et al. (2026)

**[Control of representation updating by higher-order thalamus enables history-based decision-making](https://pubmed.ncbi.nlm.nih.gov/42214337/)**. Neuron.
**Evidence type:** Mouse recordings and optogenetics; online 29 May 2026.
**Supported observation:** PPC-LP interaction stabilizes sensory history; TRN input permits updating by disrupting maintenance.
**Training implication:** Test maintenance and replacement separately within posterior cortical-thalamic loops.
**Boundary:** Mouse LP is a pulvinar homolog; one sensory-history task does not establish a universal write gate.
**Access:** primary abstract/indexed article.

### 26. Andrianova et al. (2025)

**[Hippocampal pyramidal cells of the CA1 region are not a major target of the thalamic nucleus reuniens](https://pubmed.ncbi.nlm.nih.gov/41082527/)**. PLOS Biology.
**Evidence type:** Mouse/rat ex vivo optogenetics, recordings and tracing; 13 October 2025.
**Supported observation:** NRe inputs in hippocampus proper preferentially engage interneurons; CA1 pyramidal input was not evident.
**Training implication:** Preserve target-cell specificity in PFC-NRe-EC/hippocampal interfaces.
**Boundary:** Do not infer absence of all NRe effects on CA1 or remove indirect control.
**Access:** primary abstract/indexed article.

### 27. Moore and Armstrong (2003)

**[Selective gating of visual signals by microstimulation of frontal cortex](https://www.nature.com/articles/nature01341)**. Nature.
**Evidence type:** Macaque FEF stimulation and V4 recording.
**Supported observation:** Subthreshold stimulation modifies matching visual responses.
**Training implication:** Reserve top-down feature/location gain ports from the first visual checkpoint.
**Boundary:** The experiment does not prescribe an exact gating algorithm.
**Access:** primary abstract/indexed article.

### 28. Atlan et al. (2018)

**[The Claustrum Supports Resilience to Distraction](https://www.sciencedirect.com/science/article/pii/S0960982218308595)**. Current Biology.
**Evidence type:** Mouse genetic access, perturbation and behavior.
**Supported observation:** Claustral inhibition increases susceptibility to auditory distractors; activation suppresses auditory representations.
**Training implication:** Include a later testable distraction-control branch.
**Boundary:** Do not make the claustrum a mandatory global integrator on this evidence.
**Access:** primary abstract/indexed article.

### 29. Nakazawa et al. (2002)

**[Requirement for hippocampal CA3 NMDA receptors in associative memory recall](https://pubmed.ncbi.nlm.nih.gov/12040087/)**. Science.
**Evidence type:** Mouse targeted receptor manipulation.
**Supported observation:** CA3 plasticity contributes to recall from incomplete spatial cues.
**Training implication:** Train rapid association and degraded-cue retrieval together.
**Boundary:** The task does not demonstrate all details of a human episodic memory algorithm.
**Access:** primary abstract/indexed article.

### 30. Kohara et al. (2014)

**[Cell type-specific genetic and optogenetic tools reveal hippocampal CA2 circuits](https://www.nature.com/articles/nn.3614)**. Nature Neuroscience.
**Evidence type:** Mouse tracing and optogenetic circuit physiology.
**Supported observation:** CA2 receives direct EC II and DG inputs and preferentially influences deep CA1.
**Training implication:** Preserve CA2 as a parallel branch in the memory training group.
**Boundary:** Do not put CA2 in a required CA3-to-CA1 serial chain.
**Access:** primary abstract/indexed article.

### 31. Horner et al. (2015)

**[Evidence for holistic episodic recollection via hippocampal pattern completion](https://www.nature.com/articles/ncomms8462)**. Nature Communications.
**Evidence type:** Human behavioral and fMRI evidence.
**Supported observation:** Partial event retrieval is associated with reinstatement of related cortical content.
**Training implication:** Acceptance requires useful downstream reinstatement, not only a memory-index output.
**Boundary:** Human fMRI does not identify individual CA3 recurrent synapses.
**Access:** primary abstract/indexed article.

### 32. Waye et al. (2019)

**[Lesions to the lateral mammillary nuclei disrupt spatial learning in rats](https://pubmed.ncbi.nlm.nih.gov/31647251/)**. Behavioral Neuroscience.
**Evidence type:** Rat lesion and navigation experiments.
**Supported observation:** Tegmentomammillary direction circuitry contributes to directional navigation.
**Training implication:** Bootstrap directional state before complex route planning.
**Boundary:** Do not equate a directional code with a complete allocentric position map.
**Access:** primary abstract/indexed article.

### 33. Azim et al. (2014)

**[Skilled reaching relies on a V2a propriospinal internal copy circuit](https://www.nature.com/articles/nature13021)**. Nature.
**Evidence type:** Mouse tracing, ablation and optogenetics.
**Supported observation:** Spinal premotor output branches toward motor neurons and lateral reticular/cerebellar circuitry.
**Training implication:** Include spinal-level internal copies, not only cortical action copies.
**Boundary:** Exact primate implementation and all reaching corrections are not specified.
**Access:** primary abstract/indexed article.

### 34. Herzfeld et al. (2018)

**[Encoding of error and learning to correct that error by the Purkinje cells of the cerebellum](https://www.nature.com/articles/s41593-018-0136-y)**. Nature Neuroscience.
**Evidence type:** Primate oculomotor recordings.
**Supported observation:** Complex-spike error responses relate to later simple-spike and movement changes.
**Training implication:** Train prediction/adaptation with actual delayed sensory consequences.
**Boundary:** This does not identify every cerebellar teaching signal as the same error.
**Access:** primary abstract/indexed article.

### 35. Steinberg et al. (2013)

**[A causal link between prediction errors, dopamine neurons and learning](https://www.nature.com/articles/nn.3413)**. Nature Neuroscience.
**Evidence type:** Rat optogenetic conditioning experiments.
**Supported observation:** Timed dopamine activation changes cue-reward learning.
**Training implication:** Introduce simple outcome-learning loops early.
**Boundary:** A scalar RL optimizer is an engineering approximation, not the entire dopamine system.
**Access:** primary abstract/indexed article.

### 36. Livneh et al. (2017)

**[Homeostatic circuits selectively gate food cue responses in insular cortex](https://www.nature.com/articles/nature22375)**. Nature.
**Evidence type:** Mouse imaging and pathway manipulation.
**Supported observation:** Hypothalamic need changes insular cue responses through PVT and BLA circuitry.
**Training implication:** Represent needs from the outset and calibrate their sensory effects later.
**Boundary:** Do not generalize this exact pathway to every motivation.
**Access:** primary abstract/indexed article.

### 37. Saffran, Aslin and Newport (1996)

**[Statistical learning by 8-month-old infants](https://pubmed.ncbi.nlm.nih.gov/8943209/)**. Science.
**Evidence type:** Human infant behavioral experiment.
**Supported observation:** Infants use speech transition statistics after brief exposure.
**Training implication:** Begin auditory sequence learning before grounded language instruction.
**Boundary:** Word segmentation is not full semantic or grammatical competence.
**Access:** primary abstract/indexed article.

### 38. MuJoCo documentation (2026)

**[XML Reference: sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html)**. Official documentation.
**Evidence type:** Official simulator API specification accessed 12 September 2026.
**Supported observation:** Distinguishes touch, site-frame accelerometer/gyro and privileged frame quantities.
**Training implication:** Expose receptor-like observations; keep object poses and world-frame state evaluator-only.
**Boundary:** An IMU channel requires a receptor-dynamics model to approximate biological vestibular sensing.
**Access:** primary abstract/indexed article.

### 39. Mattern et al. (2023)

**[MIMo: A Multi-Modal Infant Model for Studying Cognitive Development](https://arxiv.org/abs/2312.04318)**. Original research preprint.
**Evidence type:** Embodied simulator design and examples.
**Supported observation:** MIMo combines binocular, proprioceptive, tactile and vestibular channels with two actuation models.
**Training implication:** Use a common embodied sensorimotor substrate for all checkpoints.
**Boundary:** The simulator supplies a body, not a validated neural developmental curriculum.
**Access:** primary abstract/indexed article.

### 40. Philipp, Lopez and Triesch (2026)

**[Embodiment Shapes Rolling Behavior in a Multimodal Infant Model](https://arxiv.org/abs/2606.17456)**. Original research preprint.
**Evidence type:** Computational study; posted 16 June 2026.
**Supported observation:** MIMo rolling learned with proprioception and vestibular sensing varies with body morphology.
**Training implication:** Motor development can begin without mature object vision; evaluate morphology changes.
**Boundary:** Preprint simulation and its RL choices do not establish biological neural learning.
**Access:** primary abstract/indexed article.

### 41. Redinbaugh et al. (2020)

**[Thalamus Modulates Consciousness via Layer-Specific Control of Cortex](https://pubmed.ncbi.nlm.nih.gov/32053769/)**. Neuron.
**Evidence type:** Macaque anesthesia, recordings and central-lateral stimulation.
**Supported observation:** Thalamic stimulation restores arousal-related behavior and cortical dynamics.
**Training implication:** Treat operating state and state-dependent recurrence as cross-cutting and testable.
**Boundary:** No guarantee of artificial subjective experience follows.
**Access:** primary abstract/indexed article.

### 42. Cogitate Consortium et al. (2025)

**[Adversarial testing of global neuronal workspace and integrated information theories of consciousness](https://www.nature.com/articles/s41586-025-08888-1)**. Nature.
**Evidence type:** Preregistered human multimethod comparison; 30 April 2025.
**Supported observation:** Specific predictions of both theories were challenged by the results.
**Training implication:** Use multiple independent functional and perturbational endpoints; avoid training one theory signature as proof.
**Boundary:** This does not resolve necessary and sufficient conditions for artificial experience.
**Access:** primary abstract/indexed article.

### 43. Talpalar et al. (2013)

**[Dual-mode operation of neuronal networks involved in left-right alternation](https://www.nature.com/articles/nature12286)**. Nature.
**Evidence type:** Mouse genetic perturbation and locomotor physiology.
**Supported observation:** Different spinal V0 populations support left-right alternation at different speeds.
**Training implication:** Provide spinal coordination before voluntary visual action; test across locomotor speeds.
**Boundary:** Does not identify the correct human spinal-controller architecture or its complete learning rule.
**Access:** primary abstract/indexed article.

### 44. Rust et al. (2006)

**[How MT cells analyze the motion of visual patterns](https://www.nature.com/articles/nn1786)**. Nature Neuroscience.
**Evidence type:** Macaque recordings and fitted computational model.
**Supported observation:** A nonlinear transformation of V1 population responses predicts MT grating and plaid responses; convergence, opponency and normalization matter.
**Training implication:** Train motion pooling from actual temporal V1 activity, retaining spatial and temporal structure.
**Boundary:** A fitted response model is not evidence that the brain uses the proposed training optimizer.
**Access:** Primary publisher abstract and indexed article metadata.

### 45. Kar et al. (2019)

**[Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior](https://www.nature.com/articles/s41593-019-0392-5)**. Nature Neuroscience.
**Evidence type:** Primate behavior, neural recording and model comparison.
**Supported observation:** Challenging object images require later inferotemporal processing; recurrent models better capture relevant responses.
**Training implication:** Release basic object identity before difficult recurrent recognition; preserve return paths.
**Boundary:** Does not isolate every anatomical feedback edge or a unique learning algorithm.
**Access:** primary abstract/indexed article.

### 46. Sommer and Wurtz (2006)

**[Influence of the thalamus on spatial visual processing in frontal cortex](https://www.nature.com/articles/nature05279)**. Nature.
**Evidence type:** Macaque pathway perturbation and electrophysiology.
**Supported observation:** Superior-colliculus to mediodorsal-thalamus to frontal-eye-field corollary discharge contributes to presaccadic remapping.
**Training implication:** Supply copies of issued eye commands and train on their real sensory consequences.
**Boundary:** One contributor to visual continuity, not a complete world-coordinate system.
**Access:** primary abstract/indexed article.

### 47. Buneo et al. (2002)

**[Direct visuomotor transformations for reaching](https://www.nature.com/articles/416632a)**. Nature.
**Evidence type:** Macaque posterior-parietal recording.
**Supported observation:** Target and hand relationships are represented in forms supporting visuomotor transformations.
**Training implication:** Keep spatially organized visual output and proprioceptive/gaze information available to reach learning.
**Boundary:** Recording does not establish a unique arithmetic implementation.
**Access:** primary abstract/indexed article.

### 48. Bichot et al. (2015)

**[A Source for Feature-Based Attention in the Prefrontal Cortex](https://pubmed.ncbi.nlm.nih.gov/26526392/)**. Neuron.
**Evidence type:** Macaque recording and inactivation.
**Supported observation:** Ventral prearcuate prefrontal circuitry contributes to feature-based visual search and frontal-eye-field selection.
**Training implication:** Train a retained feature template and its effects on visual sampling; later close the feedback loop.
**Boundary:** Macaque areal naming is not an exact human atlas correspondence.
**Access:** primary abstract/indexed article.

### 49. Girardeau et al. (2009)

**[Selective suppression of hippocampal ripples impairs spatial memory](https://www.nature.com/articles/nn.2384)**. Nature Neuroscience.
**Evidence type:** Rat post-training ripple disruption.
**Supported observation:** Disrupting sharp-wave ripples impairs later spatial memory.
**Training implication:** Add replay after the first useful associative-memory stage, not only after complete cognition.
**Boundary:** Does not prescribe which replay-selection algorithm a model should use.
**Access:** primary abstract/indexed article.

### 50. Kitamura et al. (2017)

**[Engrams and circuits crucial for systems consolidation of a memory](https://doi.org/10.1126/science.aam6808)**. Science.
**Evidence type:** Mouse engram tagging and circuit manipulation.
**Supported observation:** Cortical engrams form during initial learning and mature with hippocampal support.
**Training implication:** Allow early cortical learning and later hippocampal-cortical replay rather than waiting to train cortex until memory is complete.
**Boundary:** Contextual-memory paradigm; no universal fixed transfer schedule.
**Access:** Primary author-institution abstract.

### 51. Matsumoto and Hikosaka (2007)

**[Lateral habenula as a source of negative reward signals in dopamine neurons](https://www.nature.com/articles/nature05860)**. Nature.
**Evidence type:** Primate recordings and stimulation.
**Supported observation:** Lateral habenula activity relates to negative reward outcomes and can suppress dopamine-neuron activity through the circuit.
**Training implication:** Include negative-outcome learning channels rather than treating all outcomes as a single positive reward pulse.
**Boundary:** This result does not mean habenular axons are themselves inhibitory or that all dopamine is a scalar reward error.
**Access:** primary abstract/indexed article.

### 52. Schuck et al. (2016)

**[Human Orbitofrontal Cortex Represents a Cognitive Map of State Space](https://pubmed.ncbi.nlm.nih.gov/27657452/)**. Neuron.
**Evidence type:** Human fMRI decoding.
**Supported observation:** Orbitofrontal activity represents task-relevant hidden-state information.
**Training implication:** Train inferred task state after basic observation history is retainable, before complex outcome-dependent planning.
**Boundary:** A representational finding, not proof of a particular planning algorithm.
**Access:** primary abstract/indexed article.

### 53. Ran et al. (2022)

**[A brainstem map for visceral sensations](https://www.nature.com/articles/s41586-022-05139-5)**. Nature.
**Evidence type:** Mouse brainstem imaging and perturbation.
**Supported observation:** Nucleus of the solitary tract contains structured representations of visceral-organ inputs, refined by inhibition.
**Training implication:** Model interoceptive transduction and brainstem representation from the start if bodily regulation is part of the agent.
**Boundary:** The proposed artificial physiology remains a declared environmental model.
**Access:** primary abstract/indexed article.

### 54. Adamantidis et al. (2007)

**[Neural substrates of awakening probed with optogenetic control of hypocretin neurons](https://www.nature.com/articles/nature06310)**. Nature.
**Evidence type:** Mouse optogenetic intervention.
**Supported observation:** Hypocretin-neuron activation increases transitions to wakefulness.
**Training implication:** Include operating-state control early, then test integrated state-dependent responsiveness later.
**Boundary:** Not evidence that stimulating an artificial unit induces subjective experience.
**Access:** primary abstract/indexed article.

### 55. Uematsu et al. (2017)

**[Modular organization of the brainstem noradrenaline system coordinates opposing learning states](https://www.nature.com/articles/nn.4642)**. Nature Neuroscience.
**Evidence type:** Rat projection-specific recordings and interventions.
**Supported observation:** Locus-coeruleus pathways have differentiated relationships to learning states.
**Training implication:** Use target-specific modulation and learned contingencies, not one global arousal multiplier.
**Boundary:** Functional mapping and targets depend on task and species.
**Access:** primary abstract/indexed article.

### 56. Pinto et al. (2013)

**[Fast modulation of visual perception by basal forebrain cholinergic neurons](https://www.nature.com/articles/nn.3552)**. Nature Neuroscience.
**Evidence type:** Mouse optogenetics and cortical recording.
**Supported observation:** Manipulating cholinergic neurons or their visual cortical input rapidly changes visual responses and discrimination.
**Training implication:** Expose gain/plasticity interfaces in sensory modules before training their contextual controllers.
**Boundary:** One neuromodulator cannot be assigned a single identical effect at all targets.
**Access:** primary abstract/indexed article.

### 57. Miyazaki et al. (2014)

**[Optogenetic Activation of Dorsal Raphe Serotonin Neurons Enhances Patience for Future Rewards](https://pubmed.ncbi.nlm.nih.gov/25155504/)**. Current Biology.
**Evidence type:** Mouse optogenetic waiting task.
**Supported observation:** Dorsal-raphe serotonin activation increases waiting in the tested reward task.
**Training implication:** Test delayed outcome and persistence modulation separately from visual perception and immediate motor control.
**Boundary:** Not a universal equation of serotonin with patience or a specification of all raphe functions.
**Access:** primary abstract/indexed article.

### 58. Fife et al. (2017)

**[Causal role for the subthalamic nucleus in interrupting behavior](https://elifesciences.org/articles/27689)**. eLife.
**Evidence type:** Mouse optogenetic perturbation.
**Supported observation:** STN activation pauses ongoing behavior; suppression reduces surprise-related interruption.
**Training implication:** Add interruption tests to the action-regulation loop after basic actions are available.
**Boundary:** Licking interruption does not establish a universal winner-take-all selection circuit.
**Access:** primary abstract/indexed article.

### 59. Fogassi et al. (2001)

**[Cortical mechanism for the visual guidance of hand grasping movements in the monkey](https://academic.oup.com/brain/article/124/3/571/334371)**. Brain.
**Evidence type:** Macaque cortical inactivation.
**Supported observation:** Ventral-premotor F5 inactivation disrupts visual hand preshaping, whereas M1 inactivation has a different finger-movement deficit.
**Training implication:** Separate grasp specification from descending motor execution, then train their actual closed-loop interaction.
**Boundary:** Different F5 subsectors and tasks produce different effects.
**Access:** primary abstract/indexed article.

### 60. Baldassano et al. (2017)

**[Discovering Event Structure in Continuous Narrative Perception and Memory](https://doi.org/10.1016/j.neuron.2017.06.041)**. Neuron.
**Evidence type:** Human naturalistic fMRI and event-model analysis.
**Supported observation:** Association regions exhibit longer event timescales than sensory regions; event boundaries relate to hippocampal activity and later reinstatement.
**Training implication:** Train event continuity and segmentation from learned sensory and episodic representations.
**Boundary:** Event states are inferred functional representations, not literal symbolic registers.
**Access:** primary abstract/indexed article.

### 61. DiNicola, Braga and Buckner (2020)

**[Parallel distributed networks dissociate episodic and social functions within the individual](https://doi.org/10.1152/jn.00529.2019)**. Journal of Neurophysiology.
**Evidence type:** Precision human fMRI.
**Supported observation:** Interdigitated distributed networks show differentiated episodic and social affiliations.
**Training implication:** Build event-construction and social-inference branches rather than one terminal DMN network.
**Boundary:** Functional coupling does not establish direct monosynaptic connectivity or necessity of every node.
**Access:** primary abstract/indexed article.

### 62. Fleming et al. (2014)

**[Domain-specific impairment in metacognitive accuracy following anterior prefrontal lesions](https://pubmed.ncbi.nlm.nih.gov/25100039/)**. Brain.
**Evidence type:** Human focal-lesion comparison.
**Supported observation:** Perceptual metacognitive accuracy can be impaired despite relatively preserved objective performance.
**Training implication:** Evaluate confidence calibration and its use separately from discrimination performance.
**Boundary:** Does not locate all metacognition in one prefrontal module.
**Access:** primary abstract/indexed article.

### 63. Kiani and Shadlen (2009)

**[Representation of confidence associated with a decision by neurons in the parietal cortex](https://pubmed.ncbi.nlm.nih.gov/19423820/)**. Science.
**Evidence type:** Macaque motion decision and opt-out behavior with recording.
**Supported observation:** Parietal decision-related activity carries information associated with confidence.
**Training implication:** Begin uncertainty/confidence-sensitive behavior on simple tasks; test broad metacognitive transfer later.
**Boundary:** A task-specific neural correlate, not a consciousness diagnostic.
**Access:** primary abstract/indexed article.

### 64. Kuhl, Tsao and Liu (2003)

**[Foreign-language experience in infancy: Effects of short-term exposure and social interaction on phonetic learning](https://pubmed.ncbi.nlm.nih.gov/12861072/)**. PNAS.
**Evidence type:** Human infant live-versus-recorded exposure experiment.
**Supported observation:** Live social exposure supported phonetic learning in the tested paradigm more than recorded exposure.
**Training implication:** After acoustic statistical learning, add contingent social interaction to the grounded language curriculum.
**Boundary:** Does not establish that every kind of speech learning requires live partners.
**Access:** primary abstract/indexed article.

### 65. Alexander and Nitz (2015)

**[Retrosplenial cortex maps the conjunction of internal and external spaces](https://www.nature.com/articles/nn.4058)**. Nature Neuroscience.
**Evidence type:** Rat retrosplenial single-unit recording.
**Supported observation:** RSC combines information about route progress, movement and environmental position.
**Training implication:** Calibrate route/landmark/egocentric relations using actual self-motion and sensory history.
**Boundary:** Not proof that every RSC neuron performs one universal coordinate conversion.
**Access:** primary abstract/indexed article.

### 66. Luck and Vogel (1997)

**[The capacity of visual working memory for features and conjunctions](https://www.nature.com/articles/36846)**. Nature.
**Evidence type:** Human change-detection behavior.
**Supported observation:** Classic simple-object tasks show capacity of approximately three to four objects.
**Training implication:** Measure load-dependent retention, interference and precision; use human curves as comparative constraints.
**Boundary:** Do not hard-code a universal four-item buffer across all modalities and expertise.
**Access:** primary abstract/indexed article.

### 67. Hitti and Siegelbaum (2014)

**[The hippocampal CA2 region is essential for social memory](https://www.nature.com/articles/nature13028)**. Nature.
**Evidence type:** Mouse region-specific perturbation.
**Supported observation:** CA2 inactivation impairs social recognition in the tested paradigms.
**Training implication:** Add CA2-focused social-association tests when an agent-identity stream is available.
**Boundary:** Not an obligatory precursor to nonsocial episodic memory or an exclusively social function in humans.
**Access:** primary abstract/indexed article.

### 68. Campos et al. (2018)

**[Encoding of danger by parabrachial CGRP neurons](https://www.nature.com/articles/nature25511)**. Nature.
**Evidence type:** Mouse recordings and cell-population manipulation.
**Supported observation:** Parabrachial CGRP populations respond to diverse noxious and visceral signals and contribute to conditioned defensive responses.
**Training implication:** Keep bodily danger signals and learned cue significance connected from the early curriculum.
**Boundary:** Cell-population function is not identical to necessity of the CGRP peptide or all forms of pain.
**Access:** primary abstract/indexed article.

### 69. Saalmann et al. (2012)

**[The pulvinar regulates information transmission between cortical areas based on attention demands](https://pubmed.ncbi.nlm.nih.gov/22879517/)**. Science.
**Evidence type:** Macaque tractography and simultaneous cortical-thalamic recordings.
**Supported observation:** Pulvinar activity and interareal synchronization vary with attentional allocation.
**Training implication:** Test routing with the receiving cortical partners running; distinguish feedforward information from contextual transmission.
**Boundary:** Correlated directional/synchrony evidence is not proof of a universal communication protocol.
**Access:** primary abstract/indexed article.

### 70. Hyde and Knudsen (2002)

**[The optic tectum controls visually guided adaptive plasticity in the owl's auditory space map](https://www.nature.com/articles/415073a)**. Nature.
**Evidence type:** Barn-owl targeted tectal lesions during sensory recalibration.
**Supported observation:** A visual-map lesion prevents corresponding adaptive changes in the auditory space map.
**Training implication:** Distinguish basic auditory feature learning from later crossmodal spatial calibration.
**Boundary:** Comparative bird evidence; do not transplant owl-specific nucleus organization into a primate model.
**Access:** primary abstract/indexed article.

### 71. Linkenhoker and Knudsen (2002)

**[Incremental training increases the plasticity of the auditory space map in adult barn owls](https://www.nature.com/articles/nature01002)**. Nature.
**Evidence type:** Barn-owl graded prism adaptation.
**Supported observation:** Gradual mismatch exposure permits greater adult auditory-map recalibration than a single large mismatch.
**Training implication:** Introduce new feedback or body/sensor changes gradually and retain prior-task tests.
**Boundary:** Supports the principle of graded calibration, not a universal step size or an ANN optimization guarantee.
**Access:** primary abstract/indexed article.

### 72. Goodale et al. (1991)

**[A neurological dissociation between perceiving objects and grasping them](https://www.nature.com/articles/349154a0)**. Nature.
**Evidence type:** Human neuropsychological case with quantitative grasping tests.
**Supported observation:** Accurate visually guided hand/finger movements were preserved despite severe impairment in reporting relevant object qualities.
**Training implication:** Do not require complete ventral recognition or semantic identity before basic visually guided grasping.
**Boundary:** Adult dissociation does not by itself establish the complete developmental acquisition sequence.
**Access:** Primary publisher abstract.
