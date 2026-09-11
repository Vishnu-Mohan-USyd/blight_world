# Perception: computational architecture and information flow

## 1. The architecture to reproduce

A useful computational decomposition separates **sensory state estimation, temporary retention, relational retrieval, contextual interpretation, task control, and action generation**. These processes operate at different timescales and communicate through specific anatomical pathways. Their interaction explains how a changing pattern of stimulation becomes a stable, task-relevant situation that can guide behavior.

The diagram below describes the principal dependencies. Arrows designate information or influence, not an obligatory sequence in which one stage must finish before another starts.

```text
External world and body
        |
        v
Modality-specific pathways <-----------------------------+
        |                                               |
        v                                               |
Recurrent sensory cortex <---- attentional bias ---------|
        |                                               |
        +---- object / feature representations          |
        +---- spatial / body / action representations   |
        |                  |                            |
        +----------+-------+                            |
                   |                                    |
          association cortex                            |
          /        |        \                           |
         v         v         v                          |
  semantic     temporary    event / spatial             |
  knowledge    retention    context                     |
         \         |         /                          |
          \        |        /                           |
         entorhinal-hippocampal loops                    |
         rapid relational binding and retrieval         |
                   |                                    |
         retrieved objects, relations, events           |
                   |                                    |
                   v                                    |
      task state and candidate actions                  |
      prefrontal-parietal-thalamic-striatal loops        |
                   |                                    |
                   +---- sensory selection -------------+
                   |
                   v
      parietal-premotor-motor systems
                   |
          predicted consequences <--> cerebellar loops
                   |
                   v
      eye, head, hand and whole-body actions
                   |
                   +---- changed sensory input
```

Bodily needs, arousal and learned outcome values influence several of these loops simultaneously. Anatomical tracing supports differentiated cortical pathways, thalamocortical loops and hippocampal input-output routes; causal experiments demonstrate interactions such as frontal modulation of visual responses and thalamic support for retained task states.[^1][^2][^3][^4][^5]

**The central design distinction is between content and control.** A representation of a watch contains information useful for recognizing or locating it. A search goal changes which features, locations and memories receive priority. A retained representation permits processing after its stimulus disappears. A learning update changes what a future encounter will evoke. These are different operations even when their neural substrates overlap.

In this reference, anatomical pathways and experimentally demonstrated effects are stated as biological findings. Terms such as *context state*, *retrieval key*, *priority map* and *read/write gate* describe computations. The proposed neural-network interfaces are explicitly design abstractions constrained by those findings.

## 2. How connected circuits exchange and retain information

### 2.1 Anatomical connectivity constrains the computation

Cortical connections have directionally and laminarily structured origins and targets. In sensory hierarchies, ascending and descending projections have different laminar distributions; a simplified implementation should therefore distinguish bottom-up evidence from recurrent/contextual influence rather than concatenate every input indiscriminately. The anatomy is richer than a strict layer-by-layer hierarchy: lateral and bypass connections permit several processing routes.[^1]

A new goal need not change anatomical wiring to change processing. It can change the activity arriving over existing connections, the gain of recipient populations, the balance of excitation and inhibition, or the relative influence of competing representations. Subthreshold frontal-eye-field stimulation changes corresponding V4 responses, and mediodorsal thalamic input supports prefrontal task representations by altering effective cortical interactions.[^6][^4]

A concrete instance is searching for a familiar watch. Its remembered features become relevant inputs to recognition and selection circuits. Those inputs bias which currently stimulated populations are amplified. The visual scene still supplies the evidence; task-related activity changes which parts of that evidence dominate subsequent processing. This interpretation combines the causal frontal-visual result with demonstrations of biased competition in V2/V4.[^6][^7]

### 2.2 Three timescales should remain distinct

**Current activity** represents immediately available information and evolving computations. **Retained task state** preserves selected information after its input disappears, through recurrent dynamics and potentially latent changes that can later be reactivated. **Longer-lasting synaptic change** alters the responses and associations available on future encounters. A model can implement these with different mechanisms; treating all three as one hidden vector with one decay constant would obscure important experimental dissociations.[^8][^9][^3][^10]

The same population can combine several variables. Prefrontal mixed selectivity means that activity can depend jointly on stimulus, task rule and remembered sequence. Consequently, a biological module is better approximated by a task-dependent population code than by one dedicated neuron or register per psychological noun.[^11]

## 3. Constructing sensory and bodily state

### 3.1 Vision: transform local stimulation into identity and action-relevant geometry

The major cortical route begins with retinal output reaching the lateral geniculate nucleus, then V1 and interconnected extrastriate areas. The following functional decomposition captures the information needed downstream:

| Processing level | Main information produced | Why later processing needs it |
|---|---|---|
| Retinal/LGN/early cortical processing | Spatially organized contrast, temporal change, local orientation and binocular relationships | Establishes evidence tied to the current eye position and image quality |
| Recurrent early and intermediate visual processing | Grouped boundaries, surfaces and figure-ground structure | Distinguishes an object's parts from nearby clutter |
| Ventral occipitotemporal processing, including V4 and inferotemporal cortex | Shape and object-related descriptions useful for recognition | Relates the current view to familiar categories and particular objects |
| MT and associated motion pathways | Direction and motion-related structure | Supports judgments about object motion and movement through the scene |
| Posterior parietal and parietal-premotor processing | Spatial relationships relevant to orienting, reaching and grasping | Converts visual evidence into actionable targets relative to the eyes and body |

The strongest constraints are not just activation maps: MT stimulation biases motion judgments; recognition and visually guided grasping can dissociate after injury; and AIP has anatomical access to visual, somatosensory and premotor information.[^12][^13][^2]

Two transformations matter computationally. First, an object must remain recognizable when its retinal appearance changes. Second, a recognized object must be related to a body-centered action: its identity alone does not specify how to reach it. Recognition and action geometry are therefore complementary representations, with communication between their pathways rather than a requirement that one completely solve the other's problem.[^13][^2]

Recurrent processing contributes contextual organization. In macaque V1, the initial visually driven response and later figure-ground modulation have different pharmacological dependencies. For a model, this supports allowing an initial local interpretation to be revised by evidence about surrounding structure instead of making recognition a single irreversible forward pass.[^14]

### 3.2 Hearing: retain temporal order while separating sources

Auditory processing starts with frequency- and time-structured signals and proceeds through brainstem/midbrain, thalamic and cortical pathways. At the cortical association level, anterior and caudal auditory pathways have differentiated prefrontal targets. A useful functional decomposition distinguishes sound/source identity from spatial and action-related information.[^15]

Unlike a static image, the evidence for a spoken word or sentence arrives sequentially. The system must retain enough earlier information to interpret later information, while separating a relevant voice from competing sounds. Human intracranial recordings show that attention already changes the speech representation in nonprimary auditory cortex. Thus source selection is part of constructing the auditory representation, not only choosing an answer after an unchanged acoustic recording has been processed.[^16]

The subsequent interface is between auditory/phonological representations, conceptual knowledge and the maintained conversational situation. The sound sequence, the meaning of a word and the event to which a pronoun refers are different retained variables. Frontal-temporoparietal verbal retention and anterior-temporal semantic contributions supply experimentally distinguishable parts of this architecture.[^17][^18]

### 3.3 Touch and proprioception: estimate contacts and the configuration of the body

Somatosensory processing should be implemented as parallel, convergent pathways. Anatomical work in macaques shows multiple thalamic inputs to multiple parietal somatosensory fields. Areas 3a, 3b, 1, 2 and neighboring regions do not form a single obligatory serial chain.[^19]

The computational outputs include cutaneous contact, changes in pressure or vibration, and information relevant to limb configuration and movement. Their interaction with visual and parietal-premotor signals allows a hand to be treated both as part of the body and as an effector contacting an external object. AIP's connections with somatosensory, visual and premotor territories provide an anatomical route for combining these constraints during grasp-related behavior.[^19][^2]

A useful implementation maintains a body-state estimate separately from object identity. It must track which changes are attributable to the body moving, which indicate an external object moving, and which indicate a new contact. The representation need not be a single explicit three-dimensional body model, but it must preserve the relationships needed for these distinctions.

### 3.4 Self-motion, smell and taste add different constraints

Vestibular signals constrain self-motion together with visual motion. Experiments in humans and macaques show that their relative contribution changes with reliability. The useful output is a self-motion estimate whose dependence on each sensory stream can change when that stream becomes unreliable.[^20]

Olfaction provides a particularly concrete example of recurrent transformation. Olfactory-bulb responses vary with odor concentration; recurrent piriform circuitry recruits inhibition that supports a more concentration-stable identity representation. Disrupting recurrence impairs this transformation. Separate experiments show that recurrence also stabilizes piriform representations when upstream odor responses are degraded.[^21][^22]

Taste and smell then contribute to combined flavor representations. Human imaging identifies convergence and interactions in insular and orbitofrontal regions, with differentiated relationships to pleasantness. A model should preserve the identity-related sensory information while allowing it to interact with state-dependent value, rather than reduce flavor to a single intensity variable.[^23]

### 3.5 Multisensory integration requires assignment as well as averaging

There are two computational questions: **Do the signals refer to the same event?** And, when they do, **how strongly should each influence the estimate?** A visible impact and a sound should be fused differently depending on their spatial and temporal compatibility. Behavioral and imaging studies support common-cause inference as an important part of audiovisual integration.[^24][^25]

Once cues are treated as belonging together, their reliability matters. Human visual-haptic judgments can weight cues according to reliability. For the watch search, this implies that faint visual evidence and direct tactile contact should not contribute equally to localization or identification. Applying that principle to the scenario is a computational extrapolation; the underlying reliability-dependent integration is experimentally supported.[^26]

### 3.6 Actions alter the sensory coordinate system

An eye movement displaces objects on the retina without necessarily moving them in the world. The system therefore requires information about its own movement, not just successive images. A superior-colliculus-to-mediodorsal-thalamus-to-frontal-eye-field pathway carries corollary-discharge information that contributes to predictive changes in visual processing.[^27]

For implementation, distinguish **sensory evidence**, **estimated body/eye state**, and **a copy or prediction of the intended movement**. Otherwise a visual network must rediscover from image differences alone whether the object moved or the camera moved. The same design distinction is important when modeling the sensory consequences of hand and body actions.

## 4. Memory is a set of retention and retrieval mechanisms

### 4.1 Sensory persistence is not the same capacity limit as working memory

| Representation | What remains available | Experimentally grounded constraint |
|---|---|---|
| Brief visual sensory persistence | A relatively broad, rapidly decaying trace of recent visual stimulation | Classic partial-report advantages decay over a few hundred milliseconds; this is not stable access to every object |
| Brief auditory persistence | Recent acoustic information extending beyond the physical sound | Partial-report experiments demonstrate persistence on roughly a one-second timescale in particular paradigms; duration depends on stimulus and measurement |
| Visual working memory | Selected features, objects or locations needed after their disappearance | Around three to four simple objects in classic change-detection tasks; continuous precision and priority reveal important additional limits |
| Verbal/phonological retention | Ordered speech-related representations, often refreshed through rehearsal | Span depends on the material, interference and rehearsal; a fixed universal time-based buffer is not established |
| Tactile working memory | Selected vibration/contact features or their sequence | A 2025 frequency-sequencing study estimated roughly three frequency units; this is not a limit for every tactile representation |
| Task/rule retention | Goals, relevant rules, prospective actions and selection priorities | Uses distributed frontal-parietal-thalamic mechanisms and can compete with other control demands |

These constraints come from different tasks and should not be converted into interchangeable numerical slots.[^28][^29][^30][^31][^32][^33][^34][^3]

For the watch search, the current view can contain more detail than can be retained precisely after turning away. At the same time, the person can retain the search goal and activate familiar knowledge of the room. A limit on reporting the colors of briefly displayed objects is not a limit on all those forms of information combined.

### 4.2 Cross-modal capacity depends on the representation and the task

Experiments with masked visual and auditory arrays found a shared limit under their testing conditions. Other experiments, designed to reduce coordination demands and shared coding, found much more independent visual and auditory storage. Interference becomes stronger when both tasks compete for a common representational format, such as spatial information, or require coordinated selection.[^35][^36][^37]

The architectural implication is **modality- and representation-sensitive retention coupled to shared control demands**. A model should have ways to preserve visual detail, sound sequence and bodily state without treating them as interchangeable tokens, while also modeling competition for prioritization, updating and action selection. This is a synthesis of the conflicting results, not a claim that the brain contains exactly one capacity meter for each sense.

### 4.3 Where is a temporarily retained item represented?

Visual information can be decoded from early visual cortical activity during a retention delay. In tactile comparison tasks, prefrontal activity represents the remembered first stimulus, and secondary somatosensory activity participates in the evolving comparison with the second. These findings connect retained content to the circuits needed to use it.[^8][^38][^39]

The tactile example gives an explicit computational sequence:

```text
first vibration f1
    -> somatosensory encoding
    -> retained population state carrying information about f1
second vibration f2
    -> current sensory representation
    -> interaction with retained f1 in comparison-related circuitry
    -> decision-related population activity
    -> selected motor response
```

The memory is useful because it remains accessible to a comparison operation after the original stimulation ends. Its biological implementation includes distributed activity rather than a separate storage box that waits outside the processing system.[^38][^39]

There are also latent states: an item that is weakly expressed in ongoing measurements can later become decodable or behaviorally influential again after reactivation. Conversely, recent dense recordings in macaque prefrontal and parietal cortex show population persistence during working-memory maintenance. These findings justify distinguishing **currently expressed information** from **retained availability**, without requiring either continuous spiking or completely silent storage for every memory.[^9][^40]

Thalamocortical loops contribute to retention. Perturbing components of a frontal-thalamic loop in mice disrupts persistent preparatory activity. The implementation consequence is that stable task information may depend on a recurrent loop spanning modules; it need not be stored wholly inside one cortical recurrent network.[^3]

### 4.4 The hippocampal boundary is not a stopwatch

Whether short-delay performance requires the hippocampus depends on the relational information and load. Some lesion studies find impairments for conjunctions over short intervals; others show preserved low-load performance at short delays and impairment when demands increase. A delay measured in seconds does not identify a single memory mechanism.[^41][^42]

For modeling, ask **what must be bound and retrieved**. Keeping a target orientation active and reconstructing which object occupied which place in which episode are different requirements, even when the test delays happen to be equal.

## 5. Hippocampal architecture: rapid binding, separation and completion

### 5.1 Input organization

Perirhinal and parahippocampal cortices receive differentiated association-cortical inputs. Together with entorhinal cortex, they provide routes by which object-related information, spatial/contextual information and their relationships enter the hippocampal system. Entorhinal recordings contain both object- and place-related information.[^43][^44]

DG is the dentate gyrus. CA1, CA2 and CA3 are anatomically defined fields of the cornu ammonis; the numerals are not an instruction to traverse them in numerical order. The principal excitatory routes relevant to this account are:

```text
Association cortex
     |                  ^
     v                  |
Perirhinal / parahippocampal / entorhinal circuits
     |
     +-- EC layer II -> DG --mossy fibers--> CA3
     |                                      | ^
     |                                      +-+ recurrent associations
     |                                      |
     +-- EC layer II ---------------------> CA3
     |                                      |
     |                               Schaffer collaterals
     |                                      |
     +-- EC layer III --------------------> CA1
     |                                      |
     +-- EC layer II -> CA2 ---------------> CA1
                        ^                   |
                        |                   v
                       DG             subiculum and
                                      entorhinal output routes
                                            |
                                            v
                                  distributed cortical targets
```

The trisynaptic DG-CA3-CA1 route therefore operates alongside direct entorhinal inputs and a CA2 route. Cell-type-specific mouse experiments identify entorhinal layer-II and dentate input to CA2, with CA2 preferentially targeting deep CA1 pyramidal neurons.[^5]

### 5.2 DG: reduce interference between similar episodes

Suppose the watch was on the desk yesterday and on the shelf today. Many event elements overlap: the room, the watch and the intention to leave. A useful new memory must preserve the distinguishing relation rather than merge both events into an undifferentiated average.

DG's sparse representation and the transformation between its inputs and outputs contribute to separating similar patterns. Recordings under cue-conflict conditions support different transformations in DG and CA3, with DG responses providing more differentiated input while CA3 can recover a coherent familiar representation.[^45]

**Implementation role:** map similar combinations of cortical features into sufficiently distinct episode codes to permit later disambiguation. A sparse competitive encoder is a plausible approximation. Sparsity alone is not sufficient: the model must be tested for reduced interference between similar episodes.

### 5.3 CA3: associate elements and recover an event from incomplete evidence

CA3 receives dentate input and direct entorhinal input, and its recurrent connections support interactions among elements of a stored representation. This architecture is well suited to completing a learned pattern from a subset of its cues. Reduced-cue recall is sensitive to CA3 manipulations, and simultaneous DG/CA3 recordings provide direct evidence for completion under particular input conditions.[^46][^45]

**Implementation role:** retain rapidly learned associations and let a partial query recover a coherent candidate event. A model might use a recurrent associative memory, a learned content-addressable memory, or another mechanism with equivalent tested properties. The diagnostic requirement is partial-cue retrieval without collapsing distinct episodes.

During the watch search, a remembered action such as taking it off while charging the phone can cue the associated room location. The useful output is a linked set of relations, not a perfect replay of the entire retinal image.

### 5.4 CA2: a parallel route with differentiated inputs and outputs

CA2 should not be inserted as a compulsory bridge between CA3 and CA1. Its direct entorhinal and dentate inputs allow it to influence CA1 through a parallel route. Selective CA2 inactivation in mice impairs social memory, supporting a differentiated contribution rather than an interchangeable second CA3.[^5][^47]

A CA2-inspired branch would therefore target differentiated relational processing, with social identity and encounters as useful validation tasks. Its direct input and preferential output pathways should be preserved when those are the mechanisms being modeled.

### 5.5 CA1: combine recovered associations with current cortical input

CA1 receives CA3 input and direct entorhinal layer-III input on different dendritic territories. The direct pathway can modulate CA1 spiking and plasticity. This provides a place where a recovered association can interact with information arriving from the current cortical situation.[^48]

The distinction matters when memory suggests that the watch is on the desk but current evidence shows an empty, clearly visible desk. CA1-related processing can participate in producing an updated contextual representation that reflects both streams. Treating CA1 as a literal subtraction unit with output equal to 'prediction minus sensation' is an additional model hypothesis, not what the anatomy establishes.

CA1 also transforms the code rather than merely forwarding CA3 activity. In a 2026 large-environment bat study, CA3 and CA1 differed strongly in spatial coding density and field structure. The result supports differentiated representational transformations, while leaving their precise form dependent on species and task.[^49]

### 5.6 Output and cortical reinstatement

CA1, subiculum and entorhinal circuits provide routes through which hippocampal processing can influence cortical representations. Even the entorhinal return pathway is differentiated: in mouse medial entorhinal cortex, hippocampal input preferentially reaches deep-layer populations distinct from major telencephalic output populations.[^50]

Human experiments connect hippocampal retrieval to reinstatement of associated cortical event information. The resulting computational cycle is:

```text
partial cortical cue
    -> entorhinal-hippocampal retrieval
    -> recovered event relationships
    -> hippocampal/entorhinal output influences cortex
    -> reactivated object, place and contextual information
    -> new attentional target, comparison or action plan
```

That reinstated information can bias present perception or guide a search even when the remembered object is currently invisible.[^51][^52]

Spatial and temporal structure are both relevant. Entorhinal grid-like activity and hippocampal time-cell sequences supply evidence for structured spatial and temporal representations. They allow an episode to contain relationships such as where something happened and how events unfolded, rather than only an unordered collection of object identities.[^53][^54]

## 6. Constructing the situation: semantics, context and event continuity

### 6.1 Familiar knowledge and a particular episode enter differently

Recognizing a watch requires familiar object knowledge. Remembering that this watch was placed beside the charger this morning requires an episode-specific relation. Anterior-temporal disruption impairs semantic processing, whereas hippocampal retrieval is associated with reinstating linked event elements. A model should therefore separate reusable knowledge from rapidly learned episode bindings while allowing each to cue the other.[^18][^51]

General knowledge can generate a useful search prior: watches are often near places where they are removed. Episodic retrieval can refine or override it: this morning it was put somewhere unusual. Treating both as a single undifferentiated memory lookup would make it difficult to explain how familiar expectations compete with a specific recent event.

### 6.2 Event context persists across changing sensory details

A search consists of changing views, body positions and local actions. It is still one ongoing event because selected relations remain stable: the target is the watch, the room is the current search space, and the next commitment is the meeting.

Naturalistic narrative studies show that higher-order cortical regions maintain event representations over longer intervals than sensory regions, and that higher-level event boundaries relate to hippocampal activity and later memory. Separately, intrinsic activity timescales vary across the cortical hierarchy. The two measurements are different, but both motivate multiple integration timescales in a computational architecture.[^55][^56]

A useful event-state update should preserve what remains relevant while incorporating new evidence. For example, checking the desk changes 'which candidate location remains plausible' without replacing 'what object is being sought.' This is a proposed functional decomposition of state updating, not a claim that each proposition occupies a distinct cortical register.

### 6.3 Anatomical contributions within contextual processing

Retrosplenial activity combines route-related, movement-related and environmental spatial information. Such representations can help relate the current viewpoint to a remembered environment. Parahippocampal and posterior medial association systems contribute contextual and event-related information, while lateral temporal and anterior temporal representations contribute conceptual structure.[^57][^55][^18]

The broad default-mode-network label includes differentiated systems. Precision mapping distinguishes distributed networks preferentially associated with episodic construction versus social reasoning. For a modular model, those are different input-output requirements: reconstruct an event or scene; infer another person's knowledge or intentions; relate a situation to previously learned contextual information.[^58]

Frontoparietal control can cooperate with these contextual systems during internally directed planning and with dorsal-attention systems during externally oriented tasks. The practical significance is that the same overall goal can temporarily prioritize remembered information, then return to sensory sampling.[^59]

Orbitofrontal cortex adds information about inferred task states and outcome contingencies. When a relevant state cannot be read directly from the current stimulus, orbitofrontal activity can represent its hidden features. This is useful when the same visible object has different implications depending on what has already happened or which rule is in force.[^60]

## 7. Goal maintenance, routing and selection

### 7.1 A goal changes several interfaces at once

To search for the watch, the retained task state must influence object-feature selection, spatial sampling, memory queries and the criterion for stopping. Prefrontal mixed selectivity provides a population substrate for combinations of target, rule and context; frontal-parietal-thalamic interactions support retention and selection.[^11][^3][^4]

The corresponding implementation needs more than a final action-selection head. Goal-related signals should reach the operations that construct the evidence on which that head acts. Otherwise the model has no mechanism by which looking for a watch changes what it sees or which memory it retrieves.

### 7.2 Cortical and thalamic control operate on different parts of the route

Frontal-eye-field input can enhance spatially corresponding V4 activity. Competition within visual populations can then favor the selected object or location. This supplies a concrete top-down path from retained task demands to sensory representation.[^6][^7]

Thalamic control provides additional mechanisms. In a mouse divided-attention task, prefrontal-dependent sensory selection involved the visual thalamic reticular nucleus and inhibitory control of LGN processing. In another task, mediodorsal thalamus supported prefrontal representations through its effect on cortical interactions.[^61][^4]

These should produce different model operations: **input gain**, **competition among alternatives**, and **stability of the retained task state**. A single scalar attention weight cannot automatically substitute for all three.

### 7.3 Basal-ganglia loops regulate candidate actions and configurations

Cortical input reaches striatal circuits that influence basal-ganglia output nuclei and downstream thalamic/brainstem targets. The direct route can reduce inhibitory output to selected targets; indirect routes, involving pallidal and subthalamic circuitry, change competing output patterns. A fast cortical route through the subthalamic nucleus also reaches pallidal output.[^62][^63]

For implementation, these loops motivate a selection mechanism that changes which candidate behavior or task configuration can influence downstream processing. Selecting 'inspect the shelf,' 'turn on a light,' or 'reach toward the object' changes the operative controller. Simultaneous direct- and indirect-pathway activity during action initiation makes a literal independent go/no-go pair inadequate.[^63]

Extending this mechanism to explicit memory read/write gates is a model-level choice. It is a useful hypothesis to test, but a software gate should not be presented as a directly observed anatomical instruction bit.

### 7.4 Bodily state and stress change the processing regime

Internal needs can change cue representation itself. In mice, hypothalamic hunger circuitry modifies insular food-cue responses through a pathway involving thalamus and amygdala. Amygdala populations also encode changing positive and negative learned cue values in primates.[^64][^65]

Arousal circuits maintain conditions in which cortical and thalamic processing operates, and neuromodulation can rapidly alter sensory responses and performance. These effects act on current processing, not only long-term learning.[^66][^67]

Stress introduces another mode of interaction. Human experiments link noradrenergic signaling under stress to changes in network coupling and show reduced working-memory-related prefrontal activity in particular tasks. For the watch scenario, these results support modeling stress as a change in retention, gain and control effectiveness, rather than only a higher numerical penalty for lateness.[^68][^69]

The behavioral result depends on the person and circumstances. A reasonable scenario-level prediction is that urgency can favor rapid sampling while severe stress can make it harder to maintain which locations were already checked. That prediction should be tested rather than assumed from the word 'stress.'

## 8. Action generation and prediction of consequences

### 8.1 Identity must be converted into a controllable target

Recognizing the watch is not sufficient to pick it up. Its position and orientation must be related to the hand, and the hand must approach with a suitable configuration. AIP's visual, somatosensory and premotor connections provide an anatomical basis for this convergence; premotor and motor population dynamics then contribute to the evolving movement.[^2][^70]

The output is a time-varying control process. As contact occurs, new sensory evidence changes the state on which the controller acts. A model that selects a final pose without simulating the intervening sensory feedback will omit an important part of the perceptual loop.

### 8.2 Cerebellar loops support predictive adaptation

Cerebellar circuits participate in loops with both motor and prefrontal cortex. Cortical information reaches cerebellar circuitry through pontine routes, and cerebellar outputs can return through thalamic routes. These loops permit predictions and corrections to interact with ongoing cortical processing.[^71]

Human reaching experiments show that cerebellum-dependent adaptation responds to discrepancies between predicted and observed sensory consequences. Computationally, keep **online correction** distinct from **learning that changes the next attempt**. The first changes the current action; the second updates the mapping between commands and expected outcomes.[^72]

If a cup is heavier than expected, the prediction error concerns the action's sensory consequences. It need not invalidate the cup's visual identity. This distinction is essential for assigning an error to the representation or controller that should change.

### 8.3 Learning signals have different targets

Reward-related dopamine responses motivate a mechanism for updating outcome expectations and action preferences. Other experiments distinguish dopamine spiking and release dynamics related to learning from those related to motivation. A model should therefore avoid merging every mismatch, unsuccessful movement and disappointing outcome into a single global scalar error.[^73][^74]

At another timescale, disrupting post-training hippocampal ripples impairs later spatial memory. This supports including offline or interleaved memory-dependent learning, while leaving the exact cortical training objective open.[^10]

The resulting learning architecture can have separate updates for sensory representations, movement predictions, action values, and episodic associations. That separation is a design interpretation of the distinct evidence, not an assertion that the brain uses the same optimizer in four locations.

## 9. What makes the percept functionally coherent?

A functional account of global perception needs several forms of consistency at once:

1. **Cross-sensory consistency:** which visual, auditory and tactile signals concern the same event, and how reliable are they?
2. **Spatial consistency:** how do object locations relate to the current eyes, body and remembered environment?
3. **Temporal consistency:** what persists across movement, interruption and changing views?
4. **Task consistency:** which information should influence memory queries, sampling and action now?

The mechanisms described above provide concrete constraints for these requirements: multisensory assignment and weighting, motor-copy-based updating, retained task states, event representations and cortical reinstatement.[^24][^26][^27][^3][^55][^51]

A model does not have to place every variable in one common vector. It does need interfaces that preserve relevant identity, time and coordinate relationships, and it must allow a change in one representation to alter the others when appropriate. For example, a newly visible empty desk should affect the location hypothesis, but should not erase the target identity or the meeting goal.

This specifies coherent perception and flexible use of information. It does not establish that an artificial implementation has subjective experience. The 2025 adversarial comparison of global-workspace and integrated-information theories challenged specified predictions of both; no result in the evidence assembled here licenses a unique architecture as sufficient for consciousness.[^75]

## 10. A modular neural-network specification

The following is a proposed implementation decomposition. Its columns describe testable interfaces, not literal tensor formats transmitted by biological axons.

| Module | Maintained information | Principal incoming information | Principal outputs and update requirements |
|---|---|---|---|
| Modality-specific recurrent encoders | Recent sensory features and local interpretations | Raw sensory streams; contextual and goal-dependent modulation | Evidence for features, objects or events; preserve uncertainty and temporal order |
| Body and spatial-state estimation | Eye/body configuration, contact relations and self-motion | Proprioception, touch, vestibular/visual information, motor copy | Coordinate relationships useful for localization and control |
| Object and semantic representation | Reusable identity and conceptual structure | Sensory evidence and retrieved associations | Recognition-related hypotheses and concept-based constraints |
| Temporary retention | Selected content with task-dependent precision and accessibility | Sensory/semantic content; update and prioritization signals | Delay-resistant information for comparison, search and action |
| Rapid relational memory | Episode-specific relations and competing similar events | Convergent content/context; partial retrieval cues | Pattern-separated storage, partial-cue completion and cortical reinstatement |
| Event/context representation | Ongoing situation, spatial context and temporal relations | Current evidence, recalled episodes and existing knowledge | Stable context with selective updating and event transitions |
| Task control and routing | Goal, rule, relevant subgoal and selection priorities | Context, retained information, outcomes and bodily relevance | Changes in sensory gain, retrieval priority, memory updating and controller selection |
| Action selection and execution | Candidate actions and evolving motor state | Target geometry, task priorities and body state | Eye/head/hand/body control, with feedback-dependent continuation |
| Predictive adaptation | Expected consequences of actions | Motor command, estimated current state and observed outcome | Error-specific updates and predictive information for control |
| State and learning modulation | Bodily needs, arousal-related conditions and learning signals | Internal state, outcomes and task context | Changes in gain, retention, motivation and plasticity |

Three restrictions keep such a model scientifically informative. First, do not provide privileged object identities, target locations or hidden simulator state to a module that would need to infer them from sensory input. Second, distinguish currently observed evidence from retrieved or predicted information in the implementation, even if the distinction is distributed rather than a literal source tag. Third, measure the proposed operation, not just final reward: a successful policy can bypass the intended memory or perceptual computation.

### Validation tasks and predicted failure modes

| Controlled manipulation | Diagnostic behavior | What the test distinguishes |
|---|---|---|
| Weaken recurrent sensory interaction | Greater difficulty with clutter, ambiguous grouping or context-dependent interpretation | Initial feature detection versus recurrent organization |
| Increase delay, similarity and distractor load separately | Changes in precision, swaps and forgetting | Retention quality versus interference and control |
| Mix visual and auditory tasks with matched versus different codes | Representation-dependent cross-modal costs | Modality-specific retention versus shared representational/control demands |
| Increase similarity between stored episodes | Confusions about which object was where in which event | Pattern separation versus ordinary recognition |
| Remove part of a learned retrieval cue | Failure to recover associated event elements | Completion versus exact-match lookup |
| Move the eyes/body while preserving the external scene | Self-motion-induced localization errors | Spatial updating versus static recognition |
| Change object weight without changing appearance | Initial error, online correction and across-trial adaptation | Identity, feedback control and forward-model learning |
| Interrupt a multi-step task | Loss or restoration of goal, target and completed-step information | Task-state retention and contextual reinstatement |

These tests should be used as constrained behavioral assays. A particular artificial lesion need not reproduce a biological lesion in every respect; the value is whether the module actually contributes the operation attributed to it.

## 11. Complete activity traces

These are evidence-constrained reconstructions assembled from the mechanisms above, not claims that one experiment recorded every listed circuit during the entire activity.

### 11.1 Searching for a watch in a dark room while late and stressed

**1. Establish the problem.** The retained task includes the target watch and the need to leave for a meeting. Conceptual knowledge supplies what counts as a watch; episode-specific memory can supply where this particular watch was last handled. Prefrontal-parietal-thalamic task activity maintains the operative constraints, while event/context representations preserve the broader situation.[^11][^4][^18][^51]

**2. Retrieve a candidate location.** Cortical cues such as 'took it off beside the phone' reach entorhinal-hippocampal circuits. Associative retrieval can reinstate linked place and event information, while DG-dependent separation helps prevent similar previous episodes from being merged. The recovered relation is a candidate location, not yet a current observation.[^45][^51]

**3. Convert the candidate into a search movement.** Remembered environmental relationships must be related to the current viewpoint and body position. Retrosplenial/parietal processing provides relevant spatial combinations; frontal/parietal selection biases the next gaze target. Eye movement produces both a new view and motor-related information needed to interpret the retinal displacement.[^57][^6][^27]

**4. Evaluate what the new view actually establishes.** Early and recurrent visual processing provide weak or ambiguous shape evidence in darkness. A watch-related target representation biases relevant features, but weak evidence must remain weak. Failure to detect the watch on a barely visible surface should reduce confidence in that location less than a clear, unobstructed inspection. This is a scenario-level application of reliability-sensitive inference.[^7][^14][^26]

An explicit implementation can express this distinction as:

\[
p(L\mid \neg D,q) \propto p(\neg D\mid L,q)\,p(L),
\]

where \(L\) is a candidate location, \(D\) is detection, and \(q\) describes observation quality. Under poor visibility, non-detection may be quite likely even when the watch is present. The equation is a proposed model update, not an identified neural calculation.

**5. Choose whether to look again, illuminate, or feel.** Task-control and action-selection circuits compare candidate behaviors in the current context. Switching on a light changes the future quality of visual evidence; feeling the surface acquires a different sensory constraint. The model's choice should depend on time pressure, effort, expected informativeness and current location confidence. Those explicit variables are an engineering specification of the decision problem, not a claim that one brain area computes their exact weighted sum.

Stress changes this loop's operating conditions. Noradrenergic and prefrontal effects can alter responsiveness and retention. A stressed implementation might over-prioritize rapid checking while losing track of completed inspections; that is a testable failure mode, not a universal description of stress.[^68][^69]

**6. Confirm and grasp.** A clearer view or tactile contact strengthens the candidate's identity and position. Visual and tactile evidence can be related to the same object; AIP-premotor interactions translate its geometry into hand configuration, and motor/cerebellar loops predict and adapt the evolving action.[^26][^2][^70][^72]

**7. Terminate this subgoal and update the event.** Confirmation that the watch is in hand removes the need to keep sampling candidate locations. The task state changes toward leaving for the meeting. The newly established relation between watch, location and retrieval event is available for subsequent episodic learning. This last transition is the proposed control-level consequence of confirmation and retained goals.

The important flow is therefore:

```text
goal + recent episode
    -> target features + candidate location
    -> viewpoint/coordinate update
    -> selected gaze or exploratory action
    -> sensory evidence weighted by viewing/contact conditions
    -> revised location and identity hypotheses
    -> another query/action, or confirmation and grasp
    -> next task state and event-memory update
```

### 11.2 Following a conversation while making tea

**Auditory interpretation.** The attended speaker is preferentially represented amid competing sounds. Acoustic/phonological information must remain available long enough to form words and ordered expressions; semantic processing relates them to concepts, and an event representation preserves who is discussing what.[^16][^17][^18][^55]

**Concurrent task retention.** Making tea requires a prospective task state: what is being prepared and which step is current. Visual and somatosensory information support manipulation of the cup and kettle. Some of the sensory retention demands are separable from the verbal stream; interference becomes stronger when both activities require a central decision, a task switch or competing representation updates.[^37][^2][^3]

**Interruption and resumption.** A question can prioritize the conversation while leaving aspects of the tea-making task latent or contextually recoverable. Resumption requires reinstating the relevant step, not rebuilding every earlier sensory frame. Retained availability, context and memory retrieval jointly provide routes by which that resumption can occur.[^9][^51][^59]

**Characteristic errors.** Poor verbal retention can lose the beginning of a sentence while cup manipulation continues. Poor task-state maintenance can produce a repeated or omitted tea-making step even when the sentence is understood. These different failures distinguish content storage from control and prospective-state retention.

### 11.3 Lifting a familiar cup that is unexpectedly heavy

**Before contact.** Ventral visual processing supports cup identification, while parietal-premotor processing relates its geometry to the hand. Familiar experience supplies an expected relationship between the intended lifting action and its sensory consequences.[^13][^2]

**During lifting.** The observed motion and somatosensory consequences differ from expectation. Feedback alters the evolving movement; cerebellum-dependent adaptation can change the prediction and control mapping used on later attempts.[^70][^72]

**After the mismatch.** The object can remain recognized as the same cup while the expected weight-related consequences change. Episodic memory can bind the particular encounter, while more general learning adjusts future expectations. The architecture must permit an error to update the action-prediction system without unnecessarily destroying object identity or replacing the whole contextual state.[^51][^72]

This is the practical meaning of an integrated perceptual architecture: several representations describe the same ongoing event, but each changes only when the incoming evidence is relevant to the computation it supports.

## Sources

[^1]: Markov et al. (2014). [Anatomy of hierarchy: feedforward and feedback pathways in macaque visual cortex](https://doi.org/10.1002/cne.23458). DOI: 10.1002/cne.23458.

[^2]: Borra et al. (2008 (online 2007)). [Cortical Connections of the Macaque Anterior Intraparietal (AIP) Area](https://doi.org/10.1093/cercor/bhm146). DOI: 10.1093/cercor/bhm146.

[^3]: Guo et al. (2017). [Maintenance of persistent activity in a frontal thalamocortical loop](https://doi.org/10.1038/nature22324). DOI: 10.1038/nature22324.

[^4]: Schmitt et al. (2017). [Thalamic amplification of cortical connectivity sustains attentional control](https://doi.org/10.1038/nature22073). DOI: 10.1038/nature22073.

[^5]: Kohara et al. (2014). [Cell type-specific genetic and optogenetic tools reveal hippocampal CA2 circuits](https://doi.org/10.1038/nn.3614). DOI: 10.1038/nn.3614.

[^6]: Moore and Armstrong (2003). [Selective gating of visual signals by microstimulation of frontal cortex](https://doi.org/10.1038/nature01341). DOI: 10.1038/nature01341.

[^7]: Reynolds, Chelazzi and Desimone (1999). [Competitive mechanisms subserve attention in macaque areas V2 and V4](https://doi.org/10.1523/JNEUROSCI.19-05-01736.1999). DOI: 10.1523/JNEUROSCI.19-05-01736.1999.

[^8]: Harrison and Tong (2009). [Decoding reveals the contents of visual working memory in early visual areas](https://doi.org/10.1038/nature07832). DOI: 10.1038/nature07832.

[^9]: Rose et al. (2016). [Reactivation of latent working memories with transcranial magnetic stimulation](https://doi.org/10.1126/science.aah7011). DOI: 10.1126/science.aah7011.

[^10]: Girardeau et al. (2009). [Selective suppression of hippocampal ripples impairs spatial memory](https://doi.org/10.1038/nn.2384). DOI: 10.1038/nn.2384.

[^11]: Rigotti et al. (2013). [The importance of mixed selectivity in complex cognitive tasks](https://doi.org/10.1038/nature12160). DOI: 10.1038/nature12160.

[^12]: Salzman, Britten and Newsome (1990). [Cortical microstimulation influences perceptual judgements of motion direction](https://doi.org/10.1038/346174a0). DOI: 10.1038/346174a0.

[^13]: Goodale et al. (1991). [A neurological dissociation between perceiving objects and grasping them](https://doi.org/10.1038/349154a0). DOI: 10.1038/349154a0.

[^14]: Self et al. (2012). [Different glutamate receptors convey feedforward and recurrent processing in macaque V1](https://doi.org/10.1073/pnas.1119527109). DOI: 10.1073/pnas.1119527109.

[^15]: Romanski et al. (1999). [Dual streams of auditory afferents target multiple domains in the primate prefrontal cortex](https://doi.org/10.1038/16056). DOI: 10.1038/16056.

[^16]: Mesgarani and Chang (2012). [Selective cortical representation of attended speaker in multi-talker speech perception](https://doi.org/10.1038/nature11020). DOI: 10.1038/nature11020.

[^17]: Paulesu, Frith and Frackowiak (1993). [The neural correlates of the verbal component of working memory](https://doi.org/10.1038/362342a0). DOI: 10.1038/362342a0.

[^18]: Pobric, Jefferies and Lambon Ralph (2007). [Anterior temporal lobes mediate semantic representation: mimicking semantic dementia by using rTMS in normal participants](https://doi.org/10.1073/pnas.0707383104). DOI: 10.1073/pnas.0707383104.

[^19]: Padberg et al. (2009). [Thalamocortical connections of parietal somatosensory cortical fields in macaque monkeys are highly divergent and convergent](https://doi.org/10.1093/cercor/bhn229). DOI: 10.1093/cercor/bhn229.

[^20]: Fetsch, Turner, DeAngelis and Angelaki (2009). [Dynamic reweighting of visual and vestibular cues during self-motion perception](https://doi.org/10.1523/JNEUROSCI.2574-09.2009). DOI: 10.1523/JNEUROSCI.2574-09.2009.

[^21]: Bolding and Franks (2018). [Recurrent cortical circuits implement concentration-invariant odor coding](https://doi.org/10.1126/science.aat6904). DOI: 10.1126/science.aat6904.

[^22]: Bolding et al. (2020). [Recurrent circuitry is required to stabilize piriform cortex odor representations across brain states](https://doi.org/10.7554/eLife.53125). DOI: 10.7554/eLife.53125.

[^23]: de Araujo et al. (2003). [Taste-olfactory convergence, and the representation of the pleasantness of flavour, in the human brain](https://doi.org/10.1046/j.1460-9568.2003.02915.x). DOI: 10.1046/j.1460-9568.2003.02915.x.

[^24]: Kording et al. (2007). [Causal Inference in Multisensory Perception](https://doi.org/10.1371/journal.pone.0000943). DOI: 10.1371/journal.pone.0000943.

[^25]: Rohe and Noppeney (2015). [Cortical Hierarchies Perform Bayesian Causal Inference in Multisensory Perception](https://doi.org/10.1371/journal.pbio.1002073). DOI: 10.1371/journal.pbio.1002073.

[^26]: Ernst and Banks (2002). [Humans integrate visual and haptic information in a statistically optimal fashion](https://doi.org/10.1038/415429a). DOI: 10.1038/415429a.

[^27]: Sommer and Wurtz (2006). [Influence of the thalamus on spatial visual processing in frontal cortex](https://doi.org/10.1038/nature05279). DOI: 10.1038/nature05279.

[^28]: Sperling (1960). [The information available in brief visual presentations](https://doi.org/10.1037/h0093759). DOI: 10.1037/h0093759.

[^29]: Treisman and Rostron (1972). [Brief auditory storage: a modification of Sperling's paradigm applied to audition](https://doi.org/10.1016/0001-6918(72)90021-2). DOI: 10.1016/0001-6918(72)90021-2.

[^30]: Luck and Vogel (1997). [The capacity of visual working memory for features and conjunctions](https://doi.org/10.1038/36846). DOI: 10.1038/36846.

[^31]: Bays and Husain (2008). [Dynamic shifts of limited working memory resources in human vision](https://doi.org/10.1126/science.1158023). DOI: 10.1126/science.1158023.

[^32]: Baddeley, Thomson and Buchanan (1975). [Word length and the structure of short-term memory](https://doi.org/10.1016/S0022-5371(75)80045-4). DOI: 10.1016/S0022-5371(75)80045-4.

[^33]: Lovatt, Avons and Masterson (2000). [The word-length effect and disyllabic words](https://doi.org/10.1080/713755877). DOI: 10.1080/713755877.

[^34]: Li et al. (2025). [Exploring the capacity and executive function of vibrotactile working memory through the frequency sequencing task](https://doi.org/10.1111/nyas.15327). DOI: 10.1111/nyas.15327.

[^35]: Saults and Cowan (2007). [A central capacity limit to the simultaneous storage of visual and auditory arrays in working memory](https://doi.org/10.1037/0096-3445.136.4.663). DOI: 10.1037/0096-3445.136.4.663.

[^36]: Fougnie and Marois (2011). [What limits working memory capacity? Evidence for modality-specific sources to the simultaneous storage of visual and auditory arrays](https://doi.org/10.1037/a0024834). DOI: 10.1037/a0024834.

[^37]: Fougnie et al. (2015). [Working memory storage is intrinsically domain specific](https://doi.org/10.1037/a0038211). DOI: 10.1037/a0038211.

[^38]: Romo et al. (1999). [Neuronal correlates of parametric working memory in the prefrontal cortex](https://doi.org/10.1038/20939). DOI: 10.1038/20939.

[^39]: Romo et al. (2002). [Neuronal correlates of decision-making in secondary somatosensory cortex](https://doi.org/10.1038/nn950). DOI: 10.1038/nn950.

[^40]: Mozumder et al. (2026 (online 24 December 2025)). [Asynchronous firing and off states in working memory maintenance](https://doi.org/10.1016/j.celrep.2025.116764). DOI: 10.1016/j.celrep.2025.116764.

[^41]: Olson et al. (2006). [Working memory for conjunctions relies on the medial temporal lobe](https://doi.org/10.1523/JNEUROSCI.1923-05.2006). DOI: 10.1523/JNEUROSCI.1923-05.2006.

[^42]: Jeneson, Mauldin, Hopkins and Squire (2011). [The role of the hippocampus in retaining relational information across short delays: the importance of memory load](https://doi.org/10.1101/lm.2010711). DOI: 10.1101/lm.2010711.

[^43]: Suzuki and Amaral (1994). [Perirhinal and parahippocampal cortices of the macaque monkey: cortical afferents](https://doi.org/10.1002/cne.903500402). DOI: 10.1002/cne.903500402.

[^44]: Suzuki, Miller and Desimone (1997). [Object and Place Memory in the Macaque Entorhinal Cortex](https://doi.org/10.1152/jn.1997.78.2.1062). DOI: 10.1152/jn.1997.78.2.1062.

[^45]: Neunuebel and Knierim (2014). [CA3 retrieves coherent representations from degraded input: direct evidence for CA3 pattern completion and dentate gyrus pattern separation](https://doi.org/10.1016/j.neuron.2013.11.017). DOI: 10.1016/j.neuron.2013.11.017.

[^46]: Nakazawa et al. (2002). [Requirement for hippocampal CA3 NMDA receptors in associative memory recall](https://doi.org/10.1126/science.1071795). DOI: 10.1126/science.1071795.

[^47]: Hitti and Siegelbaum (2014). [The hippocampal CA2 region is essential for social memory](https://doi.org/10.1038/nature13028). DOI: 10.1038/nature13028.

[^48]: Remondes and Schuman (2002). [Direct cortical input modulates plasticity and spiking in CA1 pyramidal neurons](https://doi.org/10.1038/416736a). DOI: 10.1038/416736a.

[^49]: Maimon et al. (2026). [Sparse-to-dense coding transformation between hippocampal areas CA3 and CA1](https://doi.org/10.1038/s41586-026-10537-0). DOI: 10.1038/s41586-026-10537-0.

[^50]: Surmeli et al. (2015). [Molecularly Defined Circuitry Reveals Input-Output Segregation in Deep Layers of the Medial Entorhinal Cortex](https://doi.org/10.1016/j.neuron.2015.10.041). DOI: 10.1016/j.neuron.2015.10.041.

[^51]: Horner et al. (2015). [Evidence for holistic episodic recollection via hippocampal pattern completion](https://doi.org/10.1038/ncomms8462). DOI: 10.1038/ncomms8462.

[^52]: Tompary, Duncan and Davachi (2016). [High-resolution investigation of memory-specific reinstatement in the hippocampus and perirhinal cortex](https://doi.org/10.1002/hipo.22582). DOI: 10.1002/hipo.22582.

[^53]: Hafting et al. (2005). [Microstructure of a spatial map in the entorhinal cortex](https://doi.org/10.1038/nature03721). DOI: 10.1038/nature03721.

[^54]: MacDonald et al. (2011). [Hippocampal Time Cells Bridge the Gap in Memory for Discontiguous Events](https://doi.org/10.1016/j.neuron.2011.07.012). DOI: 10.1016/j.neuron.2011.07.012.

[^55]: Baldassano et al. (2017). [Discovering Event Structure in Continuous Narrative Perception and Memory](https://doi.org/10.1016/j.neuron.2017.06.041). DOI: 10.1016/j.neuron.2017.06.041.

[^56]: Murray et al. (2014). [A hierarchy of intrinsic timescales across primate cortex](https://doi.org/10.1038/nn.3862). DOI: 10.1038/nn.3862.

[^57]: Alexander and Nitz (2015). [Retrosplenial cortex maps the conjunction of internal and external spaces](https://doi.org/10.1038/nn.4058). DOI: 10.1038/nn.4058.

[^58]: DiNicola, Braga and Buckner (2020). [Parallel distributed networks dissociate episodic and social functions within the individual](https://doi.org/10.1152/jn.00529.2019). DOI: 10.1152/jn.00529.2019.

[^59]: Spreng et al. (2010). [Default network activity, coupled with the frontoparietal control network, supports goal-directed cognition](https://doi.org/10.1016/j.neuroimage.2010.06.016). DOI: 10.1016/j.neuroimage.2010.06.016.

[^60]: Schuck et al. (2016). [Human Orbitofrontal Cortex Represents a Cognitive Map of State Space](https://doi.org/10.1016/j.neuron.2016.08.019). DOI: 10.1016/j.neuron.2016.08.019.

[^61]: Wimmer et al. (2015). [Thalamic control of sensory selection in divided attention](https://doi.org/10.1038/nature15398). DOI: 10.1038/nature15398.

[^62]: Nambu et al. (2000). [Excitatory cortical inputs to pallidal neurons via the subthalamic nucleus in the monkey](https://doi.org/10.1152/jn.2000.84.1.289). DOI: 10.1152/jn.2000.84.1.289.

[^63]: Cui et al. (2013). [Concurrent activation of striatal direct and indirect pathways during action initiation](https://doi.org/10.1038/nature11846). DOI: 10.1038/nature11846.

[^64]: Livneh et al. (2017). [Homeostatic circuits selectively gate food cue responses in insular cortex](https://doi.org/10.1038/nature22375). DOI: 10.1038/nature22375.

[^65]: Paton et al. (2006). [The primate amygdala represents the positive and negative value of visual stimuli during learning](https://doi.org/10.1038/nature04490). DOI: 10.1038/nature04490.

[^66]: Fuller et al. (2011). [Reassessment of the structural basis of the ascending arousal system](https://doi.org/10.1002/cne.22559). DOI: 10.1002/cne.22559.

[^67]: Pinto et al. (2013). [Fast modulation of visual perception by basal forebrain cholinergic neurons](https://doi.org/10.1038/nn.3552). DOI: 10.1038/nn.3552.

[^68]: Hermans et al. (2011). [Stress-related noradrenergic activity prompts large-scale neural network reconfiguration](https://doi.org/10.1126/science.1209603). DOI: 10.1126/science.1209603.

[^69]: Qin et al. (2009). [Acute psychological stress reduces working memory-related activity in the dorsolateral prefrontal cortex](https://doi.org/10.1016/j.biopsych.2009.03.006). DOI: 10.1016/j.biopsych.2009.03.006.

[^70]: Churchland et al. (2012). [Neural population dynamics during reaching](https://doi.org/10.1038/nature11129). DOI: 10.1038/nature11129.

[^71]: Kelly and Strick (2003). [Cerebellar loops with motor cortex and prefrontal cortex of a nonhuman primate](https://doi.org/10.1523/JNEUROSCI.23-23-08432.2003). DOI: 10.1523/JNEUROSCI.23-23-08432.2003.

[^72]: Tseng et al. (2007). [Sensory prediction errors drive cerebellum-dependent adaptation of reaching](https://doi.org/10.1152/jn.00266.2007). DOI: 10.1152/jn.00266.2007.

[^73]: Schultz, Dayan and Montague (1997). [A Neural Substrate of Prediction and Reward](https://doi.org/10.1126/science.275.5306.1593). DOI: 10.1126/science.275.5306.1593.

[^74]: Mohebi et al. (2019). [Dissociable dopamine dynamics for learning and motivation](https://doi.org/10.1038/s41586-019-1235-y). DOI: 10.1038/s41586-019-1235-y.

[^75]: COGITATE Consortium et al. (2025). [Adversarial testing of global neuronal workspace and integrated information theories of consciousness](https://doi.org/10.1038/s41586-025-08888-1). DOI: 10.1038/s41586-025-08888-1.

