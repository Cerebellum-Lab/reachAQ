# reachAQ behavior

The behavior package owns pellet-machine state, pellet presence/misplacement,
session-scoped counts, and the gates used by automatic pellet cycles. A
recording session may contain many pellet-delivery trials.

Automatic motion is enabled only when all of the following are true:

- acquisition and the behavior algorithm are running;
- a recording session is actively writing frames;
- automatic pellet cycles and pellet delivery are enabled;
- the pellet board and required inference state are ready;
- the pellet machine is in a safe state for the requested command.

The package retains load, send, retract, cover/release, DCS coordinate,
diamond/triangle calibration, motor-drift, pellet-presence, pellet-misplacement,
and post-session shift behavior. It has no tunnel, cage, head-fix, magnet,
alarm, emergency, load-cell, or webcam subsystem.

`BehaviorAlgorithm` supplies the decisions and session counters.
`SystemMachine` coordinates the retained pellet and post-session-analysis state
machines. The acquisition application owns recording state, authoritative
pellet-trial accounting, protocol adaptation, stream persistence, and automatic
session-stop arbitration.

See
[Recording sessions, pellet trials, protocols, and schema migration](../docs/acquisition/session-trials-protocols.md)
for the behavioral contract and configuration fields.
