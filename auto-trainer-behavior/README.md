# Autotrainer - Behavior

The autotrainer behavior module implements the flow of a training session from the cage to the experimental
apparatus and back to the cage.

At a high-level, the architecture can be described as follows:

* A Finite State Machine (FSM) that manages the flow of the training session based on a set of requirements linked below
* An "algorithm" and associated analysis that facilitates the decision-making of transitions in the state machine

The purpose is to separate the mechanics of the intended states and transition flow as from the actual decision-making
as much as possible.

The primary reason for this goal is to make it easier for people who understand the science and training to
make changes to the behavior of the system without having to also learn the specifics of the FSM module
and implementation. A secondary reason is that it is not yet clear that a) the current FSM package is the
right option and b) that an FSM is the right approach. As the system and training requirements evolve,
these may change and ideally changing the mechanics of the transitions should be minimally disruptive to the
behavior design.

This can be a hard line to draw - what is inherent to the "low-level" FSM operation and what is part of the
"algorithm".  As the training requirements have evolved, the decisions in the current implementation may not
be optimal.  Modifications are encouraged so long as the above objective is maintained.

## Requirements

The up-to-date session behavior is defined [here](https://lucid.app/lucidchart/fc67e3f9-932f-4450-bbc9-d26042e340b7/view?page=0_0).

Use the live diagram for development; the former repository snapshot is no
longer maintained.

## Implementation Details

The finite state machine is implemented using the `transitions` package.

Although `transitions` supports hierarchical state machines, it did not seem to fit well with the nature of the
apparent hierarchy in this system by default.  Subsystems, such as pellet management operate across higher-level
states such as "in tunnel" or "in cage".  It is conceivable to have subsystems duplicated below parents
and transition between them.  For now, the decision was made to keep discrete state machines for subsystems
that cross those higher-level states.  This is potentially an area for improvement, however.  There is some
coupling between these supposedly "independent" state machines, that represents that there is some relationship
there, if not strictly a pure hierarchy.

The module exposes two primary classes for consumers:
* `StateMachine` - the top-level FSM
* `BehaviorAlgorithm` - the decision-making used by the FSM to help with transitions

Ideally, this would be simplified for consumers.  The exposure of the state machine is largely just to
hand off the objects that provide a) implementation of the actions that should be taken with hardware
or other parts of the system as a part of transitions or states and b) event sources for changes in the
system that may trigger transitions.

The exposure of the `BehaviorAlgorithm` is largely just to allow configuration of variables, such as the
maximum number of pellets that can be consumed in a session.  Note that these may change during training
and are not just initialization.

There is a likely a simplified, more coherent interface to the behavior model that could be created.
