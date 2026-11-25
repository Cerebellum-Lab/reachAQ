from enum import Enum


class PelletState(str, Enum):

    monitoring = "monitoring"
        # this state is set when pellet is at the "deliver" position.

    loading = "loading"
        # load cycle in progress, or finished.

    sending = "sending"
        # sending pellet to the "deliver" position in progress.
        # when the pellet uuid-ack is received,
        # the pellet machine auto-transition from this state to the monitoring state.

    releasing = "releasing"
        # releasing in progress.

    covering = "covering"
        # covering in progress

    home = "home"
        # going to, or at, home position

    retract = "retract"
        # going to, or at, retract position
