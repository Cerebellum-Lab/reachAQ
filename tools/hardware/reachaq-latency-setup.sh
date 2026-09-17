#!/usr/bin/env bash
# Latency tuning for the reachAQ rig. Review before running; every action is
# reversible and none of it touches the kernel, the GPU driver, or NI-DAQmx.
#
# tools/install/reachaq-linux-install.sh applies both of these as part of a full
# install. This script exists so they can be applied, inspected, or reverted on
# their own, without re-running the whole installer.
#
# See docs/linux-install/latency-tuning.md for the measurements behind them.
#
#   sudo bash reachaq-latency-setup.sh status
#   sudo bash reachaq-latency-setup.sh governor performance
#   sudo bash reachaq-latency-setup.sh governor powersave      # revert
#   sudo bash reachaq-latency-setup.sh rtprio-enable
#   sudo bash reachaq-latency-setup.sh rtprio-disable          # revert
#
# Measured justification, on this host (12th Gen i9-12900, 8 P-cores at
# 5000-5100 MHz on CPU 0-15, 8 E-cores at 3800 MHz on CPU 16-23):
#
#   governor  P-cores sit at ~2700 MHz under sustained load with
#             intel_pstate/powersave, against a 5000 MHz policy ceiling with
#             turbo enabled.
#
#   rtprio    At 900 Hz for 60 s against 16 competing processes, the stim
#             detector's own decision time stayed at 0.066 ms p99, but
#             scheduler wake-up lateness reached 6.49 ms and 10 of 47909
#             cycles (0.021%) missed the 5 ms budget. The compute is not the
#             problem; SCHED_OTHER wake-up latency is. SCHED_FIFO is the
#             cheapest thing that bounds it, and needs no new kernel.
#
set -euo pipefail

RTPRIO_FILE="/etc/security/limits.d/90-reachaq-rtprio.conf"
RT_GROUP="reachaq-rt"

require_root() {
    if [[ "$(id -u)" != "0" ]]; then
        echo "This needs root: sudo bash $0 $*" >&2
        exit 1
    fi
}

p_core_clocks() {
    local cpu
    for cpu in 0 2 4 6; do
        printf '%s ' "$(( $(cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq") / 1000 ))"
    done
}

show_status() {
    echo "governor:      $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "P-core MHz:    $(p_core_clocks)"
    echo "policy max:    $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) / 1000 )) MHz"
    echo "turbo:         $(if [[ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo 1)" == "0" ]]; then echo enabled; else echo disabled; fi)"
    echo "kernel:        $(uname -r)"
    if [[ -f "$RTPRIO_FILE" ]]; then
        echo "rtprio limit:  configured in $RTPRIO_FILE"
        sed 's/^/               /' "$RTPRIO_FILE"
        echo "rt group:      $(getent group "$RT_GROUP" || echo "MISSING")"
    else
        echo "rtprio limit:  not configured (ulimit -r is 0, SCHED_FIFO denied)"
    fi
}

set_governor() {
    local target="$1"
    if [[ "$target" != "performance" && "$target" != "powersave" ]]; then
        echo "governor must be performance or powersave" >&2
        exit 2
    fi
    echo "before: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), P-core MHz: $(p_core_clocks)"
    if command -v cpupower >/dev/null 2>&1; then
        cpupower frequency-set -g "$target" >/dev/null
    else
        local governor_file
        for governor_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "$target" > "$governor_file"
        done
    fi
    echo "after:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), P-core MHz: $(p_core_clocks)"

    # Report any CPU that did not take the setting rather than assuming success.
    local mismatched=0 governor_file
    for governor_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ "$(cat "$governor_file")" != "$target" ]]; then
            echo "WARNING: $governor_file is $(cat "$governor_file")" >&2
            mismatched=$((mismatched + 1))
        fi
    done
    [[ "$mismatched" == "0" ]] && echo "all CPUs set to $target"

    echo
    echo "NOTE: runtime only. A reboot returns this to the distro default."
}

rtprio_enable() {
    # A dedicated group, so granting real-time priority is explicit per user and
    # revocable with gpasswd rather than by editing a limits file again.
    if ! getent group "$RT_GROUP" >/dev/null; then
        groupadd "$RT_GROUP"
        echo "created group $RT_GROUP"
    fi
    local target_user="${SUDO_USER:-christielab10}"
    if ! id -nG "$target_user" | tr ' ' '\n' | grep -qx "$RT_GROUP"; then
        usermod -aG "$RT_GROUP" "$target_user"
        echo "added $target_user to $RT_GROUP"
    fi

    # rtprio 80 leaves headroom below the kernel's own threads and below 99,
    # so a runaway loop cannot lock out everything else. memlock helps only if
    # pages are ever pinned; it is set to a bounded value rather than unlimited.
    cat > "$RTPRIO_FILE" <<EOF
# reachAQ: allow the stim-camera capture/decision thread to run SCHED_FIFO.
# The 900 Hz closed loop misses its 5 ms budget under CPU contention because of
# SCHED_OTHER wake-up latency, not because of its own compute time.
@${RT_GROUP}   -   rtprio   80
@${RT_GROUP}   -   memlock  524288
EOF
    echo "wrote $RTPRIO_FILE"
    echo
    echo "NOTE: limits apply at login. Open a NEW session before testing, then:"
    echo "  ulimit -r                 # expect 80"
    echo "  chrt -f 80 true && echo ok"
}

rtprio_disable() {
    rm -f "$RTPRIO_FILE"
    echo "removed $RTPRIO_FILE"
    if getent group "$RT_GROUP" >/dev/null; then
        echo "group $RT_GROUP left in place; remove members with:"
        echo "  sudo gpasswd -d <user> $RT_GROUP"
    fi
}

action="${1:-status}"
case "$action" in
    status)
        show_status
        ;;
    governor)
        require_root "$@"
        set_governor "${2:?usage: governor performance|powersave}"
        ;;
    rtprio-enable)
        require_root "$@"
        rtprio_enable
        ;;
    rtprio-disable)
        require_root "$@"
        rtprio_disable
        ;;
    *)
        echo "usage: $0 status|governor <perf|powersave>|rtprio-enable|rtprio-disable" >&2
        exit 2
        ;;
esac
