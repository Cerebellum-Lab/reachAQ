# Closed-loop latency tuning

Applies to rigs running the stim camera at 900 Hz for closed-loop stimulation.
Both settings below are installed by
`tools/install/reachaq-linux-install.sh`; this page explains what they do, how
to verify them, and how to turn them off.

Neither changes the kernel, the GPU driver, or NI-DAQmx.

## Why

The 900 Hz loop has a 5 ms budget. Measured on the reference workstation
(12th Gen i9-12900, 8 P-cores at 5000-5100 MHz on CPU 0-15, 8 E-cores at
3800 MHz on CPU 16-23), 60 s at 900 Hz pinned to physical P-cores, with the
loop's total cost defined as scheduler wake-up lateness plus decision time:

| condition | p50 | p99 | p99.9 | max | over 5 ms | cycles in 60 s |
|---|---|---|---|---|---|---|
| idle, SCHED_OTHER | 0.129 | 0.221 | 0.262 | 0.332 | 0 | 54,000 |
| idle, SCHED_FIFO | 0.077 | 0.170 | 0.183 | 0.246 | 0 | 54,000 |
| 16 competing processes, SCHED_OTHER | 0.099 | 3.662 | 4.481 | **6.050** | 13 (0.026%) | 49,480 |
| 16 competing processes, SCHED_FIFO | 0.047 | 0.059 | 0.122 | **0.155** | 0 | 54,000 |

The detector's own work never exceeds 0.09 ms even under load. The entire tail
was `SCHED_OTHER` wake-up latency. `SCHED_FIFO` bounds it, removes every missed
deadline, and restores the full 900 Hz cadence.

**A PREEMPT_RT kernel is not required for this.** Stock Linux already preempts
`SCHED_OTHER` for `SCHED_FIFO` immediately, and the result above leaves roughly
32x headroom under a load harsher than real acquisition.

Separately, `intel_pstate` defaults to the `powersave` governor, which held the
P-cores near 2700 MHz under sustained load against a 5000 MHz policy ceiling
with turbo enabled. Effect on live pose inference (DeepLabCut TensorFlow,
batch 2 at 128x128, on the GPU):

| governor | p50 | p99 | max |
|---|---|---|---|
| powersave | 12.55 | 17.58 | 18.96 |
| performance | **10.79** | **11.53** | 11.73 |

## Real-time priority for the stim loop

The installer creates the `reachaq-rt` group, adds the operator to it, and
writes `/etc/security/limits.d/90-reachaq-rtprio.conf`:

```
@reachaq-rt   -   rtprio   80
@reachaq-rt   -   memlock  524288
```

Priority 80 is deliberately below the kernel's own real-time threads, which run
at 99, so a runaway loop cannot lock the machine out. Linux RT throttling is a
further backstop: `sched_rt_runtime_us` reserves 5% of each period for normal
tasks by default.

Group membership and `limits.d` both apply **at login**. Log out and back in,
then verify:

```bash
ulimit -r                  # expect 80
chrt -f 80 true && echo ok
```

`VideoCapture` raises its own priority at the top of the capture loop, and only
for the camera that has a stim detector configured. Making every camera process
real-time would take CPU from the ones that only need throughput.

If the allowance is missing the request fails, a warning naming the remedy is
logged, and the loop runs at normal priority exactly as before. A rig that has
not been tuned still acquires.

### Turning it off

Set `REACHAQ_STIM_RT_PRIORITY=0` in the service environment, or remove the
limits file:

```bash
sudo rm /etc/security/limits.d/90-reachaq-rtprio.conf
sudo gpasswd -d "$USER" reachaq-rt
```

## CPU frequency governor

The installer installs and enables `reachaq-cpu-governor.service`, which sets
every CPU to the `performance` governor at boot. A runtime `cpupower` call does
not survive a reboot, which is why this is a unit.

It writes through sysfs rather than calling `cpupower`, so it does not depend on
the versioned `linux-tools` package matching the running kernel.

```bash
systemctl status reachaq-cpu-governor.service
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # expect performance
```

### Turning it off

```bash
sudo systemctl disable --now reachaq-cpu-governor.service
```

Stopping the unit returns the governor to `powersave`.

## What was measured but is not configured

**P-core pinning.** With `powersave`, pinning pose inference to the physical
P-cores was worth 4.35 ms of p99. Once the governor is set to `performance` it
is worth almost nothing (10.79 vs 10.81 ms p50), so the earlier result was the
powersave ramp being worked around rather than a scheduling problem. Nothing
pins CPU affinity, and `isolcpus`, `nohz_full` and `threadirqs` are not set.

**PREEMPT_RT.** Available on Ubuntu 22.04 through Ubuntu Pro without upgrading
the distribution. Not used, because `SCHED_FIFO` already meets the budget with
large headroom, and because the reference workstation installs the NVIDIA driver
as `linux-modules-nvidia-<version>-open-<kernel>` - Canonical's prebuilt
per-kernel modules, not DKMS. Those are not published for the realtime kernel
flavour, so booting PREEMPT_RT would leave the rig with no GPU driver at all.
Migrating to `nvidia-dkms-<version>` would be a prerequisite. NI's modules
(`nitiork`, `niwfrk`, `nixsrk`) are DKMS and would rebuild given RT headers.

## Caveat on these numbers

The load case uses 16 spinning processes, which is harsher than real
acquisition. Treat the 0.026% miss rate as a pessimistic bound, and re-measure
against real camera, DAQ and CAN activity before drawing conclusions about a
specific rig.
