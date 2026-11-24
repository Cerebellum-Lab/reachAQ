# Local API

ReachAQ publishes its events and accepts a few commands on a local TCP port, so
another program on the rig can follow a run or start and stop acquisition. The
service starts with the GUI and with the headless runner
(`src/reachaq/app/headless.py`). It is implemented in
`src/reachaq/core/api/`.

## Configuration

The service is configured from the environment, because it is registered before
any system configuration is loaded.

| Variable | Default | Meaning |
|---|---|---|
| `REACHAQ_API_ENABLED` | `1` | `0`, `false`, `no` or `off` disables the service. |
| `REACHAQ_API_HOST` | `127.0.0.1` | Address to listen on. |
| `REACHAQ_API_PORT` | `0` | Port. `0` lets the OS pick a free one; set a number to pin it. |
| `REACHAQ_API_DISCOVERY_DIR` | see below | Where the running service publishes its record. |
| `REACHAQ_SIMULATION` | unset | Set by `--simulate`; recorded as `simulated` in the record. |

Keep it on loopback. There is no authentication: any process that can reach the
port can start or stop acquisition. A pinned port that is taken is logged as an
error and the application runs without the API.

## Finding an instance

Each instance listens on a port the OS picks, so two instances on one host (a
`--simulate` run beside the rig's, say) never collide. While it listens, the
service writes `instance-<pid>.json` to the discovery directory and removes it
on stop:

```json
{"schemaVersion": 1, "pid": 1234, "processStartTime": 1763560000.12,
 "host": "127.0.0.1", "port": 41877, "simulated": false}
```

The directory is `$XDG_RUNTIME_DIR/reachaq/api`, or
`<tmp>/reachaq-runtime-<uid>/reachaq/api` without one (beside the application
lock), and `%LOCALAPPDATA%eachAQpi` on Windows. A record whose process has
ended, or whose pid now belongs to another process, is ignored and removed by the
next reader. `reachaq.core.api.discovery.list_api_instances()` lists the live
ones. The log line "Local API listening on ..." names the port and the record.

## Protocol

One TCP connection carries newline-delimited JSON: one UTF-8 JSON object per
line, at most 1 MiB. Every connected client receives every event.

### Requests

```json
{"type": "request", "nonce": "a1", "command": "GET_STATUS", "customCommand": null, "data": null}
```

`command` is a name or its integer id:

| Command | Id | Result |
|---|---:|---|
| `NONE` | 0 | `success`, no data. |
| `START_ACQUISITION` | 1 | `pending_with_notification`, then a second response with the outcome. |
| `STOP_ACQUISITION` | 2 | As for start, but refused (`failed`, `command_error`) while a session is arming, recording, stopping, analyzing or aborting, as the GUI disables System Mode then. It once aborted the recording and deleted the session. Stop the recording and wait for the session to be saved first. |
| `GET_STATUS` | 3 | `data` is the reachAQ status (`ReachAQSystemStatus`, the same payload as the `systemStatus` event). |
| `GET_CONFIGURATION` | 4 | `data` has `application_version`, `device_id`, `configuration_location`, `data_location`, `animal_location`, `log_location`, `inference_model`. |
| `USER_DEFINED` | 100 | Accepted, not yet acted on. |

`nonce` is echoed on every response to the request. Choose a unique one per
request; the final response of a pending command goes to the connection that
asked, or to every connection if that one has closed.

### Responses

```json
{"type": "response", "nonce": "a1", "command": "GET_STATUS", "result": "success",
 "errorCode": "none", "errorMessage": null, "data": {...}}
```

`result` is `success`, `failed`, `exception`, `unrecognized` or
`pending_with_notification`. `errorCode` is `none`, `command_error` or
`system_error`. Malformed JSON, an unknown command or an unknown message type
gets `unrecognized` / `command_error`.

### Events

```json
{"type": "event", "kind": 1331, "name": "sessionStarted", "when": "2026-11-18T12:00:00.123456",
 "index": 606361223517279, "context": {"session_id": "20261118_rig_001", ...}}
```

`kind` and `name` are the `ApiEventKind` id and member name, the same values the
hourly `*_events.csv` files and each session's `streams/events.csv` record.
`index` is the event's `perf_counter_ns` stamp. In `context`, event enums appear
by member name (`"mode": "RUNNING"`), times as ISO 8601, non-finite numbers as
`null`. A client that stops reading for 4096 lines is disconnected.

## Client

`reachaq.core.api.client.ApiClient` is a small blocking client. With no
`port` it connects to the one running instance, or to the one matching
`simulated=True` or `False`; when none or several match it raises
`ApiInstanceNotFound` listing them. `ApiClient(port=...)` skips discovery.

```python
from reachaq.core.api.client import ApiClient

with ApiClient() as client:
    print(client.request("GET_STATUS")["data"]["acquisition_state"])
    pending = client.request("START_ACQUISITION", nonce="start-1")
    final = client.wait_response("start-1", timeout=30)
    event = client.events.get(timeout=5)
```

## Event ids

Ids that reachAQ rigs have recorded are fixed: changing one would misread old
event files. `tests/core/api_recorded_events_test.py` checks every
id, name and payload repr against rows copied from christielab10's event files.
Members marked "assigned" in `kinds.py` had never been recorded when reachAQ
took over the enum, so their ids are reachAQ's choice.
