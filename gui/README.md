# TradingAgents web GUI

A local web dashboard for this repository, launched with:

```bash
./start.sh --web
```

It is adapted from [TheLocalLab/TradingAgents-GUI](https://github.com/TheLocalLab/TradingAgents-GUI)
(Apache-2.0), which wraps the upstream TradingAgents framework. The UI,
provider catalog, key management, report browser, and chat are theirs; the
run backend in `run_worker.py` was rewritten for the TradingAgents **v0.5.2**
API used here:

- `TradingAgentsGraph.create_run_state()` builds the initial state (memory
  settlement, instrument identity, past context) instead of a hand-built dict.
- `stream_run()` carries the agents' messages and cumulative state, including
  analysts running as subgraphs.
- `begin_checkpoint()` / `checkpoint_input()` / `end_checkpoint()` make the
  checkpoint toggle real.
- `record_decision()` and `write_report_tree()` give the GUI the same memory
  log and on-disk report tree as the CLI.
- The rating import moved to `tradingagents.agents.rating`.

The v0.2.5-only report-length control was removed because v0.5.2 has no
`report_brevity` config key. Portfolio input and explicit crypto asset
selection are not exposed by this UI yet.
