# Roadmap: integrating a real Dota 2 environment

This roadmap is intentionally not presented as a functioning adapter. The historical DotaService project exposes a gRPC interface and Gym-like reset/step pattern, but its documented environment targets older Python/Unix setups. A successful integration must be demonstrated against the user's current game build before claiming support.

Reference projects:

- https://github.com/TimZaman/dotaclient
- https://github.com/TimZaman/dotaservice
- https://github.com/NandhaKishorM/laya

## Integration gates

### Gate 1: environment starts

Use a separate WSL2/Ubuntu environment for the historical service and its dependencies. Pin the exact commit SHA and record the game build, OS, Python version and dependency lock. Confirm service health and a reset observation before attempting model inference.

### Gate 2: observation/action contract

Implement a versioned adapter with `reset(seed) -> observation` and `step(action) -> (observation, reward, terminated, info)`. Convert the protobuf observation into explicit numeric features, handle missing/invisible entities, reject unknown action schemas and validate coordinates and action legality.

### Gate 3: shadow mode

The policy reads observations and logs proposed actions, but the adapter does not execute them. Check feature ranges, missing values, action distribution, latency and schema drift on multiple resets.

### Gate 4: private-lobby execution

Enable actions only in an explicitly started local/custom-lobby session. Add a kill switch, bounded action rate, exception recovery and a per-episode artifact folder. Do not use this research bot in public matchmaking.

### Gate 5: live-environment learning

Only after Gates 1–4 pass should we wire real observations/transitions into PPO or another compatible learner. The current tabular Q-table does not transfer to Dota 2: its state abstraction and reward model are synthetic. Live-game learning requires a new observation encoder, action space, reward design and independently validated evaluation protocol.

## Laya's role

Laya is a typed decision engine, not an RL trainer and not a Dota-playing model. Keep it at a slower strategic layer if experiments show it helps. Measure decision accuracy, latency and downstream reward; remove it if it adds latency without improving decisions. Avoid calling a model per game tick.
