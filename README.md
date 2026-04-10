---
title: Aadil_env
emoji: ":truck:"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
startup_duration_timeout: 1h
---

# Warehouse Management Robot Planner

This project is an OpenEnv-style environment for warehouse automation planning. An AI agent receives a warehouse request and builds a structured robot execution plan covering goal identification, robot assignment, resource selection, zone routing, safety checks, battery-aware planning, obstacle-aware rerouting, and final task confirmation.

## Project structure

The OpenEnv-compatible structure is preserved:

- `my_env/env.py`
- `my_env/models.py`
- `my_env/tasks.py`
- `my_env/graders.py`
- `server/app.py`
- `inference.py`
- `openenv.yaml`
- `README.md`
- `config.py`

## Warehouse scenarios

- `easy`: Move one package from the receiving zone to rack B2 safely.
- `medium`: Pick three products from shelves `A1`, `B4`, and `C2` and deliver them to the packing zone.
- `hard`: Plan an urgent hospital supply order that avoids blocked aisle C, respects battery constraints, and delivers to the dispatch zone in priority mode.

## Supported planning features

- Robot type selection such as picker robot, carrier robot, and forklift robot
- Zone-based navigation across receiving, storage, packing, and dispatch
- Battery-aware planning with low-battery fallback behavior
- Obstacle-aware rerouting for blocked aisles and busy zones
- Priority handling for urgent shipments
- Safety checks including collision avoidance, overload checks, restricted zone warnings, and fragile-item handling
- Multi-step workflows such as pick, move, scan, place, and confirm
- Resource suggestions including barcode scanner, conveyor, forklift attachment, cart, and pallet support
- Inventory movement logic for inbound, picking, packing, and dispatch flows

## Environment variables

- `API_BASE_URL=https://api.openai.com/v1`
- `OPENAI_API_KEY=...`
- `MODEL_NAME=gpt-4o-mini`
- `ENV_SERVER_URL=http://127.0.0.1:7860`

## Running locally

Install dependencies and start the app on port `7860`:

```bash
pip install -r requirements.txt
python -m server.app
```

Run the local inference flow:

```bash
python inference.py
```

## OpenEnv and API behavior

- `GET /` returns a warehouse planner status message.
- `POST /reset` loads one of the `easy`, `medium`, or `hard` tasks.
- `GET /state` returns the current environment state.
- `POST /step` applies a warehouse planning action.

The environment keeps rewards strictly inside `(0, 1)` and preserves the expected OpenEnv JSON interaction pattern.

## Hugging Face Spaces

The repository remains Docker-compatible for Hugging Face Spaces and serves the FastAPI app on port `7860`. The inference script can use the HTTP server when available and automatically falls back to the local environment if the server is unavailable.

## Hackathon positioning

This project is designed to look like an intelligent warehouse automation planner rather than a renamed template. The tasks, grading logic, route handling, safety constraints, battery strategy, and inference actions are all tied directly to warehouse robotics operations.
