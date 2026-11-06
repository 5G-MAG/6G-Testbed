# Contributing

Thanks for your interest in improving the 6G AI Traffic Characterization Testbed.

## Getting Started

- Fork the repo and create a feature branch.
- Copy the environment template and add required API keys:
  - `cp .env.example .env`
- Use Python 3.10+ or Docker.

## Development Workflow

- Local run:
  - `python orchestrator.py --list-scenarios`
- Docker run:
  - `make build`
  - `./docker/run.sh python orchestrator.py --scenario chat_basic --profile ideal_6g --runs 1`

## Code Style

- Format: `make format`
- Lint: `make lint`
- Type check: `make typecheck`

## Reporting Issues

Please include:
- Scenario + profile used
- Steps to reproduce
- Error logs from `logs/`
- OS and Python/Docker versions

## Pull Requests

- Keep changes focused.
- Update docs and configs alongside code changes.
- Add tests if behavior changes.
