# =============================================================================
# 6G AI Traffic Characterization Testbed - Makefile
# =============================================================================

.PHONY: help test clean logs report \
        mcp mcp-full mcp-music mcp-trading mcp-playwright charts \
        trading playwright \
        full-test full-test-quick

# Default target
help:
	@echo ""
	@echo "6G AI Traffic Characterization Testbed"
	@echo "======================================="
	@echo ""
	@echo "Scenario Commands:"
	@echo "  make chat           Run chat scenario"
	@echo "  make agent          Run shopping agent scenario"
	@echo "  make search         Run web search agent scenario"
	@echo "  make image          Run image generation scenario"
	@echo "  make realtime       Run real-time conversation scenario"
	@echo "  make all            Run all scenarios"
	@echo "  make mcp            Run all MCP scenarios (quick)"
	@echo "  make mcp-full       Run all MCP scenarios (full)"
	@echo "  make mcp-music      Run music MCP scenarios (quick)"
	@echo "  make mcp-trading    Run trading MCP scenarios (quick)"
	@echo "  make mcp-playwright Run Playwright MCP scenarios (quick)"
	@echo "  make trading        Run single trading scenario"
	@echo "  make playwright     Run single Playwright scenario"
	@echo "  make full-test      Run full test suite (all scenarios, 30 runs)"
	@echo "  make full-test-quick  Run full test suite (3 runs, quick)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make logs           Show recent logs"
	@echo "  make charts         Generate charts from latest results"
	@echo "  make report         Generate analysis report"
	@echo "  make clean          Remove logs and captures"
	@echo ""
	@echo "Setup:"
	@echo "  1. Copy .env.example to .env and add your API keys"
	@echo "  2. pip install -r requirements.txt"
	@echo "  3. Ensure passwordless sudo for tc (netem)"
	@echo "  4. Run 'make test' to verify setup"
	@echo ""

# =============================================================================
# Quick Test
# =============================================================================

test:
	@echo "Running quick test (chat scenario, 2 runs)..."
	python orchestrator.py --scenario chat_basic --profile ideal_6g --runs 2

# =============================================================================
# Scenario Commands
# =============================================================================

chat:
	python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

chat-stream:
	python orchestrator.py --scenario chat_streaming --profile 5g_urban --runs 10

agent:
	python orchestrator.py --scenario shopping_agent --profile 5g_urban --runs 10

search:
	python orchestrator.py --scenario web_search_agent --profile 5g_urban --runs 10

image:
	python orchestrator.py --scenario image_generation --profile ideal_6g --runs 5

all:
	python orchestrator.py --scenario all --runs 10

# Direct web search scenarios (no MCP)
direct-search:
	python orchestrator.py --scenario direct_web_search --profile 5g_urban --runs 10

direct-search-burst:
	python orchestrator.py --scenario direct_web_search_burst --profile ideal_6g --runs 5

search-benchmark:
	python orchestrator.py --scenario parallel_search_benchmark --profile ideal_6g --runs 3

# Real-time conversational AI scenarios
realtime:
	python orchestrator.py --scenario realtime_text --profile 5g_urban --runs 5

realtime-interactive:
	python orchestrator.py --scenario realtime_interactive --profile 5g_urban --runs 5

realtime-technical:
	python orchestrator.py --scenario realtime_technical --profile 5g_urban --runs 5

realtime-audio:
	python orchestrator.py --scenario realtime_audio --profile ideal_6g --runs 5

# MCP scenario runner
mcp:
	bash run_mcp.sh --quick

mcp-full:
	bash run_mcp.sh --full

mcp-music:
	bash run_mcp.sh --group music --quick

mcp-trading:
	bash run_mcp.sh --group trading --quick

mcp-playwright:
	bash run_mcp.sh --group playwright --quick

# Individual MCP scenarios
trading:
	python orchestrator.py --scenario trading_market_data --profile 5g_urban --runs 5

playwright:
	python orchestrator.py --scenario playwright_web_test --profile 5g_urban --runs 5

# Full test suite (all scenarios, all profiles, report generation)
full-test:
	bash run_full_tests.sh --full

full-test-quick:
	bash run_full_tests.sh --quick

# Run with specific netem profiles
chat-netem:
	python orchestrator.py --scenario chat_basic --profile cell_edge --runs 10

agent-netem:
	python orchestrator.py --scenario shopping_agent --profile satellite --runs 10

direct-search-netem:
	python orchestrator.py --scenario direct_web_search --profile cell_edge --runs 10

# =============================================================================
# Utility Commands
# =============================================================================

logs:
	@echo "Recent experiment logs:"
	@ls -la logs/*.db 2>/dev/null || echo "No logs found"
	@echo ""
	@echo "Recent reports:"
	@ls -la reports/*.json 2>/dev/null || echo "No reports found"

charts:
	python generate_charts.py --latest

report:
	python -c "from analysis import TrafficVisualizer, MetricsCalculator; print('Analysis tools ready')"

clean:
	rm -f logs/test_mcp_*.log
	@echo "Cleaned MCP log files. Database and captures preserved."

clean-all: clean
	rm -f logs/traffic_logs.db
	rm -rf results/captures/*.pcap logs/traces
	@echo "Cleaned database, pcaps, and traces."

# =============================================================================
# Development
# =============================================================================

lint:
	python -m flake8 --max-line-length=100 .

format:
	python -m black .

typecheck:
	python -m mypy .

# =============================================================================
# Environment Setup
# =============================================================================

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
		echo "Please edit .env and add your API keys"; \
	else \
		echo ".env file already exists"; \
	fi
	pip install -r requirements.txt

validate-env:
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', 'set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET')"
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('BRAVE_API_KEY:', 'set' if os.environ.get('BRAVE_API_KEY') else 'NOT SET')"
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('SPOTIFY_CLIENT_ID:', 'set' if os.environ.get('SPOTIFY_CLIENT_ID') else 'NOT SET')"
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('ALPACA_API_KEY:', 'set' if os.environ.get('ALPACA_API_KEY') else 'NOT SET')"
