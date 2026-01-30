"""
Direct Web Search Scenarios for the 6G AI Traffic Testbed.

Implements multi-threaded web search WITHOUT MCP servers.
Uses direct HTTP requests to search engines (Google, DuckDuckGo).
"""

import json
import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

import requests

from .base import BaseScenario, ScenarioResult
from clients.base import ChatMessage, MessageRole
from analysis.logger import LogRecord


class SearchEngine(Enum):
    """Supported search engines."""
    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchResult:
    """Result from a single search request."""
    query: str
    engine: SearchEngine
    success: bool
    results: list[dict] = field(default_factory=list)

    # Traffic metrics
    request_bytes: int = 0
    response_bytes: int = 0
    latency_sec: float = 0.0
    http_status: int = 0

    # Error info
    error: Optional[str] = None

    # Raw response for debugging
    raw_response: Optional[str] = None


@dataclass
class ThreadedSearchResult:
    """Aggregated results from multi-threaded search."""
    queries: list[str]
    thread_count: int
    engine: SearchEngine

    # Aggregated metrics
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    total_latency_sec: float = 0.0

    # Per-thread timing
    wall_clock_time_sec: float = 0.0

    # Individual results
    search_results: list[SearchResult] = field(default_factory=list)


class DirectSearchClient:
    """
    Direct HTTP client for web search engines.

    Supports:
    - Google Custom Search API (requires API key + Search Engine ID)
    - DuckDuckGo HTML scraping (no API key required)
    """

    def __init__(
        self,
        engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        google_api_key: Optional[str] = None,
        google_cx: Optional[str] = None,
        timeout: float = 30.0,
        max_results: int = 10,
    ):
        self.engine = engine
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_SEARCH_API_KEY")
        self.google_cx = google_cx or os.environ.get("GOOGLE_SEARCH_CX")
        self.timeout = timeout
        self.max_results = max_results

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "6G-AI-Testbed/1.0 (Traffic Research)"
        })

    def search(self, query: str) -> SearchResult:
        """Perform a search using the configured engine."""
        if self.engine == SearchEngine.GOOGLE:
            return self._search_google(query)
        else:
            return self._search_duckduckgo(query)

    def _search_google(self, query: str) -> SearchResult:
        """Search using Google Custom Search API."""
        result = SearchResult(query=query, engine=SearchEngine.GOOGLE, success=False)

        if not self.google_api_key or not self.google_cx:
            result.error = "Google API key or CX not configured"
            return result

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": min(self.max_results, 10),  # Google max is 10 per request
        }

        t_start = time.time()
        try:
            # Calculate request size (approximate)
            request_url = f"{url}?{urllib.parse.urlencode(params)}"
            result.request_bytes = len(request_url.encode())

            response = self.session.get(url, params=params, timeout=self.timeout)
            result.latency_sec = time.time() - t_start
            result.http_status = response.status_code
            result.response_bytes = len(response.content)

            if response.status_code == 200:
                data = response.json()
                result.results = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                    for item in data.get("items", [])
                ]
                result.success = True
            else:
                result.error = f"HTTP {response.status_code}: {response.text[:200]}"
                result.raw_response = response.text[:1000]

        except requests.exceptions.Timeout:
            result.latency_sec = time.time() - t_start
            result.error = "Request timeout"
        except requests.exceptions.RequestException as e:
            result.latency_sec = time.time() - t_start
            result.error = str(e)
        except json.JSONDecodeError as e:
            result.latency_sec = time.time() - t_start
            result.error = f"JSON decode error: {e}"

        return result

    def _search_duckduckgo(self, query: str) -> SearchResult:
        """
        Search using DuckDuckGo HTML interface.

        No API key required. Uses the lite HTML version for simplicity.
        """
        result = SearchResult(query=query, engine=SearchEngine.DUCKDUCKGO, success=False)

        # DuckDuckGo lite HTML endpoint
        url = "https://lite.duckduckgo.com/lite/"
        data = {
            "q": query,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        t_start = time.time()
        try:
            # Calculate request size
            body = urllib.parse.urlencode(data).encode()
            result.request_bytes = len(body) + len(url.encode())

            response = self.session.post(
                url,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            result.latency_sec = time.time() - t_start
            result.http_status = response.status_code
            result.response_bytes = len(response.content)

            if response.status_code == 200:
                # Parse HTML response
                result.results = self._parse_duckduckgo_html(response.text)
                result.success = len(result.results) > 0
                if not result.success:
                    result.error = "No results found in HTML response"
            else:
                result.error = f"HTTP {response.status_code}"
                result.raw_response = response.text[:1000]

        except requests.exceptions.Timeout:
            result.latency_sec = time.time() - t_start
            result.error = "Request timeout"
        except requests.exceptions.RequestException as e:
            result.latency_sec = time.time() - t_start
            result.error = str(e)

        return result

    def _parse_duckduckgo_html(self, html: str) -> list[dict]:
        """Parse DuckDuckGo lite HTML for search results."""
        results = []

        # Pattern for result links in DuckDuckGo lite
        # Format: <a rel="nofollow" href="URL" class='result-link'>TITLE</a>
        link_pattern = re.compile(
            r'<a[^>]*class=[\'"]result-link[\'"][^>]*href=[\'"]([^\'"]+)[\'"][^>]*>([^<]+)</a>',
            re.IGNORECASE
        )

        # Also try alternative pattern
        alt_pattern = re.compile(
            r'<a[^>]*href=[\'"]([^\'"]+)[\'"][^>]*class=[\'"]result-link[\'"][^>]*>([^<]+)</a>',
            re.IGNORECASE
        )

        # Pattern for snippets (text after the link)
        snippet_pattern = re.compile(
            r'<td[^>]*class=[\'"]result-snippet[\'"][^>]*>([^<]+)',
            re.IGNORECASE
        )

        # Find all matches
        matches = link_pattern.findall(html) or alt_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, match in enumerate(matches[:self.max_results]):
            url, title = match
            snippet = snippets[i] if i < len(snippets) else ""

            # Decode URL if needed
            if url.startswith("//duckduckgo.com/l/?uddg="):
                # Extract actual URL from redirect
                try:
                    url = urllib.parse.unquote(url.split("uddg=")[1].split("&")[0])
                except:
                    pass

            results.append({
                "title": title.strip(),
                "url": url,
                "snippet": snippet.strip(),
            })

        # Fallback: simpler pattern if no results
        if not results:
            simple_pattern = re.compile(
                r'<a[^>]*href=[\'"]([^\'">]+)[\'"][^>]*>([^<]{10,})</a>',
                re.IGNORECASE
            )
            for match in simple_pattern.findall(html):
                url, title = match
                if url.startswith("http") and "duckduckgo" not in url.lower():
                    results.append({
                        "title": title.strip(),
                        "url": url,
                        "snippet": "",
                    })
                    if len(results) >= self.max_results:
                        break

        return results

    def close(self):
        """Close the session."""
        self.session.close()


class ThreadedSearchExecutor:
    """
    Multi-threaded search executor.

    Launches multiple threads to perform web searches in parallel,
    measuring traffic characteristics for each request.
    """

    def __init__(
        self,
        engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        max_workers: int = 5,
        **client_kwargs
    ):
        self.engine = engine
        self.max_workers = max_workers
        self.client_kwargs = client_kwargs

    def search_parallel(
        self,
        queries: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ThreadedSearchResult:
        """
        Execute multiple searches in parallel using a thread pool.

        Args:
            queries: List of search queries to execute
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            ThreadedSearchResult with aggregated metrics
        """
        result = ThreadedSearchResult(
            queries=queries,
            thread_count=min(self.max_workers, len(queries)),
            engine=self.engine,
            total_searches=len(queries),
        )

        t_wall_start = time.time()

        # Use ThreadPoolExecutor for concurrent searches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all search tasks
            future_to_query = {}
            for query in queries:
                # Each thread gets its own client for thread safety
                client = DirectSearchClient(
                    engine=self.engine,
                    **self.client_kwargs
                )
                future = executor.submit(self._search_worker, client, query)
                future_to_query[future] = query

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_query):
                search_result = future.result()
                result.search_results.append(search_result)

                # Aggregate metrics
                if search_result.success:
                    result.successful_searches += 1
                else:
                    result.failed_searches += 1

                result.total_request_bytes += search_result.request_bytes
                result.total_response_bytes += search_result.response_bytes
                result.total_latency_sec += search_result.latency_sec

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(queries))

        result.wall_clock_time_sec = time.time() - t_wall_start

        return result

    def _search_worker(
        self,
        client: DirectSearchClient,
        query: str
    ) -> SearchResult:
        """Worker function for thread pool."""
        try:
            result = client.search(query)
        finally:
            client.close()
        return result


class DirectWebSearchScenario(BaseScenario):
    """
    Direct web search scenario using multi-threaded HTTP requests.

    This scenario does NOT use MCP servers. Instead, it:
    1. Launches multiple threads to perform web searches
    2. Uses direct HTTP to Google/DuckDuckGo
    3. Optionally synthesizes results using an LLM

    Traffic pattern: Burst (parallel requests) + optional Query-Response (LLM)
    """

    def __init__(self, client, logger, config):
        super().__init__(client, logger, config)

        # Search configuration
        engine_name = config.get("search_engine", "duckduckgo").lower()
        self.engine = SearchEngine(engine_name)

        self.thread_count = config.get("thread_count", 5)
        self.search_timeout = config.get("search_timeout", 30.0)
        self.max_results_per_search = config.get("max_results", 10)
        self.synthesize_with_llm = config.get("synthesize_with_llm", True)

    @property
    def scenario_type(self) -> str:
        return "direct_web_search"

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        """Execute the direct web search scenario."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        model_alias = self._anonymizer.model_alias(model)

        # Get search queries from config or use defaults
        queries = self.config.get("queries", [
            "6G wireless technology latest developments 2024",
            "AI traffic patterns mobile networks",
            "3GPP release 19 features",
            "machine learning network optimization",
            "edge computing AI inference latency",
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        t_scenario_start = time.time()

        try:
            # Phase 1: Multi-threaded web searches
            print(f"  Launching {self.thread_count} threads for {len(queries)} searches...")

            executor = ThreadedSearchExecutor(
                engine=self.engine,
                max_workers=self.thread_count,
                timeout=self.search_timeout,
                max_results=self.max_results_per_search,
            )

            search_result = executor.search_parallel(
                queries,
                progress_callback=lambda done, total: print(
                    f"    Search progress: {done}/{total}"
                ) if done % 5 == 0 or done == total else None
            )

            # Log search phase metrics
            search_record = self._create_log_record(
                session_id=session_id,
                turn_index=0,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=search_result.total_request_bytes,
                response_bytes=search_result.total_response_bytes,
                t_request_start=t_scenario_start,
                latency_sec=search_result.wall_clock_time_sec,
                http_status=200 if search_result.successful_searches > 0 else 500,
                success=search_result.successful_searches > 0,
                tool_calls_count=search_result.total_searches,
                metadata=json.dumps({
                    "phase": "search",
                    "engine": self.engine.value,
                    "thread_count": self.thread_count,
                    "queries": len(queries),
                    "successful": search_result.successful_searches,
                    "failed": search_result.failed_searches,
                    "wall_clock_sec": search_result.wall_clock_time_sec,
                    "sum_latency_sec": search_result.total_latency_sec,
                })
            )
            self.logger.log(search_record)
            result.log_records.append(search_record)

            # Update result metrics for search phase
            result.turn_count += 1
            result.api_call_count += search_result.total_searches
            result.tool_calls_count += search_result.total_searches
            result.total_request_bytes += search_result.total_request_bytes
            result.total_response_bytes += search_result.total_response_bytes
            result.tool_total_latency_sec += search_result.total_latency_sec

            # Phase 2: Optional LLM synthesis
            if self.synthesize_with_llm and search_result.successful_searches > 0:
                print(f"  Synthesizing {search_result.successful_searches} search results with LLM...")

                # Prepare search results for LLM
                search_summary = self._format_search_results(search_result)

                synthesis_prompt = self.config.get("synthesis_prompt",
                    "Based on the following search results, provide a comprehensive summary "
                    "of the key findings and trends. Cite specific sources."
                )

                messages = [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content="You are a research assistant. Analyze the provided search results and synthesize a coherent summary."
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=f"{synthesis_prompt}\n\nSearch Results:\n{search_summary}"
                    )
                ]

                t_llm_start = time.time()

                # Check if streaming is requested
                if self.config.get("stream", False):
                    response = self.client.chat_streaming(messages, model=model)
                    result.ttft_sec = response.ttft
                    result.ttlt_sec = response.ttlt
                    request_bytes = response.request_bytes or (
                        sum(len(m.content.encode()) for m in messages) + 100
                    )
                    response_bytes = response.response_bytes or len(
                        response.total_content.encode()
                    )
                    tokens_in = response.tokens_in
                    tokens_out = response.tokens_out
                    t_first_token = response.t_first_chunk
                    t_last_token = response.t_last_chunk
                    latency_sec = response.ttlt or (time.time() - t_llm_start)
                else:
                    response = self.client.chat(messages, model=model, stream=False)
                    request_bytes = response.request_bytes
                    response_bytes = response.response_bytes
                    tokens_in = response.tokens_in
                    tokens_out = response.tokens_out
                    t_first_token = None
                    t_last_token = None
                    latency_sec = response.latency_sec

                if tokens_in is None:
                    tokens_in = self.client.estimate_message_tokens(messages, model)
                if tokens_out is None:
                    output_text = response.total_content if hasattr(response, "total_content") else response.content
                    tokens_out = self.client.estimate_tokens(output_text or "", model)

                # Log LLM synthesis
                llm_record = self._create_log_record(
                    session_id=session_id,
                    turn_index=1,
                    run_index=run_index,
                    network_profile=network_profile,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    t_request_start=t_llm_start,
                    t_first_token=t_first_token,
                    t_last_token=t_last_token,
                    latency_sec=latency_sec,
                    http_status=200,
                    success=True,
                    is_streaming=self.config.get("stream", False),
                    trace_request=getattr(response, "request_payload", None),
                    trace_response=getattr(response, "response_payload", None),
                    trace_events=getattr(response, "response_events", None),
                    trace_note="direct_search_synthesis",
                    metadata=json.dumps({
                        "phase": "synthesis",
                        "model": model_alias,
                        "input_results": search_result.successful_searches,
                    })
                )
                self.logger.log(llm_record)
                result.log_records.append(llm_record)

                # Update result metrics for LLM phase
                result.turn_count += 1
                result.api_call_count += 1
                result.total_request_bytes += request_bytes
                result.total_response_bytes += response_bytes
                result.total_tokens_in += tokens_in or 0
                result.total_tokens_out += tokens_out or 0

            result.total_latency_sec = time.time() - t_scenario_start
            result.success = search_result.successful_searches > 0

            if search_result.failed_searches > 0:
                errors = [
                    r.error for r in search_result.search_results
                    if not r.success and r.error
                ]
                result.metadata["search_errors"] = errors[:5]  # First 5 errors

            # Store detailed metrics
            result.metadata.update({
                "search_engine": self.engine.value,
                "thread_count": self.thread_count,
                "queries_count": len(queries),
                "successful_searches": search_result.successful_searches,
                "failed_searches": search_result.failed_searches,
                "search_wall_clock_sec": search_result.wall_clock_time_sec,
                "search_sum_latency_sec": search_result.total_latency_sec,
                "parallelism_factor": (
                    search_result.total_latency_sec / search_result.wall_clock_time_sec
                    if search_result.wall_clock_time_sec > 0 else 1.0
                ),
            })

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.total_latency_sec = time.time() - t_scenario_start

            error_record = self._create_log_record(
                session_id=session_id,
                turn_index=0,
                run_index=run_index,
                network_profile=network_profile,
                t_request_start=t_scenario_start,
                latency_sec=result.total_latency_sec,
                http_status=0,
                error_type=str(e),
                success=False,
            )
            self.logger.log(error_record)
            result.log_records.append(error_record)

        return result

    def _format_search_results(self, search_result: ThreadedSearchResult) -> str:
        """Format search results for LLM input."""
        lines = []

        for i, sr in enumerate(search_result.search_results):
            if not sr.success:
                continue

            lines.append(f"\n## Query: {sr.query}")
            lines.append(f"Engine: {sr.engine.value} | Results: {len(sr.results)}")

            for j, r in enumerate(sr.results[:5], 1):  # Top 5 per query
                lines.append(f"\n{j}. {r['title']}")
                lines.append(f"   URL: {r['url']}")
                if r['snippet']:
                    lines.append(f"   {r['snippet'][:200]}...")

        return "\n".join(lines)


class ParallelSearchBenchmarkScenario(BaseScenario):
    """
    Benchmark scenario for measuring parallel search performance.

    Runs searches with varying thread counts to measure:
    - Throughput vs parallelism
    - Per-thread overhead
    - Network saturation effects

    Useful for characterizing burst traffic patterns.
    """

    def __init__(self, client, logger, config):
        super().__init__(client, logger, config)

        engine_name = config.get("search_engine", "duckduckgo").lower()
        self.engine = SearchEngine(engine_name)

        # Benchmark configuration
        self.thread_counts = config.get("thread_counts", [1, 2, 5, 10, 20])
        self.queries_per_run = config.get("queries_per_run", 20)
        self.search_timeout = config.get("search_timeout", 30.0)

    @property
    def scenario_type(self) -> str:
        return "parallel_search_benchmark"

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        """Run benchmark with multiple thread counts."""
        session_id = self._create_session_id()

        # Generate benchmark queries
        base_queries = self.config.get("queries", [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "computer vision",
        ])

        # Expand to desired count by adding variations
        queries = []
        for i in range(self.queries_per_run):
            base = base_queries[i % len(base_queries)]
            queries.append(f"{base} {2024 - (i % 5)} trends")

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        t_start = time.time()
        benchmark_results = []

        try:
            for thread_count in self.thread_counts:
                print(f"  Benchmark: {thread_count} threads, {len(queries)} queries...")

                executor = ThreadedSearchExecutor(
                    engine=self.engine,
                    max_workers=thread_count,
                    timeout=self.search_timeout,
                )

                search_result = executor.search_parallel(queries)

                benchmark_data = {
                    "thread_count": thread_count,
                    "queries": len(queries),
                    "successful": search_result.successful_searches,
                    "failed": search_result.failed_searches,
                    "wall_clock_sec": search_result.wall_clock_time_sec,
                    "sum_latency_sec": search_result.total_latency_sec,
                    "throughput_qps": (
                        len(queries) / search_result.wall_clock_time_sec
                        if search_result.wall_clock_time_sec > 0 else 0
                    ),
                    "parallelism_factor": (
                        search_result.total_latency_sec / search_result.wall_clock_time_sec
                        if search_result.wall_clock_time_sec > 0 else 1.0
                    ),
                    "total_request_bytes": search_result.total_request_bytes,
                    "total_response_bytes": search_result.total_response_bytes,
                    "avg_latency_sec": (
                        search_result.total_latency_sec / len(queries)
                        if queries else 0
                    ),
                }
                benchmark_results.append(benchmark_data)

                # Log this benchmark run
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=self.thread_counts.index(thread_count),
                    run_index=run_index,
                    network_profile=network_profile,
                    request_bytes=search_result.total_request_bytes,
                    response_bytes=search_result.total_response_bytes,
                    t_request_start=time.time() - search_result.wall_clock_time_sec,
                    latency_sec=search_result.wall_clock_time_sec,
                    http_status=200 if search_result.successful_searches > 0 else 500,
                    success=search_result.successful_searches > 0,
                    tool_calls_count=len(queries),
                    metadata=json.dumps(benchmark_data)
                )
                self.logger.log(record)
                result.log_records.append(record)

                # Aggregate
                result.turn_count += 1
                result.api_call_count += len(queries)
                result.tool_calls_count += len(queries)
                result.total_request_bytes += search_result.total_request_bytes
                result.total_response_bytes += search_result.total_response_bytes
                result.tool_total_latency_sec += search_result.total_latency_sec

            result.total_latency_sec = time.time() - t_start
            result.success = True
            result.metadata["benchmark_results"] = benchmark_results

            # Print summary
            print("\n  Benchmark Summary:")
            print("  Threads | Wall Clock | Throughput | Parallelism")
            print("  --------|------------|------------|------------")
            for b in benchmark_results:
                print(f"  {b['thread_count']:7d} | {b['wall_clock_sec']:9.2f}s | "
                      f"{b['throughput_qps']:9.2f}/s | {b['parallelism_factor']:.2f}x")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.total_latency_sec = time.time() - t_start

        return result
