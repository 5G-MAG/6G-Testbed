"""
Multimodal Scenario for the 6G AI Traffic Testbed.

Executes image or PDF + text prompts using providers that support multimodal
input. Each prompt can specify its own media (image or document) via the
`media:` list in the scenario config. The legacy `image_paths:` shape is
preserved for back-compat.
"""

import json
import time
from pathlib import Path

from .base import BaseScenario, ScenarioResult


class MultimodalScenario(BaseScenario):
    """
    Multimodal scenario for image/document + text analysis.

    Two config shapes are supported:

    1. ``media:`` list — one entry per prompt, each either
       ``{image: path}`` or ``{document: path}``:

         media:
           - {document: "../../paper.pdf"}
           - {image: "examples/assets/diagram.png"}
         prompts:
           - "Summarize this paper"
           - "Describe the diagram"

    2. ``image_paths:`` list (back-compat) — paired by index with prompts,
       last entry reused for any extra prompts.

    Provider methods used per media kind:
        image    -> client.generate_content_with_image(prompt, image_path, model)
        document -> client.generate_content_with_document(prompt, document_path, model)
    """

    @property
    def scenario_type(self) -> str:
        return "multimodal"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute multimodal requests."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gemini-3-flash-preview")
        model_alias = self._anonymizer.model_alias(model)
        prompts = self.config.get("prompts", ["Describe the input."])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        try:
            media_list = self._build_media_list(prompts)
        except ValueError as e:
            result.success = False
            result.error_message = str(e)
            return result

        for turn_index, prompt in enumerate(prompts):
            self._wait_between_prompts(turn_index)
            media = media_list[min(turn_index, len(media_list) - 1)]
            kind = media["kind"]            # "image" or "document"
            resolved_path = media["path"]   # already a Path, already resolved

            t_request_start = time.time()

            if not resolved_path.exists():
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=0,
                    error_type=f"{kind.capitalize()} not found: {resolved_path}",
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = f"{kind.capitalize()} not found: {resolved_path}"
                break

            try:
                if kind == "image":
                    if not hasattr(self.client, "generate_content_with_image"):
                        raise NotImplementedError(
                            "Provider does not support image input"
                        )
                    response = self.client.generate_content_with_image(
                        prompt=prompt,
                        image_path=str(resolved_path),
                        model=model,
                    )
                else:  # document
                    if not hasattr(self.client, "generate_content_with_document"):
                        raise NotImplementedError(
                            "Provider does not support document input"
                        )
                    response = self.client.generate_content_with_document(
                        prompt=prompt,
                        document_path=str(resolved_path),
                        model=model,
                    )

                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                if tokens_in is None:
                    tokens_in = self.client.estimate_tokens(prompt, model)
                if tokens_out is None:
                    tokens_out = self.client.estimate_tokens(response.content or "", model)

                meta = {"model": model_alias, "media_kind": kind}
                path_key = f"{kind}_path"
                meta[path_key] = str(resolved_path)
                if kind == "document":
                    try:
                        meta["document_bytes"] = resolved_path.stat().st_size
                    except OSError:
                        pass

                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    request_bytes=response.request_bytes,
                    response_bytes=response.response_bytes,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    t_request_start=t_request_start,
                    latency_sec=response.latency_sec,
                    http_status=200,
                    success=True,
                    is_streaming=False,
                    trace_request=getattr(response, "request_payload", None),
                    trace_response=getattr(response, "response_payload", None),
                    trace_note=f"multimodal_{kind}",
                    metadata=json.dumps(meta),
                )
                self.logger.log(record)
                result.log_records.append(record)

                result.turn_count += 1
                result.api_call_count += 1
                result.total_latency_sec += response.latency_sec
                result.total_request_bytes += response.request_bytes
                result.total_response_bytes += response.response_bytes
                if response.tokens_in:
                    result.total_tokens_in += response.tokens_in
                if response.tokens_out:
                    result.total_tokens_out += response.tokens_out

            except NotImplementedError as e:
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=501,
                    error_type=str(e),
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(e)
                break
            except Exception as e:
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=0,
                    error_type=str(e),
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(e)
                break

        return result

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _build_media_list(self, prompts: list) -> list[dict]:
        """Normalize `media:` or legacy `image_paths:` into a list of
        {'kind': 'image'|'document', 'path': Path}. One entry per prompt (or
        fewer; the last entry is reused for any extra prompts)."""
        media_entries = self.config.get("media")
        if media_entries:
            if isinstance(media_entries, dict):
                media_entries = [media_entries]
            out: list[dict] = []
            for idx, entry in enumerate(media_entries):
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(
                        f"media[{idx}] must be a single-key mapping like "
                        f"{{image: ...}} or {{document: ...}}, got {entry!r}"
                    )
                kind, raw_path = next(iter(entry.items()))
                if kind not in ("image", "document"):
                    raise ValueError(
                        f"media[{idx}] unknown kind '{kind}' "
                        f"(expected 'image' or 'document')"
                    )
                out.append({"kind": kind, "path": self._resolve_media_path(kind, raw_path)})
            if not out:
                raise ValueError("multimodal scenario: media: list is empty")
            return out

        # Legacy fallback — image-only list paired with prompts
        image_paths = self._load_image_paths_legacy()
        if not image_paths:
            raise ValueError(
                "No media configured: set `media:` (per-prompt) or `image_paths:`"
            )
        return [
            {"kind": "image", "path": self._resolve_media_path("image", p)}
            for p in image_paths
        ]

    def _load_image_paths_legacy(self) -> list[str]:
        image_paths = self.config.get("image_paths") or self.config.get("images") or []
        image_path = self.config.get("image_path")
        if image_path:
            image_paths = [image_path]
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        return image_paths

    def _resolve_media_path(self, kind: str, raw_path: str) -> Path:
        """Resolve against optional per-kind base dir (image_base_dir /
        document_base_dir) or return as-is. Relative paths are resolved
        against the orchestrator's CWD."""
        base_dir_key = f"{kind}_base_dir"
        base_dir = self.config.get(base_dir_key)
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir).expanduser() / path
        return path
