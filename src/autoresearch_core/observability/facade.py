"""Generic Mission Control facade used by UI and CLI composition layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class MissionControlFacade:
    """Thin facade over injected query/control services."""

    queries: Any
    controller: Any
    writer: Any | None = None

    def get_summary(self, *, limit: int = 10) -> dict[str, Any]:
        return cast(dict[str, Any], self.queries.get_mission_control_summary(limit=limit))

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.queries.get_run_detail(run_id))

    def list_runs(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.queries.list_runs(limit=limit))

    def get_llm_call(self, llm_call_id: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.queries.get_llm_call(llm_call_id))

    def list_tool_calls(
        self,
        *,
        llm_call_id: str,
        limit: int = 50,
        include_payload: bool = True,
    ) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self.queries.list_tool_calls(
                llm_call_id=llm_call_id,
                limit=limit,
                include_payload=include_payload,
            ),
        )

    def get_loop_status(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.queries.get_loop_status())

    def list_recent_decisions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.queries.list_recent_decisions(limit=limit))

    def list_recent_llm_calls(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.queries.list_recent_llm_calls(limit=limit))

    def search_entity(self, fragment: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.queries.resolve_id_fragment(fragment))

    def request_pause(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.controller.request_pause())

    def resume_loop(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.controller.resume_loop())

    def request_stop(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.controller.request_stop())

    def request_terminate(self) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self.controller.request_terminate())

    def enqueue_proposal(
        self,
        *,
        description: str,
        overrides: list[str],
        requested_by: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.controller.enqueue_proposal(
                description=description,
                overrides=overrides,
                requested_by=requested_by,
                notes=notes,
            ),
        )
