"""Optional LLM brain (anthropic), with rule fallback.

Advisory only: may add a natural-language note when the agent abstains or flags low confidence.
Disabled by default; validates its own output and on any problem the caller keeps the rule result.
The default deployment makes zero paid API calls and is fully deterministic. **Never changes the
recognized glosses or the translation.**
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def note(self, glosses, text: str, abstained: bool) -> Optional[str]:
        if not self.available():
            return None
        client = self._get_client()
        if client is None:
            return None
        state = "abstained" if abstained else "low-confidence"
        prompt = (f"A sign-language translator recognized the glosses {glosses} and produced the text "
                  f"'{text}', but {state}. In ONE short sentence, advise the user (e.g. ask them to repeat "
                  f"the signs more clearly). Do NOT change the translation.")
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=80, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            return ("".join(getattr(b, "text", "") for b in msg.content).strip()) or None
        except Exception as exc:
            logger.info("LLM note failed (%s)", exc)
            return None


__all__ = ["LLMBrain"]
