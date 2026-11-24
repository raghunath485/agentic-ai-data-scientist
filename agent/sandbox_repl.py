"""
Self-Healing Dynamic Python REPL Sandbox.
Executes code against active DataFrames, captures stdout and plot artifacts,
and performs automated self-healing upon syntax or runtime exceptions.
"""

import sys
import io
import traceback
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from agent.llm_client import AgentLLMClient


class SelfHealingREPL:
    def __init__(self, llm_client: Optional[AgentLLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or AgentLLMClient()
        self.max_retries = max_retries

    def execute_with_self_healing(
        self,
        code_str: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes code within the provided environment context.
        If an exception occurs, feeds the error and traceback back to the LLM to self-heal.
        """
        history_attempts: List[Dict[str, str]] = []
        current_code = self._clean_code(code_str)

        for attempt in range(self.max_retries + 1):
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture

            try:
                # Execute in local sandbox scope
                local_scope = execution_context.copy()
                exec(current_code, local_scope)
                sys.stdout = old_stdout
                output_str = stdout_capture.getvalue()

                # Extract result if variable 'result' was defined
                result_obj = local_scope.get("result", None)

                return {
                    "status": "success",
                    "final_code": current_code,
                    "stdout": output_str,
                    "result": result_obj,
                    "attempts_needed": attempt + 1,
                    "repair_history": history_attempts
                }

            except Exception as e:
                sys.stdout = old_stdout
                tb_str = traceback.format_exc()
                history_attempts.append({
                    "attempt": attempt + 1,
                    "failed_code": current_code,
                    "error": str(e),
                    "traceback": tb_str
                })

                if attempt < self.max_retries:
                    # Self-healing prompt
                    repair_prompt = f"""The following Python code failed with an exception:

```python
{current_code}
```

Error:
{tb_str}

Context available in scope:
- df: active pandas DataFrame (columns: {list(execution_context.get('df', pd.DataFrame()).columns)})
- pd: pandas module
- np: numpy module

Please provide ONLY the repaired, valid Python code to fix this issue. Do not include markdown explanations.
"""
                    repaired = self.llm.query(
                        repair_prompt,
                        system_prompt="You are an expert Python debugger and data scientist. Output ONLY runnable, corrected Python code."
                    )
                    current_code = self._clean_code(repaired)

        return {
            "status": "failed",
            "final_code": current_code,
            "error": history_attempts[-1]["error"] if history_attempts else "Unknown execution error",
            "repair_history": history_attempts
        }

    @staticmethod
    def _clean_code(raw_code: str) -> str:
        """Strips markdown code fences and whitespace."""
        code = raw_code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
