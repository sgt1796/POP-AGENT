from typing import Dict


class BashExecApprovalPrompter:
    """Prompt the terminal user for medium/high risk bash_exec commands."""

    def __call__(self, request: Dict[str, object]) -> bool:
        try:
            command = str(request.get("command", "")).strip()
            cwd = str(request.get("cwd", "")).strip()
            risk = str(request.get("risk", "")).strip() or "unknown"
            justification = str(request.get("justification", "")).strip()

            print("\n[bash_exec] Approval requested.")
            print(f"[bash_exec] risk={risk}")
            print(f"[bash_exec] cwd={cwd}")
            print(f"[bash_exec] command={command}")
            if justification:
                print(f"[bash_exec] justification={justification}")
            else:
                print("[bash_exec] justification=(none)")

            decision = input("[bash_exec] Allow this command? [y/N]: ").strip().lower()
            return decision in {"y", "yes"}
        except Exception as exc:
            print(f"[bash_exec] approval prompt warning: {exc}")
            return False
