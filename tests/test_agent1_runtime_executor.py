from eval.executors.agent1_runtime_executor import Agent1RuntimeExecutor, _EvalSteeringGuard


class _FakeAgent:
    def __init__(self) -> None:
        self.messages = []
        self.tools = {
            "perplexity_search",
            "jina_web_snapshot",
            "perplexity_web_snapshot",
            "file_read",
        }

    def steer(self, message) -> None:
        self.messages.append(message)

    def remove_tool(self, name: str) -> bool:
        if name in self.tools:
            self.tools.remove(name)
            return True
        return False


def test_augment_eval_prompt_includes_verification_and_calculator_guidance():
    executor = Agent1RuntimeExecutor()

    prompt = executor._augment_eval_prompt("Solve the task.", [])

    assert "Evaluation execution guidance:" in prompt
    assert "Do not answer from search-result snippets alone" in prompt
    assert "extract the target field from the exact nearby passage" in prompt
    assert "recover the exact phrase in the exact title, chapter, page, or preview path" in prompt
    assert "use file_read with query and bounded context" in prompt
    assert "final_url, pdf_link_candidates, or content_preview" in prompt
    assert "saved_landing_page_path" in prompt
    assert "If jina_web_snapshot fails with a 4xx or proxy access error" in prompt
    assert "tangential names, generic summaries, or unverified numbers" in prompt
    assert "use a single expression with direct function calls and bindings" in prompt
    assert "do not use import, lambda, __import__, or attribute access like math.sin" in prompt
    assert "rewrite the expression with direct allowed calls or bindings" in prompt
    assert "Treat bash_exec as local-shell inspection only." in prompt
    assert "If bash_exec is blocked, that is a hard constraint" in prompt
    assert "Do not use calculator to open files, inspect text, or simulate scripting" in prompt
    assert "eligible candidates first and compute from those explicit values" in prompt
    assert "write down the exact source-backed operands" in prompt
    assert "verify the requested output field and counting convention" in prompt
    assert "requested precision or rounding rule" in prompt
    assert "nearest 0.001 of the reported unit requires three decimals" in prompt
    assert "do not echo the format template, labels, or extra units" in prompt
    assert "placeholder, copied template, or generic filler token" in prompt
    assert "spend one targeted verification call on the strongest candidate source" in prompt


def test_augment_eval_prompt_keeps_attachment_local_first_guidance():
    executor = Agent1RuntimeExecutor()

    prompt = executor._augment_eval_prompt(
        "Solve the task.",
        [{"workspace_path": "eval/runs/run/_attachments/sample/data.pdb"}],
    )

    assert "Required attachment files are preloaded in the workspace at:" in prompt
    assert "- eval/runs/run/_attachments/sample/data.pdb" in prompt
    assert "Local files are primary evidence for this task." in prompt
    assert "Open these exact local paths first with bounded local reads." in prompt


def test_eval_steering_guard_intervenes_on_hard_blocks_and_budget():
    agent = _FakeAgent()
    guard = _EvalSteeringGuard(agent, generic_web_budget=4)

    for _ in range(4):
        guard.on_event({"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {}}})

    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "bash_exec",
            "result": {"details": {"blocked": True, "block_reason": "command_not_allowed"}},
        }
    )
    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "calculator",
            "result": {"details": {"error": "only direct function calls are allowed"}},
        }
    )
    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "file_read",
            "result": {"details": {"error": "parse_error"}},
        }
    )
    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "jina_web_snapshot",
            "result": {
                "details": {
                    "error": "402 Client Error: Payment Required for url: https://r.jina.ai/https://example.com/page",
                    "jina_status_code": 402,
                    "url": "https://example.com/page",
                }
            },
        }
    )

    assert len(agent.messages) == 5
    texts = [message.content[0].text for message in agent.messages]
    assert "generic web discovery budget is exhausted" in texts[0]
    assert "tools are now disabled" in texts[0]
    assert "bash_exec is hard-blocked" in texts[1]
    assert "Calculator is arithmetic-only" in texts[2]
    assert "local artifact could not be parsed" in texts[3]
    assert "save it as local .html" in texts[4]
    assert "perplexity_search" not in agent.tools
    assert "jina_web_snapshot" not in agent.tools
    assert "perplexity_web_snapshot" not in agent.tools
    assert "file_read" in agent.tools


def test_eval_steering_guard_deduplicates_repeated_signals():
    agent = _FakeAgent()
    guard = _EvalSteeringGuard(agent, generic_web_budget=2)

    guard.on_event({"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {}}})
    guard.on_event({"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {}}})
    guard.on_event({"type": "tool_execution_end", "toolName": "perplexity_search", "result": {"details": {}}})
    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "bash_exec",
            "result": {"details": {"blocked": True, "block_reason": "blocked_shell_operator"}},
        }
    )
    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "bash_exec",
            "result": {"details": {"blocked": True, "block_reason": "blocked_shell_operator"}},
        }
    )

    assert len(agent.messages) == 2


def test_eval_steering_guard_surfaces_exact_landing_page_recovery_hints():
    agent = _FakeAgent()
    guard = _EvalSteeringGuard(agent)

    guard.on_event(
        {
            "type": "tool_execution_end",
            "toolName": "download_url_to_file",
            "result": {
                "details": {
                    "error": "unexpected_content_type",
                    "saved_landing_page_path": "C:\\Users\\sgt17\\OneDrive\\Desktop\\POP-agent\\A_Dark_Trace.html",
                    "final_url": "https://muse.jhu.edu/verify?url=%2Fpub%2F258%2Foa_monograph%2Fbook%2F24372%2Fpdf",
                }
            },
        }
    )

    assert len(agent.messages) == 1
    text = agent.messages[0].content[0].text
    assert "A_Dark_Trace.html" in text
    assert "https://muse.jhu.edu/verify" in text
