from eval.executors.agent1_runtime_executor import Agent1RuntimeExecutor


def test_augment_eval_prompt_includes_verification_and_calculator_guidance():
    executor = Agent1RuntimeExecutor()

    prompt = executor._augment_eval_prompt("Solve the task.", [])

    assert "Evaluation execution guidance:" in prompt
    assert "Do not answer from search-result snippets alone" in prompt
    assert "extract the target field from the exact nearby passage" in prompt
    assert "use a single expression with direct function calls and bindings" in prompt
    assert "do not use import, lambda, __import__, or attribute access like math.sin" in prompt
    assert "verify the requested output field and counting convention" in prompt


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
