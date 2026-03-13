from agent.tools import MemorySearchTool, TaskSchedulerTool
from agent_build.agent1.tools import MemorySearchTool as LegacyMemorySearchTool
from agent_build.agent1.tools import TaskSchedulerTool as LegacyTaskSchedulerTool


def test_agent1_tools_module_reexports_from_agent_tools():
    assert LegacyMemorySearchTool is MemorySearchTool
    assert LegacyTaskSchedulerTool is TaskSchedulerTool
