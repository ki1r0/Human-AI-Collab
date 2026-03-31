"""Agent module — VLM reasoning, cognitive worker, and LangGraph agent."""

from .parser import parse_json_response

try:
    from .worker import cognitive_worker
    from .graph import AgentGraph
    from .reason2 import reason2_decide, call_reason2
except ImportError:
    pass  # requires runtime.config + langgraph (available at runtime)
