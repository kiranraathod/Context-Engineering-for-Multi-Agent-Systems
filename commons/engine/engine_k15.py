# === Imports ===
import logging
import time
import json
import copy
import re
from helpers import call_llm_robust, create_mcp_message, count_tokens
from registry import AGENT_TOOLKIT

# === 6.1. The Tracer ===
class ExecutionTrace:
    """Logs the entire execution flow for debugging, cost analysis, and duration tracking."""
    def __init__(self, goal):
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.start_time = time.time()
        self.duration = 0
        logging.info(f"ExecutionTrace initialized for goal: '{self.goal}'")

    def log_plan(self, plan):
        self.plan = plan
        logging.info("Plan has been logged to the trace.")

    def log_step(self, step_num, agent, planned_input, mcp_output, resolved_input, tokens_in=0, tokens_out=0):
        """Logs the details of a single execution step, including real-time token telemetry."""
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": mcp_output.get('content') if isinstance(mcp_output, dict) else mcp_output,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            # Telemetry for cost efficiency: specifically tracking context reduction
            "tokens_saved": max(0, tokens_in - tokens_out) if agent == "Summarizer" else 0
        })
        logging.info(f"Step {step_num} ({agent}) logged. [Tokens In: {tokens_in}, Tokens Out: {tokens_out}]")

    def finalize(self, status, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time
        logging.info(f"Trace finalized: {status}. Total Duration: {self.duration:.2f}s")

# === 6.2. The Planner ===
def planner(goal, capabilities, client, generation_model):
    """
    Analyzes the goal and generates a structured Execution Plan.
    Verified Signature: 4 parameters.
    """
    logging.info("Planner activated. Synthesizing execution strategy...")
    system_prompt = f"""
You are the strategic core of the Context Engine. Analyze the user's high-level GOAL and create a step-by-step EXECUTION PLAN.

AVAILABLE CAPABILITIES
---
{capabilities}
---

INSTRUCTIONS:
1. Output MUST be a single JSON object with a "plan" key containing a list of step objects.
2. Use Context Chaining: format placeholders as "$$STEP_N_OUTPUT$$".
"""
    try:
        plan_json_string = call_llm_robust(
            system_prompt,
            goal,
            client=client,
            generation_model=generation_model,
            json_mode=True
        )
        plan_data = json.loads(plan_json_string)
        return plan_data["plan"]
    except Exception as e:
        logging.error(f"Planner failed to generate a valid plan: {e}")
        raise e

# === 6.3. The Executor ===
def resolve_dependencies(input_params, state):
    """
    Upgraded Resolver: Uses Regex to find $$STEP_N_OUTPUT$$ placeholders within strings,
    making context chaining resilient against extraneous LLM text.
    """
    resolved_input = copy.deepcopy(input_params)
    # Pattern to match $$STEP_1_OUTPUT$$, $$STEP_2_OUTPUT$$, etc.
    pattern = r"\$\$(STEP_\d+_OUTPUT)\$\$"

    def resolve(value):
        if isinstance(value, str):
            matches = re.findall(pattern, value)
            if not matches:
                return value
            
            # If the value is EXACTLY one placeholder, return the raw object (could be dict/list)
            if re.fullmatch(pattern, value.strip()):
                ref_key = matches[0]
                return state.get(ref_key, value)
            
            # If the placeholder is embedded in text, replace with string representation
            for ref_key in matches:
                replacement = str(state.get(ref_key, f"$${ref_key}$$"))
                value = value.replace(f"$${ref_key}$$", replacement)
            return value
            
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(resolved_input)

def context_engine(goal, client, pc, index_name, generation_model, embedding_model, namespace_context, namespace_knowledge):
    """
    Main entry point for orchestration.
    Verified Signature: 8 parameters.
    """
    logging.info(f"--- [Context Engine] Initializing Task: {goal} ---")
    trace = ExecutionTrace(goal)
    registry = AGENT_TOOLKIT

    try:
        index = pc.Index(index_name)
        # Registry Pathing: Using verified registry logic
        capabilities = registry.get_capabilities_description()
        
        plan = planner(goal, capabilities, client=client, generation_model=generation_model)
        trace.log_plan(plan)
    except Exception as e:
        trace.finalize(f"Init Error: {e}")
        return None, trace

    state = {}
    for step in plan:
        step_num = step.get("step")
        agent_name = step.get("agent")
        planned_input = step.get("input")

        logging.info(f"--- [Step {step_num}] Executing {agent_name} ---")
        try:
            handler = registry.get_handler(
                agent_name,
                client=client,
                index=index,
                generation_model=generation_model,
                embedding_model=embedding_model,
                namespace_context=namespace_context,
                namespace_knowledge=namespace_knowledge
            )
            
            # 1. Regex-based Resolution
            resolved_input = resolve_dependencies(planned_input, state)
            
            # 2. Token Accountability (Input)
            t_in = count_tokens(str(resolved_input))
            
            # 3. Execution via MCP messaging
            mcp_request = create_mcp_message("Engine", resolved_input)
            mcp_response = handler(mcp_request)
            output_data = mcp_response.get("content")
            
            # 4. Token Accountability (Output)
            t_out = count_tokens(str(output_data))
            
            # 5. State Management & Telemetry Logging
            state[f"STEP_{step_num}_OUTPUT"] = output_data
            trace.log_step(
                step_num, 
                agent_name, 
                planned_input, 
                mcp_response, 
                resolved_input, 
                tokens_in=t_in, 
                tokens_out=t_out
            )
            
        except Exception as e:
            msg = f"Step {step_num} ({agent_name}) failed: {e}"
            logging.error(msg)
            trace.finalize(f"Fatal Step Error: {msg}")
            return None, trace

    # --- Finalization ---
    final_key = f"STEP_{len(plan)}_OUTPUT"
    final_output = state.get(final_key)
    trace.finalize("Success", final_output)
    return final_output, trace
