# === Imports ===
import logging
import json
from helpers import query_pinecone, call_llm_robust, create_mcp_message

# === 4.1. Context Librarian Agent ===
def agent_context_librarian(mcp_message, client, index, embedding_model, namespace_context):
    """Retrieves the appropriate Semantic Blueprint."""
    logging.info("[Librarian] Activated. Analyzing intent...")
    try:
        requested_intent = mcp_message['content'].get('intent_query')
        if not requested_intent:
            raise ValueError("Librarian requires 'intent_query' in the input content.")

        results = query_pinecone(
            query_text=requested_intent,
            namespace=namespace_context,
            top_k=1,
            index=index,
            client=client,
            embedding_model=embedding_model
        )

        if results:
            match = results[0]
            logging.info(f"[Librarian] Found blueprint '{match['id']}' (Score: {match['score']:.2f})")
            blueprint_json = match['metadata']['blueprint_json']
            content = {"blueprint_json": blueprint_json}
        else:
            logging.warning("[Librarian] No blueprint found. Returning default.")
            content = {"blueprint_json": json.dumps({"instruction": "Generate content neutrally."})}

        return create_mcp_message("Librarian", content)
    except Exception as e:
        logging.error(f"[Librarian] Error: {e}")
        raise e

# === 4.2. Researcher Agent ===
def agent_researcher(mcp_message, client, index, generation_model, embedding_model, namespace_knowledge):
    """Retrieves facts with High-Fidelity (k=15)."""
    logging.info("[Researcher] Activated. Gathering evidence...")
    try:
        topic_query = mcp_message['content'].get('topic_query')
        if not topic_query:
            raise ValueError("Researcher requires 'topic_query'.")

        # UPGRADE: k=15 ensures all ingested applicant documents are caught
        results = query_pinecone(
            query_text=topic_query,
            namespace=namespace_knowledge,
            top_k=15, 
            index=index,
            client=client,
            embedding_model=embedding_model
        )

        if not results:
            return create_mcp_message("Researcher", {"facts": "No evidence found."})

        context_text = "\n\n".join([f"SOURCE: {m['metadata'].get('source')}\nCONTENT: {m['metadata'].get('text')}" for m in results])
        system_prompt = "Synthesize evidence into a factual report. Cite sources. If data is missing, state it."
        user_prompt = f"Objective: {topic_query}\n\nEvidence:\n{context_text}"

        facts = call_llm_robust(system_prompt, user_prompt, client, generation_model)
        return create_mcp_message("Researcher", {"facts": facts})
    except Exception as e:
        logging.error(f"[Researcher] Error: {e}")
        raise e

# === 4.3. Writer Agent (FIXED FOR STEP 5 CRASH) ===
def agent_writer(mcp_message, client, generation_model):
    """Generates the final report with robust key handling."""
    logging.info("[Writer] Activated. Synthesizing report...")
    try:
        content = mcp_message['content']
        
        # Robust Key Resolution: Look for standard keys or common variations
        raw_blueprint = content.get('blueprint') or content.get('blueprint_json') or content.get('instruction')
        raw_facts = content.get('facts') or content.get('evidence') or content.get('summary')
        previous_content = content.get('previous_content')

        # Fallback: If keys are mismatched, take values by index
        if not raw_blueprint and len(content.values()) > 0:
            raw_blueprint = list(content.values())[0]
        if not raw_facts and len(content.values()) > 1:
            raw_facts = list(content.values())[1]

        # Extract text if the value is a dictionary (common with Summarizer output)
        def get_text(val):
            if isinstance(val, dict):
                return val.get('blueprint_json') or val.get('summary') or val.get('facts') or str(val)
            return val

        blueprint = get_text(raw_blueprint)
        facts = get_text(raw_facts)

        if not blueprint or not (facts or previous_content):
            raise ValueError(f"Writer missing inputs. Keys received: {list(content.keys())}")

        system_prompt = "Apply Blueprint logic to Evidence to produce a deterministic audit report."
        user_prompt = f"--- BLUEPRINT ---\n{blueprint}\n\n--- EVIDENCE ---\n{facts if facts else previous_content}"

        report = call_llm_robust(system_prompt, user_prompt, client, generation_model)
        return create_mcp_message("Writer", {"report": report})
    except Exception as e:
        logging.error(f"[Writer] Error: {e}")
        raise e

# === 4.4. Summarizer Agent ===
def agent_summarizer(mcp_message, client, generation_model):
    """Reduces context density."""
    logging.info("[Summarizer] Activated. Compressing context...")
    try:
        text = mcp_message['content'].get('text_to_summarize')
        objective = mcp_message['content'].get('summary_objective')

        if isinstance(text, dict):
            text = text.get('facts') or text.get('report') or str(text)

        system_prompt = "Summarize the text based on the objective."
        user_prompt = f"Objective: {objective}\n\nText: {text}"

        summary = call_llm_robust(system_prompt, user_prompt, client, generation_model)
        return create_mcp_message("Summarizer", {"summary": summary})
    except Exception as e:
        logging.error(f"[Summarizer] Error: {e}")
        raise e