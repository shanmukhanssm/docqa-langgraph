
from langgraph.graph import StateGraph, START, END

from nodes import GraphState, preprocess_node, extract_node, summarize_node, answer_node


def route_start(state: GraphState) -> str:
    """decides the entry point:
    if the state has a question -> go directly to node 4
    otherwise -> start from node 1 with the file"""
    if state.get("question"):
        return "answer"
    return "preprocess"


def route_after_preprocess(state: GraphState) -> str:
    """after node 1 : continue only if the file was accepted"""
    if state.get("file_ok"):
        return "extract"
    return "reject"


def route_after_extract(state: GraphState) -> str:
    """after node 2 : continue only if we actually got some text"""
    if state.get("num_pages", 0) > 0:
        return "summarize"
    return "reject"


def build_graph():
    """creates and compiles the langgraph graph with all the 4 nodes"""
    builder = StateGraph(GraphState)

    # add all the 4 nodes
    builder.add_node("preprocess", preprocess_node)   # node 1
    builder.add_node("extract", extract_node)         # node 2
    builder.add_node("summarize", summarize_node)     # node 3
    builder.add_node("answer", answer_node)           # node 4

    # smart entry point (new file -> node 1, question -> node 4)
    builder.add_conditional_edges(START, route_start,
                                  {"preprocess": "preprocess", "answer": "answer"})

    # node 1 -> node 2 (only if the file is valid, otherwise straight to END)
    builder.add_conditional_edges("preprocess", route_after_preprocess,
                                  {"extract": "extract", "reject": END})

    # node 2 -> node 3 (only if text was found, otherwise straight to END)
    builder.add_conditional_edges("extract", route_after_extract,
                                  {"summarize": "summarize", "reject": END})

    # node 3 and node 4 finish the flow
    builder.add_edge("summarize", END)
    builder.add_edge("answer", END)

    return builder.compile()
