import re
from typing import TypedDict

from langgraph.graph import StateGraph, END

from .models import AssistantResponse
from .prompts import build_prompt
from .vectorstore import retrieve
from .config import MOCK_LLM


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "track",
    "cancel",
    "cancellation",
    "gift card",
    "giftcard",
    "support hours",
    "customer support",
]


class State(TypedDict, total=False):
    query: str
    intent: str
    context: list
    answer: str
    sources: list
    confidence: float


def classify_intent(state: State):
    q = state["query"].lower()
    intent = (
        "policy_question"
        if any(keyword in q for keyword in POLICY_KEYWORDS)
        else "general_question"
    )
    return {"intent": intent}


def route_after_classification(state: State):
    return state["intent"]


def _mock_policy_answer(query: str, context: list):
    if not context:
        return (
            "I can only answer Zepto policy questions using the available "
            "policy context, and I could not find relevant policy information."
        )

    text = context[0]["text"]
    q = query.lower()

    # Deterministic, grounded baseline: select relevant sentence(s).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    terms = set(re.findall(r"[a-z0-9]+", q))
    scored = []

    for sentence in sentences:
        st = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(terms & st)
        if score:
            scored.append((score, sentence))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        answer = " ".join(s for _, s in scored[:2])
    else:
        answer = sentences[0] if sentences else text

    return f"Based on the retrieved context: {answer}"


def _mock_general_answer():
    return "I can only answer questions about Zepto policies in this support assistant."


def retrieve_and_answer(state: State):
    context = retrieve(state["query"], top_k=3)
    answer = _mock_policy_answer(state["query"], context)

    return {
        "context": context,
        "answer": answer,
        "sources": [item["source"] for item in context],
        "confidence": 0.85 if context else 0.40,
    }


def direct_answer(state: State):
    return {
        "answer": _mock_general_answer(),
        "sources": [],
        "confidence": 0.60,
    }


def build_graph():
    graph = StateGraph(State)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_GRAPH = None


def ask(query: str) -> AssistantResponse:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    result = _GRAPH.invoke({"query": query})

    return AssistantResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=float(result.get("confidence", 0.0)),
    )
