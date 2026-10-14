"""
Multi-agent research workflow built with LangGraph.

Agents (each is a graph node with its own prompt and job):
    planner    -> splits the question into focused subquestions
    researcher -> runs a search tool loop per subquestion, records sources
    writer     -> drafts a cited answer from the researcher's notes
    critic     -> reviews the draft, then hands off to one of:
                    finalize   (approved, or revision budget spent)
                    writer     (fix wording / unsupported claims)
                    researcher (evidence is missing; search more)

Handoffs use LangGraph's `Command(goto=..., update=...)`, so each agent
decides who runs next and what state to pass along.

Setup:
    pip install langgraph langchain-anthropic pydantic requests
    export ANTHROPIC_API_KEY=...
    export TAVILY_API_KEY=...      # or swap search_web() for any search API

Run:
    python multi_agent_workflow.py "Why did the Bronze Age collapse happen?"
"""
import os
import re
import sys
from typing import Literal, TypedDict

import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

MODEL = "claude-sonnet-5-5"
MAX_REVISIONS = 2       # critic -> writer/researcher round trips
MAX_TOOL_TURNS = 4      # search calls per subquestion

llm = ChatAnthropic(model=MODEL, max_tokens=2000)


# --------------------------------------------------------------------------
# Shared state: this is what agents hand to each other
# --------------------------------------------------------------------------
class Review(BaseModel):
    approved: bool = Field(description="True only if every claim is cited and supported.")
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix.")
    missing_info_queries: list[str] = Field(
        default_factory=list,
        description="Search queries needed if evidence is missing. Empty if none.",
    )


class Plan(BaseModel):
    subquestions: list[str] = Field(description="2-4 focused, searchable subquestions.")


class State(TypedDict, total=False):
    question: str
    subquestions: list[str]
    extra_queries: list[str]
    notes: list[str]
    sources: dict[int, dict]     # id -> {"title", "url"}
    draft: str
    review: Review
    revisions: int
    final: str


# --------------------------------------------------------------------------
# Search tool + source registry
# --------------------------------------------------------------------------
class WebSearch(BaseModel):
    """Search the web. Returns numbered results with title, url and snippet."""
    query: str = Field(description="Short, specific search query.")


def search_web(query: str, n: int = 5) -> list[dict]:
    """Swap this function to use a different search provider."""
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": n},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in resp.json().get("results", [])
    ]


def register(sources: dict[int, dict], title: str, url: str) -> int:
    for sid, s in sources.items():
        if s["url"] == url:
            return sid
    sid = len(sources) + 1
    sources[sid] = {"title": title, "url": url}
    return sid


def run_search(query: str, sources: dict[int, dict]) -> str:
    try:
        results = search_web(query)
    except Exception as e:  # let the agent see failures and adapt
        return f"Search failed: {e}"
    if not results:
        return "No results. Try a different query."
    return "\n".join(
        f"[{register(sources, r['title'], r['url'])}] {r['title']}\n"
        f"URL: {r['url']}\n{r['content']}\n"
        for r in results
    )


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
def planner(state: State) -> Command[Literal["researcher"]]:
    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage("You plan research. Break the question into 2-4 focused "
                      "subquestions that together cover it. Each must be searchable."),
        HumanMessage(state["question"]),
    ])
    return Command(
        goto="researcher",
        update={"subquestions": plan.subquestions, "notes": [], "sources": {}, "revisions": 0},
    )


RESEARCHER_PROMPT = """You are a researcher. Answer the given query using ONLY
what you find via web_search (use it more than once if results are thin).
Write 3-6 sentences of findings. Cite each claim inline with the source ids
shown in the results, like [2]. Never invent ids. If the results don't answer
the query, say so plainly. Paraphrase; don't copy passages."""


def researcher(state: State) -> Command[Literal["writer"]]:
    sources = dict(state.get("sources", {}))
    notes = list(state.get("notes", []))
    queries = state.get("extra_queries") or state["subquestions"]
    agent = llm.bind_tools([WebSearch])

    for q in queries:
        messages = [SystemMessage(RESEARCHER_PROMPT), HumanMessage(q)]
        for _ in range(MAX_TOOL_TURNS):
            ai = agent.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                break
            for call in ai.tool_calls:
                out = run_search(call["args"]["query"], sources)
                messages.append(ToolMessage(content=out, tool_call_id=call["id"]))
        else:  # tool budget spent: ask for findings with no more searching
            messages.append(HumanMessage("Write your findings now from what you have."))
            ai = llm.invoke(messages)
        notes.append(f"Query: {q}\n{ai.text()}")

    return Command(
        goto="writer",
        update={"notes": notes, "sources": sources, "extra_queries": []},
    )


WRITER_PROMPT = """You are a writer. Answer the question using ONLY the research
notes. Cite inline with the ids from the notes, like [1][3]. Never cite an id
that is not in the notes. Where sources conflict or evidence is missing, say so.
Be concise and well organized. Do not add a sources list."""


def writer(state: State) -> Command[Literal["critic"]]:
    parts = [f"Question: {state['question']}", "Research notes:\n" + "\n\n".join(state["notes"])]
    if state.get("draft") and state.get("review"):
        parts.append(f"Previous draft:\n{state['draft']}")
        parts.append("Critic's issues to fix:\n- " + "\n- ".join(state["review"].issues))
    draft = llm.invoke([SystemMessage(WRITER_PROMPT), HumanMessage("\n\n".join(parts))])
    return Command(goto="critic", update={"draft": draft.text()})


CRITIC_PROMPT = """You are a strict reviewer. Check the draft against the notes:
- Is every factual claim cited, and does the cited note actually support it?
- Are there claims with no support in the notes?
- Does it fully answer the question?
Approve only if the draft is fully supported. If wording or citation fixes are
enough, list them in `issues`. If new evidence is needed, also give search
queries in `missing_info_queries`."""


def critic(state: State) -> Command[Literal["writer", "researcher", "finalize"]]:
    review = llm.with_structured_output(Review).invoke([
        SystemMessage(CRITIC_PROMPT),
        HumanMessage(
            f"Question: {state['question']}\n\nNotes:\n" + "\n\n".join(state["notes"])
            + f"\n\nDraft:\n{state['draft']}"
        ),
    ])
    revisions = state.get("revisions", 0)
    update = {"review": review, "revisions": revisions + 1}

    if review.approved or revisions >= MAX_REVISIONS:
        return Command(goto="finalize", update=update)
    if review.missing_info_queries:  # hand back to the researcher
        return Command(goto="researcher",
                       update={**update, "extra_queries": review.missing_info_queries})
    return Command(goto="writer", update=update)  # hand back to the writer


def finalize(state: State) -> dict:
    """Plain function node: validate citations and append the sources list."""
    text = state["draft"]
    sources = state["sources"]
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    unknown = sorted(cited - sources.keys())
    valid = sorted(cited & sources.keys())

    if unknown:
        text += f"\n\n(Warning: unknown source ids {unknown} were cited; treat those claims as unverified.)"
    review = state.get("review")
    if review and not review.approved:
        text += "\n\n(Note: revision limit reached; unresolved reviewer issues: " \
                + "; ".join(review.issues) + ")"
    if valid:
        text += "\n\nSources:\n" + "\n".join(
            f"[{i}] {sources[i]['title']} - {sources[i]['url']}" for i in valid
        )
    else:
        text += "\n\n(No sources were cited.)"
    return {"final": text}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("finalize", finalize)
    g.add_edge(START, "planner")   # everything after the planner is routed by Command
    g.add_edge("finalize", END)
    return g.compile()


def run(question: str, verbose: bool = True) -> str:
    graph = build_graph()
    final = ""
    for step in graph.stream({"question": question}, stream_mode="updates",
                             config={"recursion_limit": 30}):
        for node, update in step.items():
            if verbose:
                print(f"-> {node}", file=sys.stderr)
            if node == "finalize":
                final = update["final"]
    return final


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Research question: ")
    print(run(q))
