# from typing import Annotated, TypedDict, Sequence
# from langchain_ollama import ChatOllama
# from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from web_search import ddg_search

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# def build_agent(hybrid_retriever, source_description: str):

#     llm = ChatOllama(model="ministral-3:3b", temperature=0.3,streaming=False,stream=False)

#     @tool
#     def retriever_tool(query: str) -> str:
#         """
#         Retrieve information from the local knowledge base
#         and return chunks with inline citations.
#         """
#         docs = hybrid_retriever.retrieve(query)
#         if not docs:
#             return "No relevant local information found."

#         out = []
#         for i, d in enumerate(docs, 1):
#             src = d.metadata.get("source", d.metadata.get("page", "local"))
#             out.append(f"[{i}] ({src})\n{d.page_content}")
#         return str("\n\n".join(out))


#     @tool
#     def web_search_tool(query: str) -> str:
#         """
#         Use DuckDuckGo web search when local knowledge is insufficient.
#         """
#         return ddg_search(query)

#     tools = [retriever_tool, web_search_tool]
#     llm = llm.bind_tools(tools)

#     def should_continue(state: AgentState):
#         last = state["messages"][-1]
#         return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

#     system_prompt = f"""
# You are an Agentic RAG assistant.

# Knowledge source:
# {source_description}

# Rules:
# - Prefer local retrieval
# - Use web search only if needed
# - Cite sources inline as [1], [2], etc.
# """

#     def call_llm(state: AgentState):
#         msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
#         return {"messages": [llm.invoke(msgs)]}

#     tools_map = {t.name: t for t in tools}

#     def take_action(state: AgentState):
#         calls = state["messages"][-1].tool_calls
#         results = []
#         for c in calls:
#             result = tools_map[c["name"]].invoke(c["args"].get("query", ""))
#             results.append(
#                 ToolMessage(
#                     tool_call_id=c["id"],
#                     name=c["name"],
#                     content=str(result),
#                 )
#             )
#         return {"messages": results}

#     graph = StateGraph(AgentState)
#     graph.add_node("llm", call_llm)
#     graph.add_node("tools", take_action)
#     graph.add_conditional_edges("llm", should_continue, {True: "tools", False: END})
#     graph.add_edge("tools", "llm")
#     graph.set_entry_point("llm")

#     return graph.compile()

from typing import Annotated, TypedDict, Sequence
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from web_search import ddg_search


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    answer: str
    retries: int


def build_agent(hybrid_retriever, source_description: str):

    llm = ChatOllama(
        model="ministral-3:3b",
        temperature=0.3,
        streaming=False,
        stream=False,
    )

    evaluator = ChatOllama(
        model="ministral-3:3b",
        temperature=0.0,
        streaming=False,
        stream=False,
    )

    @tool
    def retriever_tool(query: str) -> str:
        """
        Retrieve grounded information from the local knowledge base.
        """
        docs = hybrid_retriever.retrieve(query)
        if not docs:
            return "NO_LOCAL_INFO"

        out = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "local")
            text = d.page_content.replace("*", "-")
            out.append(f"[{i}] ({src})\n{text}")

        return "\n\n".join(out)[:4000]

    @tool
    def web_search_tool(query: str) -> str:
        """
        Search the web when local knowledge is insufficient.
        """
        return ddg_search(query)[:4000]

    tools = [retriever_tool, web_search_tool]
    llm = llm.bind_tools(tools, streaming=False)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    system_prompt = f"""
You are an Agentic RAG assistant.

Knowledge Source:
{source_description}

Rules:
- Prefer local retrieval
- Use web search only if needed
- Cite sources as [1], [2], etc.
"""

    def call_llm(state: AgentState):
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        reply = llm.invoke(msgs)
        return {"messages": [reply], "answer": reply.content}

    def take_action(state: AgentState):
        calls = state["messages"][-1].tool_calls
        results = []

        for c in calls:
            result = tools[[t.name for t in tools].index(c["name"])].invoke(
                c["args"].get("query", "")
            )
            results.append(
                ToolMessage(
                    tool_call_id=c["id"],
                    name=c["name"],
                    content=str(result),
                )
            )

        return {"messages": results}

    def evaluate(state: AgentState):
        prompt = f"""
Score the following answer from 0 to 1.
Only return a number.

Question:
{state['question']}

Answer:
{state['answer']}
"""
        score = evaluator.invoke(prompt).content.strip()
        try:
            score = float(score)
        except:
            score = 0.0

        if score >= 0.7 or state["retries"] >= 2:
            return END

        reflection = f"""
The previous answer was insufficient.

Improve it by:
- Fixing mistakes
- Adding missing information
- Using sources better
"""
        improved = llm.invoke(reflection).content
        return {
            "answer": improved,
            "retries": state["retries"] + 1,
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)
    graph.add_node("evaluate", evaluate)

    graph.add_conditional_edges("llm", should_continue, {True: "tools", False: "evaluate"})
    graph.add_edge("tools", "llm")

    graph.set_entry_point("llm")
    return graph.compile()

