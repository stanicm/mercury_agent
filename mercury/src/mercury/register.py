# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import haystack_agent  # noqa: F401, pylint: disable=unused-import
from . import llama_index_rag_tool  # noqa: F401, pylint: disable=unused-import

logger = logging.getLogger(__name__)


class MercuryWorkflowConfig(FunctionBaseConfig, name="mercury"):
    # Add your custom configuration parameters here
    llm: LLMRef = "nim_llm"
    data_dir: str = "/home/coder/dev/ai-query-engine/examples/multi_frameworks/data/"
    research_tool: FunctionRef
    rag_tool: FunctionRef
    chitchat_agent: FunctionRef


@register_function(config_type=MercuryWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mercury_workflow(config: MercuryWorkflowConfig, builder: Builder):
    # Implement your workflow logic here
    from typing import TypedDict

    from colorama import Fore
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langgraph.graph import END
    from langgraph.graph import StateGraph

    # Use builder to generate framework specific tools and llms
    logger.info("workflow config = %s", config)

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    research_tool = builder.get_tool(fn_name=config.research_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    rag_tool = builder.get_tool(fn_name=config.rag_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chitchat_agent = builder.get_tool(fn_name=config.chitchat_agent, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    chat_hist = ChatMessageHistory()

    router_prompt = """
    Given the user input below, classify it as either being about 'Research', 'Retrieve' or 'General' topic.
    Just use one of these words as your response. \
    'Research' - any question that requires searching Wikipedia for information, such as "tell me about quantum computing" or "who is Albert Einstein?"...etc
    'Retrieve' - any question related to the topic of AIQ Toolkit or its workflows, especially concerning the particular workflow called mercury which show case using multiple frameworks such as langchain, llama-index ..etc
    'General' - answering small greeting or chitchat type of questions or everything else that does not fall into any of the above topics.
    User query: {input}
    Classifcation topic:"""

    routing_chain = ({
        "input": RunnablePassthrough()
    }
                     | PromptTemplate.from_template(router_prompt)
                     | llm
                     | StrOutputParser())

    supervisor_chain_with_message_history = RunnableWithMessageHistory(
        routing_chain,
        lambda _: chat_hist,
        history_messages_key="chat_history",
    )

    class AgentState(TypedDict):
        """"
            Will hold the agent state in between messages
        """
        input: str
        chat_history: list[BaseMessage] | None
        chosen_worker_agent: str | None
        final_output: str | None

    async def supervisor(state: AgentState):
        query = state["input"]
        chosen_agent = (await supervisor_chain_with_message_history.ainvoke(
            {"input": query},
            {"configurable": {
                "session_id": "unused"
            }},
        ))
        logger.info("%s========== inside **supervisor node**  current status = \n %s", Fore.BLUE, state)

        return {'input': query, "chosen_worker_agent": chosen_agent, "chat_history": chat_hist}

    async def router(state: AgentState):
        """
        Route the response to the appropriate handler
        """

        status = list(state.keys())
        logger.info("========== inside **router node**  current status = \n %s, %s", Fore.CYAN, status)
        if 'final_output' in status:
            route_to = "end"
        elif 'chosen_worker_agent' not in status:
            logger.info(" ############# router to --> supervisor %s", Fore.RESET)
            route_to = "supevisor"
        elif 'chosen_worker_agent' in status:
            logger.info(" ############# router to --> workers %s", Fore.RESET)
            route_to = "workers"
        else:
            route_to = "end"
        return route_to

    async def workers(state: AgentState):
        query = state["input"]
        worker_choice = state["chosen_worker_agent"]
        logger.info("========== inside **workers node**  current status = \n %s, %s", Fore.YELLOW, state)
        if "retrieve" in worker_choice.lower():
            out = (await rag_tool.ainvoke(query))
            output = out
            logger.info("**using rag_tool via llama_index_rag_agent >>> output:  \n %s, %s", output, Fore.RESET)
        elif "general" in worker_choice.lower():
            output = (await chitchat_agent.ainvoke(query))
            logger.info("**using general chitchat chain >>> output:  \n %s, %s", output, Fore.RESET)
        elif 'research' in worker_choice.lower():
            # Add specific search terms to help focus the search
            search_query = query
            if "quantum computing" in query.lower():
                search_query = "quantum computer quantum computing definition overview"
            inputs = {"question": search_query}
            wiki_results = (await research_tool.ainvoke(inputs))
            
            # Create a prompt for summarizing the Wikipedia results
            summary_prompt = PromptTemplate.from_template(
                """You are a helpful assistant that summarizes information from Wikipedia articles.
                Please provide a clear and concise summary of the following Wikipedia content, focusing on answering the user's question: {question}

                Wikipedia content:
                {content}

                Please provide a well-structured summary that directly answers the user's question. If the Wikipedia content is not relevant to the question, please indicate that we should try another search."""
            )
            
            # Create the summarization chain
            summary_chain = summary_prompt | llm | StrOutputParser()
            
            # Get the summary
            output = await summary_chain.ainvoke({
                "question": query,
                "content": wiki_results
            })
            
            logger.info("**using research tool with summarization >>> output: \n %s, %s", output, Fore.RESET)
        else:
            output = ("Apologies, I am not sure what to say, I can answer general questions, retrieve info about this "
                      "mercury workflow, and search Wikipedia for information, but nothing more.")
            logger.info("**!!! not suppose to happen, try to debug this >>> output:  \n %s, %s", output, Fore.RESET)

        return {'input': query, "chosen_worker_agent": worker_choice, "chat_history": chat_hist, "final_output": output}

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor)
    workflow.set_entry_point("supervisor")
    workflow.add_node("workers", workers)
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "workers": "workers", "end": END
        },
    )
    workflow.add_edge("supervisor", "workers")
    workflow.add_edge("workers", END)
    app = workflow.compile()

    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output

        try:
            logger.debug("Starting agent execution")
            out = (await app.ainvoke({"input": input_message, "chat_history": chat_hist}))
            output = out["final_output"]
            logger.info("final_output : %s ", output)
            return output
        finally:
            logger.debug("Finished agent execution")

    try:
        yield _response_fn
    except GeneratorExit:
        logger.exception("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up mercury workflow.")
