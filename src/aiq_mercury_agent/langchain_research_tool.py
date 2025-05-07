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
import wikipedia

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LangChainResearchConfig(FunctionBaseConfig, name="langchain_researcher_tool"):
    llm_name: LLMRef


@register_function(config_type=LangChainResearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def langchain_research(tool_config: LangChainResearchConfig, builder: Builder):

    import os
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel
    from pydantic import Field

    api_token = os.getenv("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = api_token

    if not api_token:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def wikipedia_search(topic: str) -> str:
        try:
            # Search Wikipedia
            search_results = wikipedia.search(topic, results=3)
            if not search_results:
                return f"No Wikipedia articles found for topic: {topic}"
            
            # Get the summary of the first result
            summary = wikipedia.summary(search_results[0], sentences=3)
            return f"Wikipedia Summary for '{search_results[0]}':\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages
            return f"Multiple matches found for '{topic}'. Please be more specific. Options: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for topic: {topic}"
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    prompt_template = """
    You are an expert at extracting the most relevant search topic from user queries to search on Wikipedia.
    ------
    {inputs}
    ------
    The output MUST use the following format:
    '''
    topic: a clear and specific topic or keyword to search on Wikipedia, focusing on the main subject of interest
    '''
    Begin!
    [/INST]
    """
    prompt = PromptTemplate(
        input_variables=['inputs'],
        template=prompt_template,
    )

    class TopicExtract(BaseModel):
        topic: str = Field(description="most important keyword or topic to search on Wikipedia")

    llm_with_output_structure = llm.with_structured_output(TopicExtract)

    async def execute_tool(out):
        try:
            topic = out.topic
            if topic is not None and topic not in ['', '\n']:
                output_summary = await wikipedia_search(topic)
            else:
                output_summary = f"Could not extract a valid topic for Wikipedia search"

        except Exception as e:
            output_summary = f"Error in Wikipedia search: {str(e)}"
            logger.exception("error in executing tool: %s", e, exc_info=True)

        return output_summary

    research = (prompt | llm_with_output_structure | execute_tool)

    async def _arun(inputs: str) -> str:
        """
        Search Wikipedia using a topic extracted from user input
        Args:
            inputs : user input
        """
        output = await research.ainvoke(inputs)
        logger.info("output from langchain_research_tool: %s", output)
        return output

    yield FunctionInfo.from_fn(_arun, description="extract relevant information from Wikipedia search")
