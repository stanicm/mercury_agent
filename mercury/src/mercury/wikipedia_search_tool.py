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
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class WikipediaSearchConfig(FunctionBaseConfig, name="wikipedia_search_tool"):
    llm_name: LLMRef


@register_function(config_type=WikipediaSearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def wikipedia_search(tool_config: WikipediaSearchConfig, builder: Builder):
    import os
    from langchain_community.tools import WikipediaQueryRun
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    api_token = os.getenv("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = api_token

    if not api_token:
        raise ValueError(
            "API token must be provided in the configuration or in the environment variable `NVIDIA_API_KEY`")

    llm = await builder.get_llm(llm_name=tool_config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    wikipedia_tool = WikipediaQueryRun()

    async def wiki_search(topic: str) -> str:
        try:
            result = wikipedia_tool.run(topic)
            return result
        except Exception as e:
            logger.exception("Error in Wikipedia search: %s", e, exc_info=True)
            return f"Failed to search Wikipedia for topic: {topic}. Error: {str(e)}"

    prompt_template = """
    You are an expert at extracting search topics from user queries to find relevant information on Wikipedia.
    ------
    {inputs}
    ------
    The output MUST use the following format:
    '''
    topic: a clear and specific topic to search for on Wikipedia
    '''
    Begin!
    [/INST]
    """
    prompt = PromptTemplate(
        input_variables=['inputs'],
        template=prompt_template,
    )

    class TopicExtract(BaseModel):
        topic: str = Field(description="specific topic or keyword to search for on Wikipedia")

    llm_with_output_structure = llm.with_structured_output(TopicExtract)

    async def execute_tool(out):
        try:
            topic = out.topic
            if topic and topic.strip():
                output_summary = await wiki_search(topic)
            else:
                output_summary = f"Could not perform Wikipedia search - invalid topic: {topic}"
        except Exception as e:
            output_summary = f"Failed to search Wikipedia for topic: {topic}. Error: {str(e)}"
            logger.exception("Error in executing tool: %s", e, exc_info=True)

        return output_summary

    research = (prompt | llm_with_output_structure | execute_tool)

    async def _arun(inputs: str) -> str:
        """
        Search Wikipedia for information based on a topic extracted from user input
        Args:
            inputs : user input
        """
        output = await research.ainvoke(inputs)
        logger.info("output from wikipedia_search_tool: %s", output)
        return output

    yield FunctionInfo.from_fn(_arun, description="search Wikipedia for relevant information") 