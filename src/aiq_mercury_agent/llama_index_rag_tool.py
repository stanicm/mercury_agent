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
import os
from typing import Optional
import json

import httpx
from pydantic import ConfigDict

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class RAGServerConfig(FunctionBaseConfig, name="llama_index_rag"):
    model_config = ConfigDict(protected_namespaces=())

    base_url: str = "http://0.0.0.0:8081/v1"
    collection_name: str = "SPH"
    top_k: int = 3
    timeout: int = 120
    use_knowledge_base: bool = True


@register_function(config_type=RAGServerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def llama_index_rag_tool(tool_config: RAGServerConfig, builder: Builder):
    from colorama import Fore

    async def _arun(query: str) -> str:
        """
        Query the RAG server for relevant information
        Args:
            query: user query
        Returns:
            str: response from the RAG server
        """
        try:
            async with httpx.AsyncClient(timeout=tool_config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{tool_config.base_url}/generate",
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "use_knowledge_base": tool_config.use_knowledge_base,
                        "collection_name": tool_config.collection_name,
                        "reranker_top_k": tool_config.top_k,
                        "vdb_top_k": tool_config.top_k
                    }
                ) as response:
                    response.raise_for_status()
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("choices") and len(data["choices"]) > 0:
                                    content = data["choices"][0]["delta"].get("content", "")
                                    if content:
                                        full_response += content
                            except json.JSONDecodeError:
                                continue
                    logger.info("%s RAG Server Response: %s %s", Fore.MAGENTA, full_response, Fore.RESET)
                    return full_response if full_response else "No response from RAG server"
        except Exception as e:
            logger.error("Error querying RAG server: %s", str(e))
            return f"Error querying RAG server: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Query the RAG server for relevant information")
