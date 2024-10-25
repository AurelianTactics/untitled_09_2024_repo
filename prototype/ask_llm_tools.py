'''
Idea with this is some questiosn want to the LLM and want to parallelize so not asking the LLM three questions at the same time
    plus may want to have more specific outputs or more specific inputs
    unsure if should be done htis way or with nodes and LLMs
    or just ask the LLM directly

minimal version
    - used car price
    - new car price
    - does the care qualify for federal rebate
    

working:



to do:

        
backlog
    feed in examples dynimcally
    add RAG context


'''

import re
from typing import List, Optional

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_ASK_LLM_TOOLS_DESCRIPTION = (
    "answer_user_question(problem: str, context: Optional[list[str]]) -> str:\n"
    "You are a helpful assistant. Concisely answer the question.\n"
    "Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.\n"
    "Depending on the question may be give additional context.\n"
    "Depending on the question may be asked to output in a specific way.\n"
    # add examples?
)

_SYSTEM_PROMPT = """
Take the question and concisely answer the question. Output it in a JSON format:

Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.
Depending on the question may be give additional context.
Depending on the question may be asked to output in a specific way.

Examples:
- Question: What is the price of a new Tesla Model Y?
- Answer: {"type": "new", "make": "Tesla", "model": "Model Y"  "price": 42290}

- Question: What is the price of a 2020 Ford F-150 with 50,000 miles?
- Answer: {"type": "used", "make": "Ford", "model": "F-150"  "price": 30,491, "mileage_minimum": 40000, "mileage_maximum": 60000}

"""

_ADDITIONAL_CONTEXT_PROMPT = """"""

def answer_user_question(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output()

    def answer_question(
        problem: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"problem": problem}
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
                chain_input["context"] = [SystemMessage(content=context_str)]
        extractor_response = extractor.invoke(chain_input, config)

        return extractor_response

    return StructuredTool.from_function(
        name="answerquestion",
        func=answer_question,
        description=_ASK_LLM_TOOLS_DESCRIPTION,
    )
