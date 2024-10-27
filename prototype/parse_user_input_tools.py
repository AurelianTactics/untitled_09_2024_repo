'''
TBD

working:



to do:
    test this
        
backlog
    make tools mroe flexible
    have leased be one of the purchase types

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

_PARSE_USER_INPUT_TOOLS_DESCRIPTION = (
    "parse_user_input(problem: str, context: Optional[list[str]]) -> str:\n"
    "You are a helpful assistant. Parse the user input to JSON format per the prompt.\n"
    "Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.\n"
    # add examples?
)

_SYSTEM_PROMPT = """
Given a list of user inputs, extract the vehicle information and output to a JSON format.
If the user has not given that information, leave the information null.
If the user has given multiple vehicles, have an entry per each vehicle.

For each vehicle, get the following fields:
    - purchase_type: new or used
    - year: the year of the vehicle (if applicable)
    - make: the manufacturer of the vehicle
    - model: the model of the vehicle
    - price: the price of the vehicle
    - mileage_minimum: the minimum mileage
    - mileage_maximum: the maximum mileage
    - miles_driven_per_year: user miles driven per year
    

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

def get_parse_user_input_tool(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output()

    def calculate_parse_user_input(
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
        name="parse_user_input",
        func=calculate_parse_user_input,
        description=_PARSE_USER_INPUT_TOOLS_DESCRIPTION,
    )
