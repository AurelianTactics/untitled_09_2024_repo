'''

working:


    TEST maintenance part


to do:
    not sure how and what I am feeding into this
    not really sure how parts of this work

        
backlog
    break out maintenance and fuel to seperate files
    figure out how to use the description part. is that just examples?
    way to feed in info from the state rather than hardcode the defaults/leave up to the LLM?

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

_MAINTENANCE_DESCRIPTION = (
    "get_annual_maintenance_cost(problem: str, context: Optional[list[str]]) -> json:\n"
    "Take the user question and output the annual cost of vehicle maintenance. Output it in a JSON format.\n"
    # add examples?
)

_SYSTEM_PROMPT = """
Take the user question and output the annual cost of vehicle maintenance. Output it in a JSON format:

{
    "maintenance":
        {
            "year_1": 1000.00
        },
        {
            "year_2": 1200.00
        },
        {
            "year_3": 1400.00
        },
        {
            "year_4": 1500.00
        },
        {
            "year_5": 1600.00
        }
}

-If number of years is not given assume 5.
-If new or used car designation is not given assume new.
-If new assume the mileage on the car is 0.
-If used and the mileage on the car not give assume the care has 50,000 miles.
-If the number of miles driven per year is not given assume 10,000 miles per year.
-If the location is not given use the overall United States of America average.
-If the vehicle class is not given use the overall United States of America average.
-If the vehicle class is given use that to give a better estimate.
-If the make and model of the care is given, use that to give a better estimate.

Examples:

"""

_ADDITIONAL_CONTEXT_PROMPT = """"""

def get_maintenance_tool(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output()

    def calculate_maintenance(
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
        name="maintenance",
        func=calculate_maintenance,
        description=_MAINTENANCE_DESCRIPTION,
    )
