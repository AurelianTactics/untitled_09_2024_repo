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


vehicle_parse_json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "vehicle_list": {
            "type": "array",
            "description": "A list of vehicles, each with its own properties. Must return at least one.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["new", "used"],
                        "description": "Vehicle type (new or used). Default to new if input is unspecified."
                    },
                    "year": {
                        "type": ["integer", "null"],
                        "description": "Vehicle year. If unspecified and vehicle type is new use current year. If unspecified and vehicle type \
                            is used, use the current year minus three years." 
                    },
                    "make": {
                        "type": ["integer", "null"],
                        "description": "Vehicle make. If unspecified, use null."
                    },
                    "model": {
                        "type": ["integer", "null"],
                        "description": "Vehicle model. If unspecified, use null."
                    },
                    "vehicle_class": {
                        "type": ["string", "null"],
                        "description": "Vehicle class. If unspecified, and make and model are null then use Compact SUV."
                    },
                    "price": {
                        "type": ["number", "null"],
                        "description": "Vehicle price. If unspecified, use null."
                    },
                    "mileage_minimum": {
                        "type": ["integer", "null"],
                        "description": "Minimum mileage for used vehicles. null for new vehicle. If unspecified for used, use 30,000 miles."
                    },
                    "mileage_maximum": {
                        "type": ["integer", "null"],
                        "description": "Maximum mileage for used vehicles. null for new vehicle. If unspecified for used, use 50,000 miles."
                    },
                    "miles_driven_per_year": {
                        "type": ["integer", "null"],
                        "description": "Estimated annual mileage. If unspecified use 7,000 miles per year."
                    },
                    "annual_maintenance_cost": {
                        "type": ["number", "null"],
                        "description": "Estimated annual maintenance cost. If unspecified use null."
                    },
                    "annual_insurance_cost": {
                        "type": ["number", "null"],
                        "description": "Estimated annual insurance cost. If unspecified use null."
                    }
                },
                "required": ["type", "vehicle_class",]
            }
            }
        },
        "required": ["vehicle_list"]
    }

_PARSE_USER_INPUT_TOOLS_DESCRIPTION = (
    "parse_user_input(problem: str, context: Optional[list[str]]) -> str:\n"
    "You are a helpful assistant. Parse the user input to JSON format per the prompt.\n"
    "Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.\n"
    # add examples?
)

_SYSTEM_PROMPT = f"""
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
    - annual_maintenance_cost: estimated annual maintenance cost
    - annual_insurance_cost: estimated annual insurance cost
    

JSON Schema: 
    {vehicle_parse_json_schema}

Examples:
- Question: I want to know if the cost of a owning new Tesla Model Y versus a used Ford F-150 with 50,000 miles over the next five years.

- Answer: [
    {"type": "new", 
    "vehicle_class": "mid-size SUV",
    "year": 2024,
    "make": "Tesla", 
    "model": "Model Y"  
    "price": null, 
    "mileage_minimum": null, 
    "mileage_maximum": null, 
    "miles_driven_per_year": null, 
    "annual_maintenance_cost": null, 
    "annual_insurance_cost": null},
    {"type": "used", 
    "vehicle_class": "light-duty truck",
    "year": 2021,
    "make": "Ford", 
    "model": "F-150"  
    "price": null, 
    "mileage_minimum": 40000, 
    "mileage_maximum": 60000,
    "miles_driven_per_year": null, 
    "annual_maintenance_cost": null, 
    "annual_insurance_cost": null}]

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

    
    

    extractor = prompt | llm.with_structured_output(vehicle_parse_json_schema)

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
