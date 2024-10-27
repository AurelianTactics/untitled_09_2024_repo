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

_ASK_LLM_USED_CAR_FEDERAL_REBATE_TOOLS_DESCRIPTION = (
    "get_answer_user_question_used_federal_rebate(problem: str, context: Optional[list[str]]) -> str:\n"
    "You are a helpful assistant. Concisely answer the question.\n"
    "Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.\n"
    "Depending on the question may be give additional context.\n"
    "Depending on the question may be asked to output in a specific way.\n"
    "Answer yes or no if the vehicle will qualify for the federal used car electric vehicle rebate.\n"
)

_SYSTEM_PROMPT = """
Take the question and concisely answer the question. Output it in a JSON format:

Your answer will not go to the user. Your answer will be used as one part of a larger question and workflow.
Depending on the question may be give additional context.
Depending on the question may be asked to output in a specific way.
Answer yes or no if the vehicle will qualify for the federal used car electric vehicle rebate.
Always caveat your answer that the dealer must confirm the vehicle's eligibility.

Context:

Beginning January 1, 2023, if you buy a qualified used electric vehicle (EV) or fuel cell vehicle (FCV) from a licensed dealer for $25,000 or less, you may be eligible for a used clean vehicle tax credit. The credit equals 30% of the sale price up to a maximum credit of $4,000.

If you do not transfer the credit, it is nonrefundable when you file your taxes, so you can't get back more on the credit than you owe in taxes. You can't apply any excess credit to future tax years.

At the time of sale, a seller must give you information about your vehicle's qualifications. Sellers must also register online and report the same information to the IRS. If they don't, your vehicle won't be eligible for the credit.

Purchases made before 2023 don't qualify.

Who qualifies
You may qualify for a credit for buying a previously owned, qualified plug-in electric vehicle (EV) or fuel cell vehicle (FCV), including cars and light trucks, under Internal Revenue Code Section 25E.

To qualify, you must:

Be an individual who bought the vehicle for use and not for resale
Not be the original owner
Not be claimed as a dependent on another person's tax return
Not have claimed another used clean vehicle credit in the 3 years before the purchase date
In addition, your modified adjusted gross income (AGI) may not exceed:

$150,000 for married filing jointly or a surviving spouse
$112,500 for heads of households
$75,000 for all other filers
You can use your modified AGI from the year you take delivery of the vehicle or the year before, whichever is less. If your income is below the threshold for 1 of the 2 years, you can claim the credit.

For more information on how to qualify see Publication 5866-A, Used Clean Vehicle Tax Credit Checklist PDF.

Qualified vehicles and sales
To see if a vehicle is eligible for the used clean vehicle credit:


To qualify, a vehicle must meet all of these requirements:

Have a sale price of $25,000 or less. Sale price includes all dealer-imposed costs or fees not required by law. It doesn't include costs or fees required by law, such as taxes or title and registration fees.
Have a model year at least 2 years earlier than the calendar year when you buy it. For example, a vehicle purchased in 2023 would need a model year of 2021 or older.
Not have already been transferred after August 16, 2022 to a qualified buyer.
Have a gross vehicle weight rating of less than 14,000 pounds
Be an eligible FCV or plug-in EV with a battery capacity of least 7 kilowatt hours
Be for use primarily in the United States
The sale qualifies only if:

You buy the vehicle from a dealer.
For qualified used EVs, the dealer reports required information to you at the time of sale and to the IRS.
A dealer is a person licensed to sell motor vehicles in a state, the District of Columbia, the Commonwealth of Puerto Rico, any other territory or possession of the United States, an Indian tribal government or any Alaska Native Corporation.

Required information includes:

Dealer's name and taxpayer ID number
Buyer's name and taxpayer ID number
Sale date and sale price
Maximum credit allowable under IRC 25E
Vehicle identification number (VIN), unless the vehicle is not assigned one
Battery capacity
How to claim the credit
You can apply the Clean Vehicle Tax Credit immediately toward the amount you pay for the vehicle by transferring the credit to the dealer or you can wait and claim the credit when you file your tax return.

To transfer the credit at the time of sale, you must buy the vehicle from a registered dealer primarily for personal use (not for resale).

Get a time-of-sale report
The dealer should give you a paper copy of a time-of-sale report when you complete your purchase.

Keep this copy for your records because it affirms that the dealer sent a report to the IRS on the purchase date.
If you didn't receive a copy of the report, follow our step-by-step guide.
File Form 8936 with your tax return
You must file Form 8936 when you file your tax return for the year in which you take delivery of the vehicle. This is true whether you transferred the credit at the time of sale or you're waiting to claim the credit when you file.

For detailed instructions, follow our step-by-step guide.


Examples:
- Question: Does a 2020 Kia Nero qualify for the federal used car electric vehicle rebate?
- Answer: {"qualifies": "yes", "rebate_amount": 4000, "caveat": "The dealer must confirm the vehicle's eligibility", "type": "used", "make": "Kia", "model": "Niro EV"}

"""

_ADDITIONAL_CONTEXT_PROMPT = """"""

def get_answer_user_question_used_federal_rebate(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output()

    def answer_question_used_federal_rebate(
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
        name="answerquestion_used_federal_rebate",
        func=answer_question_used_federal_rebate,
        description=_ASK_LLM_USED_CAR_FEDERAL_REBATE_TOOLS_DESCRIPTION ,
    )
