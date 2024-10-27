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

_ASK_LLM_TOOLS_NEW_CAR_FEDERAL_REBATE_DESCRIPTION = (
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
Answer yes or no if the vehicle will qualify for the federal used car electric vehicle rebate.
Answer with the amount of the rebate if applicable.
Always caveat your answer that the dealer must confirm the vehicle's eligibility.

Context:
If you place in service a new plug-in electric vehicle (EV) or fuel cell vehicle (FCV) in 2023 or after, you may qualify for a clean vehicle tax credit. For more information on how to qualify see Publication 5866, New Clean Vehicle Tax Credit Checklist PDF.

At the time of sale, a seller must give you information about your vehicle's qualifications. Sellers must also register online and report the same information to the IRS. If they don't, your vehicle won't be eligible for the credit. For more information see Publication 5905, Information for Consumers Purchasing a New or Used Clean Vehicle PDF.

Find information on credits for used clean vehicles, qualified commercial clean vehicles and new plug-in EVs purchased before 2023.

Who qualifies
You may qualify for a credit up to $7,500 under Internal Revenue Code Section 30D if you buy a new, qualified plug-in EV or fuel cell electric vehicle (FCV). The Inflation Reduction Act of 2022 changed the rules for this credit for vehicles purchased from 2023 to 2032.

The credit is available to individuals and their businesses.

To qualify, you must:

Buy it for your own use, not for resale
Use it primarily in the U.S.
In addition, your modified adjusted gross income (AGI) may not exceed:

$300,000 for married couples filing jointly or a surviving spouse
$225,000 for heads of households
$150,000 for all other filers
You can use your modified AGI from the year you take delivery of the vehicle or the year before, whichever is less. If your modified AGI is below the threshold in 1 of the 2 years, you can claim the credit.

If you do not transfer the credit, it is nonrefundable when you file your taxes, so you can't get back more on the credit than you owe in taxes. You can't apply any excess credit to future tax years.

Credit amount
The amount of the credit depends on when you placed the vehicle in-service (took delivery), regardless of purchase date.

For vehicles placed in-service January 1 to April 17, 2023:
$2,500 base amount
Plus $417 for a vehicle with at least 7 kilowatt hours of battery capacity
Plus $417 for each kilowatt hour of battery capacity beyond 5 kilowatt hours
Up to $7,500 total
In general, the minimum credit will be $3,751 ($2,500 + 3 times $417), the credit amount for a vehicle with the minimum 7 kilowatt hours of battery capacity.

For vehicles placed in-service April 18, 2023 and after:
Vehicles will have to meet all of the same criteria listed above, plus meet new critical mineral and battery component requirements for a credit up to:

$3,750 if the vehicle meets the critical minerals requirement only
$3,750 if the vehicle meets the battery components requirement only
$7,500 if the vehicle meets both
A vehicle that doesn't meet either requirement will not be eligible for a credit.

Qualified vehicles
Click the button below to see if a vehicle is eligible for the new clean vehicle credit.

To qualify, a vehicle must:

Have a battery capacity of at least 7 kilowatt  hours
Have a gross vehicle weight rating of less than 14,000 pounds
Be made by a qualified manufacturer  
Undergo final assembly in North America
Meet critical mineral and battery component requirements (as of April 18, 2023)
The sale qualifies only if:

You buy the vehicle new.
The seller reports required information to you at the time of sale and to the IRS.
Sellers are required to report your name and taxpayer identification number to the IRS for you to be eligible to claim the credit.
In addition, the vehicle's manufacturer suggested retail price (MSRP) can't exceed:

$80,000 for vans, sport utility vehicles and pickup trucks
$55,000 for other vehicles
MSRP is the retail price of the automobile suggested by the manufacturer, including manufacturer installed options, accessories and trim but excluding destination fees. It isn't necessarily the price you pay.

You can find your vehicle's weight, battery capacity, final assembly location (listed as "final assembly point") and VIN on the vehicle's window sticker.

How to claim the credit
To claim the credit, file Form 8936, Clean Vehicle Credits with your tax return. You will need to provide your vehicle's VIN.

Get a time-of-sale report
The dealer should give you a paper copy of a time-of-sale report when you complete your purchase.

Keep this copy for your records because it affirms that the dealer sent a report to the IRS on the purchase date.
If you didn’t receive a copy of the report, follow our step-by-step guide.
File Form 8936 with your tax return
You must file Form 8936 when you file your tax return for the year in which you take delivery of the vehicle. This is true whether you transferred the credit at the time of sale or you’re waiting to claim the credit when you file.

If you have questions or concerns, follow our step-by-step guide.

Make	Model	Model Year	Vehicle Type	Credit Amount	MSRP Limit
Acura					
	ZDX	2024	EV	$7,500	$80,000
Audi					
	Q5 PHEV 55 TFSI e quattro	2023–2024	PHEV	$3,750	$80,000
	Q5 S Line 55 TFSI e quattro	2023–2024	PHEV	$3,750	$80,000
Cadillac					
	LYRIQ	2024	EV	$7,500	$80,000
Chevrolet					
	Blazer EV	2024	EV	$7,500	$80,000
	Bolt EUV	2022–2023	EV	$7,500	$55,000
	Bolt EV	2022–2023	EV	$7,500	$55,000
	Equinox EV	2024	EV	$7,500	$80,000
Chrysler					
	Pacifica PHEV	2022–2024	PHEV	$7,500	$80,000
Ford					
	Escape Plug-in Hybrid	2022–2025	PHEV	$3,750	$80,000
	F-150 Lightning (Extended Range Battery)	2022–2025	EV	$7,500	$80,000
	F-150 Lightning (Standard Range Battery)	2022–2025	EV	$7,500	$80,000
Honda					
	Prologue	2024	EV	$7,500	$80,000
Jeep					
	Grand Cherokee PHEV 4xe	2022–2024	PHEV	$3,750	$80,000
	Wrangler PHEV 4xe	2022–2024	PHEV	$3,750	$80,000
Lincoln					
	Corsair Grand Touring	2022–2025	PHEV	$3,750	$80,000
Nissan					
	LEAF S	2024	EV	$3,750	$55,000
	LEAF SV PLUS	2024	EV	$3,750	$55,000
Rivian					
	R1S Dual Large	2023–2024	EV	$3,750	$80,000
	R1S Dual Standard	2024	EV	$3,750	$80,000
	R1S Dual Standard+	2024	EV	$3,750	$80,000
	R1S Performance Dual Standard+	2024	EV	$3,750	$80,000
	R1S Quad Large	2022–2024	EV	$3,750	$80,000
	R1T Dual Large	2023–2025	EV	$3,750	$80,000
	R1T Dual Max	2023–2024	EV	$3,750	$80,000
	R1T Dual Performance Large	2023	EV	$3,750	$80,000
	R1T Dual Standard	2024	EV	$3,750	$80,000
	R1T Dual Standard+	2024	EV	$3,750	$80,000
	R1T Performance Dual Standard+	2024	EV	$3,750	$80,000
	R1T Quad Large	2022–2024	EV	$3,750	$80,000
Tesla					
	Model 3 Long Range All-Wheel Drive	2024	EV	$7,500	$55,000
	Model 3 Long Range All-Wheel Drive	2025	EV		$55,000
	Model 3 Long Range Rear-Wheel Drive	2024–2025	EV	$7,500	$55,000
	Model 3 Performance	2023–2025	EV	$7,500	$55,000
	Model X All-Wheel Drive	2023–2024	EV	$7,500	$80,000
	Model Y All-Wheel Drive	2023–2024	EV	$7,500	$80,000
	Model Y Long Range All-Wheel Drive	2025	EV	$7,500	$80,000
	Model Y Long Range Rear-Wheel Drive	2024–2025	EV	$7,500	$80,000
	Model Y Performance	2023–2025	EV	$7,500	$80,000
	Model Y Rear-Wheel Drive	2024	EV	$7,500	$80,000
Volkswagen					
	ID.4 AWD PRO	2023–2024	EV	$7,500	$80,000
	ID.4 AWD PRO S	2023–2024	EV	$7,500	$80,000
	ID.4 AWD PRO S PLUS	2023–2024	EV	$7,500	$80,000
	ID.4 PRO	2023–2024	EV	$7,500	$80,000
	ID.4 PRO S	2023–2024	EV	$7,500	$80,000
	ID.4 PRO S PLUS	2023–2024	EV	$7,500	$80,000
	ID.4 S	2023–2024	EV	$7,500	$80,000
	ID.4 STANDARD	2023–2024	EV	$7,500	$80,000


Examples:
- Question: Does a 2024 Chevrolet Equinox qualify for the federal used car electric vehicle rebate?
- Answer: {"qualifies": "yes", "rebate_amount": 7500, "caveat": "The dealer must confirm the vehicle's eligibility.", "type": "new", "make": "Chevrolet", "model": "Equinox"}

"""

_ADDITIONAL_CONTEXT_PROMPT = """"""

def get_answer_user_question_new_car_federal_rebate(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output()

    def answer_question_new_car_federal_rebate(
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
        name="answerquestion_new_car_federal_rebate",
        func=answer_question_new_car_federal_rebate,
        description=_ASK_LLM_TOOLS_NEW_CAR_FEDERAL_REBATE_DESCRIPTION,
    )
