'''

working:
    do the description
    do something similar to the math tools file
        maybe two files
    description
    LLM call
    formatting

    repeat for fuel cost

want tools or LLM calls for
maintenance cost
fuel cost

both can be answered by LLMs, want to make it a "tool" call so it  uses LLM compiler

output should be by year expense

feed in the relevant info
    maintenance
        miles per year if given
        vehicle class if given
        car if given
        starting milage of car if given
        location if given

    fuel cost
        location if given
        gas/hybrid/electric
        price per gallon if given
        kw/h if given

returns
    JSON:
    {
        "maintenance":
            {
                "year_1": 1234
            },
            ...
            {
                "year_x": 1234
            }
    }

    {
        "fuel_cost":
            {
                "year_1": 1234
            },
            ...
            {
                "year_x": 1234
            }
    }
    
backlog
    break out maintenance and fuel to seperate files
    figure out how to use the description part. is that just examples?
    way to feed in info from the state rather than hardcode the defaults/leave up to the LLM?

'''

_MAINTENANCE_DESCRIPTION = (
    "get_annual_maintenance_cost(problem: str, context: Optional[list[str]]) -> json:\n"
    # "math(problem: str, context: Optional[list[str]]) -> float:\n"
    # " - Solves the provided math problem.\n"
    # ' - `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g. "how many apples are there if there are 3 apples and 2 apples").\n'
    # " - You cannot calculate multiple expressions in one call. For instance, `math('1 + 3, 2 + 4')` does not work. "
    # "If you need to calculate multiple expressions, you need to call them separately like `math('1 + 3')` and then `math('2 + 4')`\n"
    # " - Minimize the number of `math` actions as much as possible. For instance, instead of calling "
    # '2. math("what is the 10% of $1") and then call 3. math("$1 + $2"), '
    # 'you MUST call 2. math("what is the 110% of $1") instead, which will reduce the number of math actions.\n'
    # # Context specific rules below
    # " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    # "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    # " - `math` action will not see the output of the previous actions unless you provide it as `context`. "
    # "You MUST provide the output of the previous actions as `context` if you need to do math on it.\n"
    # " - You MUST NEVER provide `search` type action's outputs as a variable in the `problem` argument. "
    # "This is because `search` returns a text blob that contains the information about the entity, not a number or value. "
    # "Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `math` action. "
    # 'For example, 1. search("Barack Obama") and then 2. math("age of $1") is NEVER allowed. '
    # 'Use 2. math("age of Barack Obama", context=["$1"]) instead.\n'
    # " - When you ask a question about `context`, specify the units. "
    # 'For instance, "what is xx in height?" or "what is xx in millions?" instead of "what is xx?"\n'
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