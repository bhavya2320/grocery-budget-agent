import os
import time
import gradio as gr

from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. "
        "Please add it in Render → Environment."
    )


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=API_KEY,
    temperature=0.2
)


# ============================================================
# GROCERY PRICE DATABASE
# ============================================================

GROCERY_DATA = """
STAPLES
Rice - ₹60 per kg
Wheat Flour - ₹55 per kg
Rava - ₹70 per kg
Poha - ₹60 per kg

PULSES
Toor Dal - ₹150 per kg
Moong Dal - ₹130 per kg
Chana Dal - ₹110 per kg
Rajma - ₹140 per kg

VEGETABLES
Potato - ₹40 per kg
Onion - ₹40 per kg
Tomato - ₹50 per kg
Carrot - ₹60 per kg
Capsicum - ₹80 per kg
Cabbage - ₹40 per kg
Beans - ₹70 per kg
Spinach - ₹30 per bunch

FRUITS
Banana - ₹60 per dozen
Apple - ₹150 per kg
Orange - ₹100 per kg
Papaya - ₹60 per kg

DAIRY
Milk - ₹65 per litre
Curd - ₹80 per kg
Paneer - ₹350 per kg

PROTEIN
Eggs - ₹7 per egg
Chicken - ₹220 per kg

ESSENTIALS
Cooking Oil - ₹150 per litre
Salt - ₹25 per kg
Sugar - ₹50 per kg
Tea - ₹250 per 500g

These are demonstration prices only.
They are NOT live market prices.
"""


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GroceryState(TypedDict):
    user_request: str
    final_report: str


# ============================================================
# AI AGENT
# ============================================================

def grocery_agent_node(state: GroceryState):

    user_request = state["user_request"]

    prompt = f"""
You are an AI Grocery Budget Agent.

USER REQUEST:
{user_request}

AVAILABLE GROCERY PRICES:
{GROCERY_DATA}

Your job is to create a practical grocery plan based on the
user's budget, number of people and duration.

Analyze:
- budget
- number of people
- number of days
- food preference
- dietary restrictions
- ingredients already available

Create:

1. Grocery shopping list
2. Quantities
3. Estimated cost
4. Total budget
5. Remaining amount
6. Simple meal plan
7. Budget optimization
8. Money-saving tips
9. Final shopping checklist

IMPORTANT:
- Use ONLY the prices provided above.
- Do not invent grocery prices.
- Make reasonable assumptions if information is missing.
- Keep the answer practical and concise.

Return exactly this structure:

# 🛒 Grocery Budget Report

## Household Details

| Item | Details |
|---|---|
| People | ... |
| Budget | ... |
| Duration | ... |
| Food Preference | ... |
| Existing Ingredients | ... |

## Budget Summary

| Category | Cost |
|---|---:|
| Staples | ₹... |
| Pulses | ₹... |
| Vegetables | ₹... |
| Fruits | ₹... |
| Dairy | ₹... |
| Protein | ₹... |
| Essentials | ₹... |
| **TOTAL** | **₹...** |
| **REMAINING** | **₹...** |

## Shopping List

| Item | Quantity | Cost |
|---|---:|---:|
| ... | ... | ₹... |

## Meal Plan

### Day 1
- Breakfast: ...
- Lunch: ...
- Dinner: ...

Continue according to the requested duration.

## Budget Optimization

Explain briefly how the budget was optimized.

## Money-Saving Tips

1. ...
2. ...
3. ...
4. ...
5. ...

## Shopping Checklist

- [ ] ...
- [ ] ...
- [ ] ...

### Price Disclaimer

These are demonstration prices, not live market prices.
"""


    # ========================================================
    # RETRY FOR TEMPORARY GEMINI ERRORS
    # ========================================================

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = llm.invoke(prompt)

            state["final_report"] = response.content

            return state

        except Exception as e:

            error = str(e)

            # Temporary server overload
            if "503" in error or "UNAVAILABLE" in error:

                if attempt < max_attempts - 1:

                    wait_seconds = 2 ** (attempt + 1)

                    time.sleep(wait_seconds)

                    continue

                state["final_report"] = """
# ⚠️ Gemini Temporarily Unavailable

Gemini is currently busy.

The application automatically retried the request.
Please wait a few seconds and try again.
"""

                return state

            # API quota / rate limit
            if "429" in error or "RESOURCE_EXHAUSTED" in error:

                state["final_report"] = """
# ⚠️ Gemini API Limit Reached

The Gemini API quota or rate limit for this API key
has been reached.

Please wait for the quota to reset or use an API
project with available quota.
"""

                return state

            # Any other error
            state["final_report"] = f"""
# ❌ Gemini Error

{error}

Please try again.
"""

            return state

    return state


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(GroceryState)

workflow.add_node(
    "grocery_agent",
    grocery_agent_node
)

workflow.set_entry_point(
    "grocery_agent"
)

workflow.add_edge(
    "grocery_agent",
    END
)

grocery_graph = workflow.compile()


# ============================================================
# MAIN FUNCTION
# ============================================================

def grocery_agent(user_request):

    if not user_request or not user_request.strip():

        return "⚠️ Please enter your grocery requirements."

    result = grocery_graph.invoke({
        "user_request": user_request,
        "final_report": ""
    })

    return result["final_report"]


# ============================================================
# CLEAN UI
# ============================================================

CSS = """
body {
    background: #f6f7fb !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    padding: 30px !important;
    font-family: Arial, sans-serif !important;
}

/* Header */

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 32px;
    color: #222;
    margin-bottom: 8px;
}

.header p {
    color: #666;
    font-size: 15px;
}

/* Input / Output */

.panel {
    background: white;
    border: 1px solid #e2e4e8;
    border-radius: 12px;
    padding: 20px;
}

/* Textbox */

textarea {
    border-radius: 8px !important;
    font-size: 15px !important;
}

/* Generate button */

.generate-btn {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.generate-btn:hover {
    background: #4338ca !important;
}

/* Clear button */

.clear-btn {
    border-radius: 8px !important;
}

/* Output */

.output {
    min-height: 500px;
}

/* Footer */

.footer {
    text-align: center;
    color: #888;
    font-size: 12px;
    margin-top: 25px;
}
"""


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="Grocery Budget Agent",
    css=CSS
) as demo:

    # Header

    gr.HTML(
        """
        <div class="header">
            <h1>🛒 Grocery Budget Agent</h1>
            <p>
                Plan your groceries, meals and budget with AI.
            </p>
        </div>
        """
    )

    # Main area

    with gr.Row():

        # LEFT

        with gr.Column():

            with gr.Group(elem_classes="panel"):

                gr.Markdown("### Grocery Requirements")

                user_input = gr.Textbox(
                    label="",
                    placeholder=(
                        "Example:\n"
                        "I have ₹3000 for groceries for 2 people "
                        "for 7 days. We eat vegetarian Indian food. "
                        "I already have rice and cooking oil."
                    ),
                    lines=8
                )

                with gr.Row():

                    clear_button = gr.Button(
                        "Clear",
                        elem_classes="clear-btn"
                    )

                    generate_button = gr.Button(
                        "Generate Plan",
                        variant="primary",
                        elem_classes="generate-btn"
                    )

            gr.Markdown("### Example")

            gr.Examples(
                examples=[
                    [
                        "I have ₹3000 for groceries for 2 people "
                        "for 7 days. We eat vegetarian Indian food."
                    ],
                    [
                        "I have ₹2000 for groceries for 1 person "
                        "for 7 days. I need simple Indian meals."
                    ],
                    [
                        "I have ₹5000 for groceries for 3 people "
                        "for 10 days. We eat vegetarian food."
                    ]
                ],
                inputs=user_input
            )

        # RIGHT

        with gr.Column():

            with gr.Group(elem_classes="panel"):

                gr.Markdown("### Your Grocery Plan")

                output = gr.Markdown(
                    """
Enter your requirements and click **Generate Plan**.

Your result will contain:

- Budget breakdown
- Shopping list
- Quantities
- Meal plan
- Budget optimization
- Money-saving tips
- Shopping checklist
                    """,
                    elem_classes="output"
                )


    # Footer

    gr.HTML(
        """
        <div class="footer">
            Grocery Budget Agent • LangGraph + Gemini
        </div>
        """
    )


    # ========================================================
    # BUTTON EVENTS
    # ========================================================

    generate_button.click(
        fn=grocery_agent,
        inputs=user_input,
        outputs=output
    )

    user_input.submit(
        fn=grocery_agent,
        inputs=user_input,
        outputs=output
    )

    clear_button.click(
        fn=lambda: ("", """
Enter your requirements and click **Generate Plan**.
"""),
        inputs=None,
        outputs=[user_input, output]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
