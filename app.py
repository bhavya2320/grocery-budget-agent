import os
import time
import gradio as gr

from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# ============================================================
# 🛒 GROCERY BUDGET AGENT
# ============================================================

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not configured. "
        "Please add your Gemini API key in Render Environment Variables."
    )


# ============================================================
# GEMINI 3.7 FLASH
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=API_KEY,
    temperature=0.3
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
# GROCERY AGENT NODE
# ============================================================

def grocery_agent_node(state):

    user_request = state["user_request"]

    prompt = f"""
You are an intelligent AI Grocery Budget Agent.

USER REQUEST:
{user_request}

GROCERY PRICE DATABASE:
{GROCERY_DATA}

Your task is to create a practical grocery budget plan.

Analyze:

1. Number of people
2. Budget
3. Number of days
4. Food preference
5. Dietary restrictions
6. Meals per day
7. Existing ingredients
8. Other constraints

Then:

- Create a grocery list.
- Estimate practical quantities.
- Calculate costs.
- Compare total cost with the user's budget.
- Optimize the grocery plan.
- Create a meal plan.
- Create a shopping checklist.
- Give money-saving tips.

IMPORTANT:

Use ONLY the prices provided in the grocery database.

Do NOT invent prices.

If some information is missing, make a reasonable assumption
and clearly mention the assumption.

Keep the answer practical and easy to understand.

Return the following format:

# 🛒 Grocery Budget Report

## 👤 Household Requirements

| Requirement | Details |
|---|---|
| People | ... |
| Budget | ... |
| Duration | ... |
| Food Preference | ... |
| Dietary Preference | ... |
| Existing Ingredients | ... |

## 💰 Budget Summary

| Category | Estimated Cost |
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

## 🛍️ Grocery Shopping List

| Item | Quantity | Estimated Cost |
|---|---:|---:|
| ... | ... | ₹... |

## 🍽️ Meal Plan

### Day 1
**Breakfast:** ...
**Lunch:** ...
**Dinner:** ...

Continue for all requested days.

## 💡 Budget Optimization

Explain how the budget was optimized.

## 💰 Money-Saving Tips

1. ...
2. ...
3. ...
4. ...
5. ...

## ✅ Final Shopping Checklist

- [ ] Item
- [ ] Item
- [ ] Item

## ⚠️ Price Disclaimer

Prices shown are demonstration prices and are NOT live
market prices. Verify actual prices before purchasing.
"""


    # ========================================================
    # GEMINI REQUEST + 503 RETRY
    # ========================================================

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = llm.invoke(prompt)

            state["final_report"] = response.content

            return state

        except Exception as e:

            error_message = str(e)

            # ------------------------------------------------
            # TEMPORARY 503 HIGH DEMAND ERROR
            # ------------------------------------------------

            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < max_retries - 1:

                    wait_time = 3 * (2 ** attempt)

                    time.sleep(wait_time)

                    continue

                state["final_report"] = """
# ⚠️ Gemini Temporarily Busy

Gemini is currently experiencing high demand.

The Grocery Budget Agent automatically retried the request
multiple times, but the service is still unavailable.

Please wait a little and click **Generate Plan** again.
"""

                return state

            # ------------------------------------------------
            # QUOTA / RATE LIMIT ERROR
            # ------------------------------------------------

            elif "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

                state["final_report"] = """
# ⚠️ Gemini Rate Limit Reached

The Gemini API rate limit has been reached for this API key.

Please wait for the quota to reset or use an API project
with available quota.
"""

                return state

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            else:

                state["final_report"] = f"""
# ❌ Gemini Error

Something went wrong while generating the grocery plan.

**Error:**
{error_message}

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
# MAIN GRADIO FUNCTION
# ============================================================

def grocery_agent(user_request):

    if not user_request or not user_request.strip():

        return """
# 👋 Welcome!

Please enter your grocery requirements to generate
your personalized budget plan.
"""

    result = grocery_graph.invoke({
        "user_request": user_request,
        "final_report": ""
    })

    return result["final_report"]


# ============================================================
# PROFESSIONAL UI
# ============================================================

CSS = """

/* ================================
   GLOBAL
================================ */

body {
    background: #f5f7fb !important;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding: 25px 35px 40px !important;
    font-family: Inter, Arial, sans-serif !important;
}


/* ================================
   HEADER
================================ */

.app-header {
    background: white;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

.app-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    margin-bottom: 8px !important;
}

.app-subtitle {
    font-size: 15px !important;
    color: #6b7280 !important;
}


/* ================================
   CARDS
================================ */

.card {
    background: white;
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 18px rgba(0,0,0,0.04);
}


/* ================================
   INPUT
================================ */

.input-title {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #111827 !important;
}

textarea {
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
    background: #ffffff !important;
    color: #111827 !important;
    font-size: 15px !important;
}

textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
}


/* ================================
   BUTTONS
================================ */

.primary-btn {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    height: 48px !important;
}

.primary-btn:hover {
    background: #4338ca !important;
}

.secondary-btn {
    border-radius: 11px !important;
    font-weight: 600 !important;
}


/* ================================
   OUTPUT
================================ */

.output-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 18px rgba(0,0,0,0.04);
}

.output-card h1 {
    color: #111827 !important;
}

.output-card h2 {
    color: #374151 !important;
}

.output-card h3 {
    color: #4f46e5 !important;
}


/* ================================
   EXAMPLES
================================ */

.examples-title {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    margin-top: 12px !important;
}


/* ================================
   FOOTER
================================ */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    padding-top: 20px;
}

"""


# ============================================================
# GRADIO BLOCKS UI
# ============================================================

with gr.Blocks(
    title="Grocery Budget Agent",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate"
    )
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="app-header">
            <div class="app-title">
                🛒 Grocery Budget Agent
            </div>

            <div class="app-subtitle">
                AI-powered grocery planning assistant using
                LangGraph + Google Gemini.
            </div>
        </div>
        """
    )


    # --------------------------------------------------------
    # MAIN SECTION
    # --------------------------------------------------------

    with gr.Row(equal_height=True):

        # LEFT SIDE
        with gr.Column(scale=1):

            with gr.Group(elem_classes="card"):

                gr.Markdown(
                    "### 📝 Your Grocery Requirements",
                    elem_classes="input-title"
                )

                user_input = gr.Textbox(
                    show_label=False,
                    placeholder=(
                        "Example:\n\n"
                        "I have ₹3000 for groceries for 2 people "
                        "for 7 days. We eat vegetarian Indian food. "
                        "I already have rice and cooking oil."
                    ),
                    lines=9
                )

                with gr.Row():

                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_classes="secondary-btn"
                    )

                    submit_btn = gr.Button(
                        "✨ Generate Plan",
                        variant="primary",
                        elem_classes="primary-btn"
                    )


            # ------------------------------------------------
            # QUICK EXAMPLES
            # ------------------------------------------------

            gr.Markdown(
                "### 💡 Try an example",
                elem_classes="examples-title"
            )

            gr.Examples(
                examples=[
                    [
                        "I have ₹3000 for groceries for 2 people "
                        "for 7 days. We eat vegetarian Indian food. "
                        "I already have rice and cooking oil."
                    ],
                    [
                        "I have ₹5000 for groceries for 3 people "
                        "for 10 days. We eat vegetarian Indian food."
                    ],
                    [
                        "I have ₹2000 for groceries for 1 person "
                        "for 7 days. I need simple Indian meals."
                    ],
                    [
                        "I have ₹4000 for groceries for 2 people "
                        "for 7 days. We eat high-protein food."
                    ]
                ],
                inputs=user_input,
                label=""
            )


        # RIGHT SIDE
        with gr.Column(scale=1):

            with gr.Group(elem_classes="output-card"):

                gr.Markdown(
                    "### 📊 Your Personalized Grocery Plan"
                )

                output = gr.Markdown(
                    value="""
### 👋 Welcome!

Enter your grocery requirements on the left and click
**Generate Plan**.

Your AI-generated plan will include:

- 💰 Budget breakdown
- 🛍️ Grocery shopping list
- 🍽️ Meal plan
- 📦 Recommended quantities
- 💡 Budget optimization
- 💰 Money-saving tips
- ✅ Shopping checklist
"""
                )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">
            Grocery Budget Agent • LangGraph + Gemini 3.7 Flash
            <br>
            Prices shown are demonstration prices only.
        </div>
        """
    )


    # ========================================================
    # BUTTON ACTIONS
    # ========================================================

    submit_btn.click(
        fn=grocery_agent,
        inputs=user_input,
        outputs=output
    )

    user_input.submit(
        fn=grocery_agent,
        inputs=user_input,
        outputs=output
    )

    clear_btn.click(
        fn=lambda: ("", """
### 👋 Welcome!

Enter your grocery requirements on the left and click
**Generate Plan**.
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
