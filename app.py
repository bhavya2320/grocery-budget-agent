import os
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
        "Please add your Gemini API key "
        "as an environment variable."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=API_KEY
)


# ============================================================
# GROCERY DATA
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
They are not live market prices.
"""


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GroceryState(TypedDict):
    user_request: str
    final_report: str


# ============================================================
# GROCERY AGENT
# ============================================================

def grocery_agent_node(state):

    user_request = state["user_request"]

    prompt = f"""
You are an intelligent AI Grocery Budget Agent.

USER REQUEST:

{user_request}

GROCERY PRICE DATABASE:

{GROCERY_DATA}

Solve the user's grocery planning problem.

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
- Compare with the budget.
- Optimize the budget.
- Create a meal plan.
- Create a shopping checklist.
- Give money-saving tips.

Use ONLY the provided prices.

Do NOT invent prices.

If information is missing, make a reasonable assumption
and mention it.

Return:

# 🛒 GROCERY BUDGET REPORT

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

Continue for every requested day.

## 💡 Budget Optimization

Explain the optimization.

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
market prices. Please verify actual prices before purchasing.
"""

    try:

        response = llm.invoke(prompt)

        state["final_report"] = response.content

    except Exception as e:

        state["final_report"] = f"""
# ❌ Gemini Error

{str(e)}

Please try again later.
"""

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
# GRADIO FUNCTION
# ============================================================

def grocery_agent(user_request):

    if not user_request.strip():

        return "⚠️ Please enter your grocery requirements."

    result = grocery_graph.invoke({

        "user_request": user_request,

        "final_report": ""
    })

    return result["final_report"]


# ============================================================
# GRADIO UI
# ============================================================

demo = gr.Interface(

    fn=grocery_agent,

    inputs=gr.Textbox(

        label="🛒 Grocery Requirements",

        placeholder=(
            "Example: I have ₹3000 for groceries "
            "for 2 people for 7 days. "
            "We eat vegetarian Indian food. "
            "I already have rice and cooking oil."
        ),

        lines=7
    ),

    outputs=gr.Markdown(),

    title="🛒 Grocery Budget Agent",

    description="""
    AI-powered grocery planning assistant
    using LangGraph + Google Gemini.

    Enter your budget, household size,
    duration, food preferences and
    ingredients you already have.
    """,

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
        ]
    ]
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
