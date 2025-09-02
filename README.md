# Strands Agents – Agent Initialization Parameters (Full README)

This README includes all explanations, code, raw notes, and outputs extracted directly from the Jupyter notebook.

# Strands Agents – Agent Initialization Parameters

This notebook explains the 15 key parameters for initializing an agent in the Strands SDK, with examples.

See [Strands Documentation](https://strandsagents.com/latest/documentation/docs/) for more.

## 1. `model`
Specifies which LLM backend the agent should use. Can be a string or model object.

```python
import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel

load_dotenv()
gemini_model = LiteLLMModel(
    model_id="gemini/gemini-2.5-pro",
    params={
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_tokens": 11000000,
        "temperature": 0.2
    }
)

```

```python

# To run this, you will first need to install the litellm optional dependency:
# pip install 'strands-agents[litellm]'
agent = Agent(
    model=gemini_model,
    system_prompt="You are a precise assistant."
)
# ask agent who is the president of the united states
response = agent("Who is the president of the United States?")
print(response)
```

## 2. `system_prompt`
Defines the agent’s core instructions/personality.

```python
agent = Agent(
    model=gemini_model,
    system_prompt="You are a legal advisor. Always answer with references to laws."
)
response = agent("Can a person be arrested without a warrant?")
print(response)
```

## 3. `messages`
Preloads a chat history at startup.

```python
agent = Agent(
    model=gemini_model,
    system_prompt="You are a tutor.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "who is the president of the united states?"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The president of the United States is Joe Biden."}]
        }
    ]
)

response = agent("what question did I ask before and what answer did you reply to my question ?")
print(response)
```

## 4. `tools`
Defines which external tools the agent can use.

```
 1. Why the Browser Tool Needs Bedrock AgentCore
	•	strands_tools.browser depends on AWS Bedrock AgentCore SDK.
	•	That SDK isn’t on PyPI; it’s distributed as part of the AWS Bedrock Agent runtime (inside AWS environments, like ECS, Lambda, or Bedrock itself).
	•	So: you can’t use the browser tool purely locally unless you pull in that SDK.

⸻

🔹 2. AWS Setup for Browser Tool

To use the Browser tool in AWS Strands Agents, you need:

✅ AWS Account with Bedrock + Bedrock AgentCore enabled
✅ IAM Role that allows:
	•	bedrock:InvokeModel (to call LLMs)
	•	bedrock-agent:InvokeAgent (to use tools)
	•	s3:* (if your tool writes/reads objects)
	•	ssm:GetParameter (if credentials are stored in SSM)

✅ Install Bedrock AgentCore in your environment:

If you’re running in AWS, the SDK is usually pre-installed.
If not, you’ll need to fetch the package from AWS’s distribution (not PyPI).

⸻

🔹 3. Using Browser Tool in an Agent

Once Bedrock AgentCore is available, your code looks like this:
```

```python
from strands_tools import calculator,browser

agent = Agent(
    model=gemini_model,
            system_prompt="Answer with facts.",
            tools=[ calculator,browser ]
)
print("Agent initialized with browser and calculator tools.")
response = agent("What is the population of Japan divided by 100?")
print(response)
```

```python
from strands_tools import calculator,http_request

agent = Agent(
    model=gemini_model,
    system_prompt="Answer with facts.",
    tools=[ calculator,http_request ]
)
response = agent("What is the population of Japan divided by 100?")
print(response)
```

## 5. `callback_handler`

Purpose of callback_handler
	•	A callback_handler in Strands controls logging, debugging, and monitoring of the agent’s execution steps.
	•	It’s useful when you want visibility into:
	•	What prompts the agent is sending to the LLM,
	•	What responses are being returned,
	•	Which tools are being called,
	•	Errors or intermediate steps.

	•	PrintingCallbackHandler() → A built-in handler that prints events to the console.
	•	callback_handler parameter → Passed to the Agent so it hooks into execution.
	•	Result: every time the agent runs, you’ll see step-by-step logs in your terminal (LLM calls, tool usage, errors, outputs).

When to Use
	•	During development and debugging (so you can trace what’s happening inside your agent).
	•	In production, you’d usually replace it with a more sophisticated handler, like:
	•	Sending logs to a file,
	•	Monitoring via an external system (Datadog, CloudWatch, etc.),
	•	Triggering alerts when errors occur.


```python
from strands_tools import calculator,http_request
from strands.handlers.callback_handler import PrintingCallbackHandler

agent = Agent(
    model=gemini_model,
    callback_handler=PrintingCallbackHandler(),#default
    system_prompt="Answer with facts.",
    tools=[ calculator,http_request ]
)

response = agent("What is the population of Japan divided by 100? use tools [calculator, http_request] for realtime population data. use http_request for websearch")
print(response)
```

## 6. `conversation_manager`

```

In Strands, your agent’s “context” is the running history the model sees each turn: user messages, the agent’s replies, tool calls/results, and your system prompt. Conversation managers decide which parts of that history to keep so you don’t blow past the model’s context window and cost/time budgets.  ￼

By default, Agent uses a SlidingWindowConversationManager, which maintains only the most recent portion of the chat.  ￼

Sliding window: how it works
	•	Keeps the last N message pairs (a user message + the assistant’s reply count as one pair). When a new turn arrives, the oldest pair falls out of the window.  ￼
	•	This trimming applies to the same message list the agent uses, which includes tool calls/results when you let Strands record them. (You can toggle this with record_direct_tool_call=False.)  ￼
	•	The goal is to stay within model context limits and keep recent, relevant dialog, while automatically shrinking history if a turn risks overflow.  
```

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models.openai import OpenAIModel  # swap with GeminiModel / BedrockModel if needed
import warnings
import asyncio

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel

load_dotenv()
gemini_model = LiteLLMModel(
    model_id="gemini/gemini-2.5-pro",
    params={
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_tokens": 11000000,
        "temperature": 0.2
    }
)


conv_manager = SlidingWindowConversationManager(window_size=5)

# 3. Create the agent
agent = Agent(
    model=gemini_model,
    system_prompt="You are a friendly chatbot .",
    conversation_manager=conv_manager,
)

print("✅ Agent initialized with sliding window memory!")


```

```python
# 4. Simulate a conversation
print("User: Hello, my name is vardhaman just remember  do not return anything ?")
agent("Hello, my name is vardhaman just remember  do not return anything ?")


```

```python
print("User: I like cricket. Remember that.")
response2 = agent("I like cricket. Remember that.")
```

```python
for i in range(1, 8):
    print(f"User: This is secret number {i}")
    resp = agent(f"This is secret number {i}.")
    
```

```python
# 5. Now test memory trimming
print("User: Do you still remember what sport I like?")
final_response = agent("Do you still remember what sport I like?")
print("Agent:", final_response)
```

```python
# 5. Now test memory trimming
print("User: can you summarize our conversation?")
print("Agent:", final_response)
```

### SummarizingConversationManager

```
Now let’s see SummarizingConversationManager — this one summarizes older history instead of discarding it.

⸻

🔎 Concept
	•	SlidingWindow → Keeps last N messages only, forgets everything else.
	•	Summarizing → Keeps the last few messages plus a running summary of everything older.

This means:
	•	The agent remembers key facts (like “User likes cricket”), even after many turns.
	•	Instead of dropping older messages, they get compressed into a short summary string that gets prepended each time.
```

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager,SummarizingConversationManager
from strands.models.openai import OpenAIModel  # swap with GeminiModel / BedrockModel if needed
import warnings
import asyncio

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel

load_dotenv()
gemini_model = LiteLLMModel(
    model_id="gemini/gemini-2.5-pro",
    params={
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_tokens": 11000000,
        "temperature": 0.2
    }
)

conv_manager = SummarizingConversationManager(
    summary_ratio=0.5,
    preserve_recent_messages=5,
    summarization_system_prompt=(
        "Summarize user facts, preferences, and goals. "
        "Always preserve names, dates, locations, and commitments and numbers."
    )
)
# 3. Create the agent
agent = Agent(
    model=gemini_model,
    system_prompt="You are a friendly chatbot that remembers what I say",
    conversation_manager=conv_manager,
)

print("✅ Agent initialized with summarizing memory!")



```

```python
# 4. Simulate a conversation
msg="My name is vardhaman , I love to play cricket, just remember do not return anything "
print(f"User: {msg}")
resp=agent(msg)
```

```python
for i in range(8):
    msg=f"secret code is {i} just remember this do not return anything"
    print(f"User: {msg}")
    resp=agent(msg)
```

```python
# 4. Simulate a conversation
msg=" how many number of recent messages you remembered ?"
print(f"User: {msg}")
resp=agent(msg)
```

```python
msg=" but in conv_manager I set limit to recent messages to 5 how come you remembered all messages ? "
print(f"User: {msg}")
resp=agent(msg)
```

## 7. `record_direct_tool_call`

```

🔹 What is record_direct_tool_call?

When you create an Agent, you can call tools directly via agent.tool.tool_name(...).
This flag decides whether those direct tool calls are recorded in the conversation/chat history or not.
	•	True → Tool calls are logged as if they happened in the conversation (traceable, visible in logs).
	•	False → Tool calls are executed but not logged (invisible in chat history).


👉 So the rule of thumb:
	•	True = full trace (great for debugging, auditing, or when you want transparency).
	•	False = hide results from history (good for silent background work or when you don’t want tool chatter cluttering up logs).
```

```python
from strands import Agent
from strands_tools import calculator
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models.openai import OpenAIModel  # swap with GeminiModel / BedrockModel if needed
import warnings
import asyncio

import os
from dotenv import load_dotenv
from strands import Agent
from strands.models.litellm import LiteLLMModel

load_dotenv()
gemini_model = LiteLLMModel(
    model_id="gemini/gemini-2.5-pro",
    params={
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_tokens": 11000000,
        "temperature": 0.2
    }
)

# 2. Create the agent
agent = Agent(
    model=gemini_model,
    system_prompt="You are a friendly chatbot.",
)

# --- Case A: record tool calls in history ---
agent_true = Agent(tools=[calculator], model=gemini_model, record_direct_tool_call=True)
agent_true('add 2 + 2')
print("Chat history (True):")
for msg in agent_true.messages:
    print(msg)


# --- Case B: do NOT record tool calls ---
agent_false = Agent(tools=[calculator], model=gemini_model, record_direct_tool_call=False)
agent_false('add 2 + 2')
print("\nChat history (False):")
for msg in agent_false.messages:
    print(msg)
```

```python
agent = Agent(record_direct_tool_call=False)
print("Agent initialized to not record direct tool calls.")
# result = agent.tool.calculator("2+2")  # Won’t appear in chat history
```

## 8. `load_tools_from_directory`
Auto-imports Python files from ./tools/ folder.

```python
agent = Agent(
    system_prompt="You can use custom tools.",
    load_tools_from_directory=True,
    model=gemini_model
)
print("Agent initialized to load tools from the './tools' directory.")

agent.tool.greet(name="Vardhaman")
agent.tool.weather(city="Bengaluru")
```

## 9. `trace_attributes`

```
trace_attributes option in Strands is for observability + logging. It attaches custom metadata (tags like team, env, request_id, etc.) to every trace (agent run, tool call, message).

Here’s how you can test and inspect that the trace attributes are actually being recorded:
```

```
Great question 🙌 — on the surface trace_attributes just looks like “extra metadata.” But in practice it’s super useful for observability, debugging, and operations when you’re running agents in production.

Here are some real-world use cases:

⸻

🔹 1. Multi-team Environments

If several teams share the same agent infrastructure:
	•	Add trace_attributes={"team": "data-eng"}
	•	Observability dashboards (Datadog, Grafana, Jaeger, etc.) can filter traces by team, so each team only sees their own runs.

👉 Example: Your Data Engineering team’s runs won’t get mixed up with the Support team’s agent runs.

⸻

🔹 2. Environment / Deployment Tracking

Attach environment tags like:

trace_attributes={"env": "staging"}

Now you can:
	•	Separate traces between dev, staging, and prod.
	•	Verify new code behaves in staging before pushing to prod.
	•	Roll up errors by environment (e.g., “Only staging is failing → check new deployment”).

⸻

🔹 3. Customer / Request Correlation

Attach IDs that help you correlate logs:

trace_attributes={"customer_id": "12345", "request_id": "abc-xyz"}

	•	Lets you trace one user’s request across the whole system.
	•	If a customer says “my report failed,” you can grep by customer_id.
	•	If you’re using OpenTelemetry, all logs and spans for that request are grouped.

⸻

🔹 4. A/B Testing & Feature Flags

You can attach experiment tags:

trace_attributes={"experiment": "prompt_v2"}

	•	Compare latency, error rates, or outcomes between prompt versions.
	•	Easily rollback by filtering results.

⸻

🔹 5. Performance & Cost Monitoring

Suppose you attach:

trace_attributes={"workflow": "find_email", "model": "gemini-flash"}

You can now:
	•	Break down latency and cost by workflow.
	•	Answer questions like “Which workflow is costing the most tokens?”
	•	Spot slow workflows in dashboards.

⸻

🔹 6. Compliance & Auditing

In regulated industries, you might tag:

trace_attributes={"region": "EU", "gdpr": "true"}

	•	Ensures you can prove that EU user data stayed in EU runs.
	•	Helps auditors filter only traces subject to compliance.

⸻

⚡ TL;DR

trace_attributes = labels for observability.
They make it possible to filter, debug, and analyze agent activity at scale across teams, environments, customers, and workflows.

⸻

```

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(
    system_prompt="Tracing demo.",
    tools=[calculator],
    model=gemini_model,
    trace_attributes={"team": "data-eng", "env": "staging"}
)

# Run something so a trace is produced
result = agent("What is 3 + 3?")
print("Agent result:", result)

# Inspect the messages
print("\n--- Messages ---")
for msg in agent.messages:
    print(msg)

# Inspect the trace attributes (directly on the agent)
print("\n--- Trace attributes ---")
print(agent.trace_attributes)
```

```python
agent = Agent(
    system_prompt="Tracing demo.",
    trace_attributes={"team": "data-eng", "env": "staging"}
)
print("Agent initialized with trace attributes.")
```

## 10. `agent_id`
Unique identifier for the agent instance.

```python
agent = Agent(
    system_prompt="Travel bot.",
    agent_id="travel-agent-01"
)
print("Agent initialized with a unique agent_id.")
```

## 11. `name`
Friendly display name for the agent.

```python
agent = Agent(
    system_prompt="Finance expert.",
    name="FinanceBot"
)
print("Agent initialized with the name 'FinanceBot'.")
```

## 12. `description`
Describes the agent’s purpose.

```python
agent = Agent(
    system_prompt="You are a compliance advisor.",
    description="Helps check financial documents for compliance."
)
print("Agent initialized with a description.")
```

## 13. `state`

```
🟢 What is state?

Think of agent.state like a small notebook the agent always carries.
	•	You (the programmer) can write things in it.
	•	The agent can look at it later.
	•	It stays around even when the agent finishes answering one question.

⸻

🟢 Why do we need it?

Normally, when you chat:
	•	The agent only remembers the conversation messages.
	•	If you want to keep track of a score, settings, or progress, you’d have to repeat it in text.

With state, you can keep extra memory outside the chat.

⸻

🟢 Example in plain words
	1.	Start with score = 0

agent = Agent(
    system_prompt="Game assistant.",
    state=AgentState(data={"score": 0})
)

	2.	Update the score when the player earns points:

agent.state.data["score"] += 5
print(agent.state.data["score"])   # now it’s 5

Even if the agent forgets the old messages, the score is still there.

⸻

🟢 Real-life simple uses
	•	Game bot → keep track of a player’s score.
	•	Quiz bot → remember how many answers the user got right.
	•	Shopping bot → remember items added to the cart.
	•	Preference bot → remember user wants answers in “short form” or “Kannada language”.

⸻

👉 In one line:
state is like the agent’s little pocket diary where you can store numbers, flags, or settings between runs.

```

```python
from strands import Agent

# Initialize agent with some state
initial = {
    "user_preferences": {"theme": "dark"},
    "session_count": 0
}
agent = Agent(state=initial,model=gemini_model)

# Access state
print("Theme:", agent.state.get("user_preferences"))        # {'theme': 'dark'}

# Update state
agent.state.set("last_action", "login")
agent.state.set("session_count", agent.state.get("session_count") + 1)

print("Session count:", agent.state.get("session_count"))    # 1
print("Last action:", agent.state.get("last_action"))        # 'login'
```

## 14. `hooks`

```
Got it 👍 — here’s a purely detailed explanation, no code, so you can keep it as notes.

⸻

📝 Strands Hooks — Detailed Explanation

⸻

🔹 What Hooks Are

Hooks in Strands are like event listeners that let you run custom logic at important steps in the agent’s lifecycle.
They don’t change the agent’s reasoning itself, but they let you observe, log, validate, or extend what’s happening inside.

Think of them as “interceptors” — they trigger before, during, or after specific events.

⸻

🔹 How Hooks Work
	•	Hooks are organized around events (for example, “before invocation” or “after message added”).
	•	To use hooks, you provide a HookProvider to the agent.
	•	Inside the provider, you say: “when event X happens, call my function Y.”
	•	Each event object gives you information about what’s happening (messages, agent state, etc.).

⸻

🔹 Main Events in Your Version

When we checked your installed version, the following events are available:
	1.	AgentInitializedEvent
	•	Fires when the agent is created.
	•	Useful for logging startup, loading configs, setting up state.
	2.	BeforeInvocationEvent
	•	Fires right before the agent processes a user input.
	•	Lets you inspect the last message or block an invalid input.
	3.	AfterInvocationEvent
	•	Fires after the agent finishes its reasoning for a turn.
	•	Lets you capture outputs, update logs, or save results in a database.
	4.	MessageAddedEvent
	•	Fires whenever a new message is added (from user, assistant, or tool).
	•	This is the most powerful in your version — it gives you visibility into:
	•	User prompts
	•	Assistant text replies
	•	Tool calls
	•	Tool results (success or error)

⸻

🔹 How Tool Calls Show Up

Your version doesn’t have separate tool events. Instead, tool activity appears as special messages:
	•	Tool call → Assistant emits a message with a toolUse block.
Example: “Agent is about to call the calculator tool.”
	•	Tool result (success) → User role emits a toolResult with "status": "success".
Example: “Calculator returned 42.”
	•	Tool result (error) → Same structure, but with "status": "error".
Example: “Error: division by zero.”

By watching MessageAddedEvent, you can detect tool start, tool result, and whether it succeeded or failed.

⸻

🔹 The Hook Lifecycle During a Run

Let’s walk through what happens if you ask the agent “What is 5 * 6?”:
	1.	AgentInitializedEvent
	•	Fired once, when you first create the agent.
	2.	MessageAddedEvent (user)
	•	User message “What is 5 * 6?” is added.
	3.	BeforeInvocationEvent
	•	Agent is about to process that user input.
	4.	MessageAddedEvent (assistant, toolUse)
	•	Assistant decides to call the calculator tool.
	•	Tool name and input appear in the event.
	5.	MessageAddedEvent (user, toolResult)
	•	Calculator returns the result → status = success.
	•	If it failed, status = error.
	6.	MessageAddedEvent (assistant, text)
	•	Assistant produces the final text answer: “OK. 5 * 6 = 30.”
	7.	AfterInvocationEvent
	•	Fired at the very end of the cycle, when the reply is complete.

⸻

🔹 What Hooks Are Useful For
	1.	Observability / Logging
	•	Record every input, output, and tool call for later analysis.
	•	Capture how often each tool is used.
	2.	Error Tracking
	•	Detect when a tool result has "status": "error".
	•	Alert or log failures without stopping the agent.
	3.	Metrics
	•	Count how many runs the agent has processed.
	•	Count tool successes vs errors.
	•	Measure latency (e.g., time between toolUse and toolResult).
	4.	Auditing / Compliance
	•	Keep a trace of what the user asked, what tools were invoked, and what the system replied.
	•	Useful for regulated environments.
	5.	Customization
	•	You can enforce rules (e.g., block certain inputs before invocation).
	•	Modify how results are handled after invocation.
	•	Add custom logging formats for integration with monitoring systems.

⸻

🔹 Key Differences Between Versions
	•	Your version: only has general lifecycle events + message-added events. Tools must be tracked through messages.
	•	Newer versions: also have dedicated tool events (BeforeToolInvocationEvent, AfterToolInvocationEvent) and model-level events (BeforeModelInvocationEvent, AfterModelInvocationEvent).

⸻

🔹 Mental Model
	•	AgentInitializedEvent → setup / startup
	•	BeforeInvocationEvent → input validation / pre-processing
	•	MessageAddedEvent → observe user input, assistant replies, tool calls, tool results (success or error)
	•	AfterInvocationEvent → wrap-up, logging, persistence

⸻

✅ In short: Hooks = observability, control, and customization points.
They give you insight into the “inside” of the agent loop and let you log, measure, and react to what’s happening.
```

```python
from strands import Agent
from strands_tools import calculator
from strands.hooks import HookProvider
from strands.hooks.registry import HookRegistry
from strands.hooks.events import (
    AgentInitializedEvent,
    BeforeInvocationEvent,
    AfterInvocationEvent,
    MessageAddedEvent,
)

class GeneralHooks(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AgentInitializedEvent, self.on_init)
        registry.add_callback(BeforeInvocationEvent, self.before_invocation)
        registry.add_callback(AfterInvocationEvent, self.after_invocation)
        registry.add_callback(MessageAddedEvent, self.on_message)

    # ---- Lifecycle hooks ----
    def on_init(self, event: AgentInitializedEvent):
        print("[HOOK] Agent initialized")

    def before_invocation(self, event: BeforeInvocationEvent):
        last_msg = event.agent.messages[-1] if event.agent.messages else None
        print(f"[HOOK] Before invocation → last message: {last_msg}")

    def after_invocation(self, event: AfterInvocationEvent):
        last_msg = event.agent.messages[-1] if event.agent.messages else None
        print(f"[HOOK] After invocation → final message: {last_msg}")

    # ---- Message hook (handles tools & replies) ----
    def on_message(self, event: MessageAddedEvent):
        msg = event.message   # dict
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            print(f"[HOOK] User said: {content}")

        elif role == "assistant" and isinstance(content, list):
            block = content[0]

            # Tool call
            if "toolUse" in block:
                tool_use = block["toolUse"]
                print(f"[HOOK] Tool call → {tool_use['name']} with input {tool_use['input']}")

            # Tool result (success or error)
            elif "toolResult" in block:
                tool_result = block["toolResult"]
                status = tool_result.get("status")
                if status == "success":
                    print(f"[HOOK] Tool returned SUCCESS → {tool_result}")
                elif status == "error":
                    print(f"[HOOK] Tool returned ERROR → {tool_result}")

            # Normal assistant text
            elif "text" in block:
                print(f"[HOOK] Assistant replied → {block['text']}")

            else:
                print(f"[HOOK] Assistant message (unrecognized): {content}")

# ---------------------------------------
# Agent with generalized hooks
# ---------------------------------------
agent = Agent(
    system_prompt="Demo agent with generalized hooks",
    model=gemini_model,
    tools=[calculator],
    hooks=[GeneralHooks()]
)

print("\n--- Running successful tool call ---")
agent("What is 5 * 6?")

print("\n--- Running tool error case ---")
agent("What is 1 / 0?")   # calculator will error here
```

## 15. `session_manager`

```
Here’s the “session_manager” idea in plain, practical terms so you can use it confidently.

What a session manager is

Think of the session manager as the agent’s filing cabinet.
Each session_id is a separate folder in that cabinet. The manager’s job is to:
	•	create a folder the first time it sees a new session_id,
	•	keep the conversation history (and optionally other per-session data) in that folder,
	•	pull the right folder before the agent replies,
	•	put the updated folder back after the reply.

With a session manager, multiple users—or multiple parallel conversations from the same user—don’t mix.

Why you need it
	•	Multi-user apps (web/chatbots/Slack): each user gets their own conversation thread.
	•	Parallel tasks: you can run “shopping-assistant:userA” and “returns-assistant:userA” at the same time without the messages leaking into each other.
	•	Long-lived context: you can come back hours later with the same session_id and the agent remembers.

What “DefaultSessionManager” does
	•	It’s a simple, in-memory implementation.
	•	It keeps a map like: session_id → list of messages (+ metadata).
	•	It survives across calls inside the same Python process.
	•	If the process restarts, the memory is gone (i.e., no persistence).
	•	Good for local notebooks, demos, unit tests; not ideal for production or multiple workers.

How sessions change behavior
	•	Without a session_id: all calls share one implicit conversation (easy to step on each other).
	•	With a session_id: each call is routed to that session’s own history.
“user1” and “user2” can ask the same question and get answers tailored to their own prior context.

What exactly gets stored per session

Minimum: the message history (user messages, assistant replies, toolUse/toolResult blocks).
Depending on your setup, the manager can also store:
	•	per-session “state” (e.g., a cart, a running score, workflow progress),
	•	lightweight metadata (timestamps, model used, last tool called),
	•	housekeeping (token counters, last access time for TTL cleanup).

How this differs from other knobs
	•	state: mutable variables you set for logic (counters, flags, preferences). Can be tied to a session or global; the session manager is where you persist it per session_id.
	•	messages/chat history: the raw transcript. Session manager organizes this by session.
	•	trace_attributes: tags for observability (team/env/request_id). They help search & monitor, but don’t store conversation content.
	•	hooks: event callbacks; you can use them to augment the session (e.g., log every run, enforce limits) but they aren’t storage.

Production patterns (what teams usually do)
	•	Swap the default for a persistent session manager backed by:
	•	Redis (fast, TTL eviction, great for many short sessions),
	•	Postgres/RDS (auditable, queryable, durable),
	•	S3/Blob (cheap long-term transcripts),
	•	or a hybrid (Redis cache + DB archive).
	•	Add TTL & eviction rules so idle folders are cleaned up automatically.
	•	Encrypt PII and restrict cross-session access (don’t let one user’s folder be read by another).
	•	Version your stored structures (in case you later change message format).

Common gotchas (and fixes)
	•	Mixing prompts: changing the system prompt mid-session can confuse the model. Start a new session_id if you change core behavior.
	•	ID collisions: generate robust session_ids (UUIDs, or userID+channelID). Don’t reuse the same ID across different products/roles.
	•	Memory growth: long chats get big. Use a conversation manager (sliding window/summarizer) alongside the session manager so the stored transcript stays model-friendly.
	•	Multi-process apps: the default in-memory manager won’t share across workers. Use Redis/DB so all app instances see the same sessions.

When to create a new session vs reuse
	•	Reuse when it’s the same ongoing task for the same person (shopping cart, ticket thread).
	•	New session when the purpose changes (new project, role switch), after a long idle period, or when you want a clean slate to avoid old context bias.

How to think about it with your workflows
	•	Customer support: session_id = supportTicketId. The bot stays scoped to that ticket’s history.
	•	Data/AI pipelines: session_id = jobRunId. The agent remembers earlier steps and artifacts for the run.
	•	Land/real-estate lead intake: session_id = leadId. The bot collects documents over multiple visits without mixing leads.
	•	Upwork/micro-SaaS demos: keep each demo room separate so investors/clients don’t see cross-talk.

Quick checklist
	•	Decide your session_id strategy (what uniquely identifies a conversation).
	•	Pick a session backend (memory for dev; Redis/DB for prod).
	•	Set retention/TTL and max history length (pair with a summarizer if needed).
	•	Log who/what/when (for audits) but avoid storing secrets in clear text.
	•	Test concurrency: multiple calls with the same session_id shouldn’t race or corrupt the folder.

```

```python
from strands.session import DefaultSessionManager

agent = Agent(
    system_prompt="Shopping assistant.",
    session_manager=DefaultSessionManager()
)
print("Agent initialized with a default session manager.")
# Separate sessions
# response1 = agent("I want shoes", session_id="user1")
# response2 = agent("I want laptops", session_id="user2")
# print(f"User 1: {response1}")
# print(f"User 2: {response2}")
```

```python
import asyncio
import asyncpg
import nest_asyncio
import json
from datetime import datetime
from dataclasses import asdict
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.session_repository import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage
from strands_tools import calculator
nest_asyncio.apply()

def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


class PostgresRepository(SessionRepository):
    def __init__(self, dsn):
        self.dsn = dsn

    async def _get_conn(self):
        return await asyncpg.connect(self.dsn)

    def _current_ts(self):
        return datetime.utcnow().isoformat()

    # -----------------------
    # Session methods
    # -----------------------
    def create_session(self, session: Session) -> Session:
        data = asdict(session)
        data['created_at'] = self._current_ts()
        data['updated_at'] = self._current_ts()
        run_sync(self._exec(
            """
            INSERT INTO sessions_meta (session_id, data)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (session_id) DO UPDATE SET data=$2::jsonb
            """,
            session.session_id,
            json.dumps(data)   # ✅ dict → JSON string
        ))
        return session

    def read_session(self, session_id: str) -> Session | None:
        row = run_sync(self._fetchrow(
            "SELECT data FROM sessions_meta WHERE session_id = $1",
            session_id
        ))
        if not row:
            return None
        return Session.from_dict(json.loads(row['data']))  # ✅ JSON string → dict

    # -----------------------
    # Agent methods
    # -----------------------
    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        data = asdict(session_agent)
        data['created_at'] = self._current_ts()
        data['updated_at'] = self._current_ts()
        run_sync(self._exec(
            """
            INSERT INTO agents_meta (session_id, agent_id, data)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (session_id, agent_id) DO UPDATE SET data=$3::jsonb
            """,
            session_id,
            session_agent.agent_id,
            json.dumps(data)   # ✅ dict → JSON string
        ))

    def read_agent(self, session_id: str, agent_id: str) -> SessionAgent | None:
        row = run_sync(self._fetchrow(
            "SELECT data FROM agents_meta WHERE session_id=$1 AND agent_id=$2",
            session_id, agent_id
        ))
        return SessionAgent.from_dict(json.loads(row['data'])) if row else None

    def update_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        data = asdict(session_agent)
        data['updated_at'] = self._current_ts()
        run_sync(self._exec(
            "UPDATE agents_meta SET data=$1::jsonb WHERE session_id=$2 AND agent_id=$3",
            json.dumps(data),   # ✅ dict → JSON string
            session_id,
            session_agent.agent_id
        ))

    # -----------------------
    # Message methods
    # -----------------------
    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        data = asdict(session_message)
        data['created_at'] = self._current_ts()
        run_sync(self._exec(
            """
            INSERT INTO messages (session_id, agent_id, message_id, data)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (session_id, agent_id, message_id) DO UPDATE SET data=$4::jsonb
            """,
            session_id,
            agent_id,
            session_message.message_id,
            json.dumps(data)   # ✅ dict → JSON string
        ))

    def read_message(self, session_id: str, agent_id: str, message_id: str) -> SessionMessage | None:
        row = run_sync(self._fetchrow(
            "SELECT data FROM messages WHERE session_id=$1 AND agent_id=$2 AND message_id=$3",
            session_id, agent_id, message_id
        ))
        return SessionMessage.from_dict(json.loads(row['data'])) if row else None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        data = asdict(session_message)
        data['updated_at'] = self._current_ts()
        run_sync(self._exec(
            "UPDATE messages SET data=$1::jsonb WHERE session_id=$2 AND agent_id=$3 AND message_id=$4",
            json.dumps(data),   # ✅ dict → JSON string
            session_id,
            agent_id,
            session_message.message_id
        ))

    def list_messages(self, session_id: str, agent_id: str, limit=None, offset=0) -> list[SessionMessage]:
        rows = run_sync(self._fetch(
            "SELECT data FROM messages WHERE session_id=$1 AND agent_id=$2 ORDER BY message_id ASC LIMIT $3 OFFSET $4",
            session_id, agent_id, limit or 100, offset
        ))
        return [SessionMessage.from_dict(json.loads(r['data'])) for r in rows]

    # -----------------------
    # Internal helpers
    # -----------------------
    async def _exec(self, query, *args):
        conn = await self._get_conn()
        try:
            await conn.execute(query, *args)
        finally:
            await conn.close()

    async def _fetchrow(self, query, *args):
        conn = await self._get_conn()
        try:
            return await conn.fetchrow(query, *args)
        finally:
            await conn.close()

    async def _fetch(self, query, *args):
        conn = await self._get_conn()
        try:
            return await conn.fetch(query, *args)
        finally:
            await conn.close()
```

```python
repo = PostgresRepository("postgresql://vardhamanrparappanavar:password@localhost/postgres")
session_manager = RepositorySessionManager(session_id="varun", session_repository=repo)

agent = Agent(
    system_prompt=" you are help assistant remember about my personal details",
    tools=[calculator],
    model=gemini_model,
    session_manager=session_manager
)

resp1 = agent("My name is vardhaman ", session_id="varun")
resp2 = agent("I like to play football", session_id="varun")
resp3 = agent("I hate golf", session_id="varun")
```

```python
repo = PostgresRepository("postgresql://vardhamanrparappanavar:password@localhost/postgres")
session_manager = RepositorySessionManager(session_id="varun", session_repository=repo)

agent = Agent(
    system_prompt=" you are help assistant",
    tools=[calculator],
    model=gemini_model,
    session_manager=session_manager
)

resp1 = agent("tell me about me ", session_id="varun")
```

## 📊 Quick Summary Table

| Param | Purpose | Example Use |
|-------|---------|-------------|
| model | Choose LLM | GPT-4 vs Palmyra-X5 |
| system_prompt | Define role | “Legal assistant” |
| messages | Preload history | Resume old chat |
| tools | Enable functions | web_search, calc |
| callback_handler | Logging/debug | Print steps |
| conversation_manager | Short-term memory | Keep last 5 msgs |
| record_direct_tool_call | Log tool calls? | Hide/surface |
| load_tools_from_directory | Auto-load tools | ./tools/weather.py |
| trace_attributes | Observability | env=staging |
| agent_id | Unique ID | “finance-01” |
| name | Display name | “FinanceBot” |
| description | Metadata | “Compliance checker” |
| state | Internal memory | {score: 0} |
| hooks | Inject custom logic | Before/after reply |
| session_manager | Long-term sessions | Keep user-specific chats |

