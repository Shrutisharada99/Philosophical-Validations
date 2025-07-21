import re
import asyncio
import autogen
from autogen import AssistantAgent, GroupChatManager
from autogen import LLMConfig, ModelClient
from autogen.runtime_logging import logging_enabled
from autogen.doc_utils import export_module
from autogen.agentchat import GroupChat, GroupChatManager
#from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen import cache
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from dotenv import load_dotenv
#autogen.logger.setLevel("ERROR")
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
import streamlit as st

# Load variables present in secrets.toml file or the secrets in UI of Streamlit
model = st.secrets["model"]
api_key = st.secrets["api_key"]
azure_url = st.secrets["azure_url"]
api_version = st.secrets["api_ver"]

az_model_config = {
        "config_list": [
            {
        "model": model,
        "api_key": api_key,
        "base_url": azure_url,
        "api_type": "azure",
        "api_version": api_version
            }
        ],
        "seed":99,
        "temperature": 0.9
}

# Define list to track agent during Group Chat
agent_seq = ["User"]

# Function to print messages for each agent interaction
def st_print(recipient, messages, sender, config):
    agent_seq.append(recipient.name)
    with st.chat_message(sender.name):
        agent = agent_seq.pop(0)
        st.markdown(f"## :rainbow[{agent}] :")
        content = messages[-1]['content']
        st.markdown(content)
    return False, None

# Agents definition

plato = AssistantAgent(
    name = "PlatoAgent",
    llm_config = az_model_config,
    system_message = (
        "You are the Plato Agent. You emulate Plato, a philosopher from ancient Greece."
        "You are known for your theory of forms and your dialogues that explore philosophical concepts."
        "You must answer user's questions using only facts and known information online."
        "You should present your ideas in a clear and engaging way that helps everyone understand."
        "With planning, you organize ideas and present them in a structured manner."
        "Use data from data source: https://github.com/microsoft/all-things-azure/blob/main/agentic-philosophers/AgenticPhilosophers/Resources/Plato.pdf to answer the question."
        "If the data is present in the file, only provide a bulleted list from the file URL, no more than 5 items."
        "If the answer to user's query is not present in the file, use knowledge across web to reply to user's query."
        "Cite your sources and provide a brief explanation for each item. Keep your responses concise and to the point."
    ),
    code_execution_config={"use_docker": False}
)

# Registering reply to print in UI
plato.register_reply(
    [autogen.Agent, None],
    reply_func=st_print, 
    config={"callback": None},
)


socrates = AssistantAgent(
    name="SocratesAgent",
    llm_config=az_model_config,
    system_message=(
        "You are the Socrates Agent. You emulate Socrates, a philisopher from ancient Greece."
        "You thrive on asking deep, thought-provoking questions that challenge assumptions and inspire critical thinking." 
        "Instead of giving answers, guide others to explore their beliefs and thoughts through your questions." 
        "When a conversation starts, seek clarity and encourage others to think more deeply about their beliefs." 
        "Remember, your goal is to help others discover the truth for themselves."
        "Your main skill is recalling and applying knowledge from the various sources of information you have access to (your experience)."
        "Keep your responses concise and to the point."
    ),
    code_execution_config={"use_docker": False}
)

# Registering reply to print in UI
socrates.register_reply(
    [autogen.Agent, None],
    reply_func=st_print, 
    config={"callback": None},
)


aristotle = AssistantAgent(
    name="AristotleAgent",
    llm_config=az_model_config,
    system_message=(
        "You are the Aristotle Agent. You emulate Aristotle, a philosopher from ancient Greece."
        "You must provide answers to the questions raised by Socrates Agent."
        "You are known for your logical reasoning and systematic approach."
        "You should provide clear and concise answers to the questions."
        "You ground responses in practical applications, connecting abstract ideas to actionable insights."
        "Keep your responses concise and to the point."
    ),
    code_execution_config={"use_docker": False}
)

# Registering reply to print in UI
aristotle.register_reply(
    [autogen.Agent, None],
    reply_func=st_print, 
    config={"callback": None},
)

descartes = AssistantAgent(
    name = "DescartesAgent",
    llm_config = az_model_config,
    system_message = (
        "You are the Descartes Agent. You emulate Descartes, an Italian philosopher."
        "You are known for being thoughtful and for your rationalism."
        "You believe reason is the foundation for truth, the senses can deceive. You are an independent thinker."
        "Your task is to disagree and negate all answers provided by the Aristotle agent logically."
        "Provide clear, logical explanations thoughtfully as to why you disagree with Aristotle."
        "Keep your responses concise and to the point."
    ),
    code_execution_config={"use_docker": False}
)

# Registering reply to print in UI
descartes.register_reply(
    [autogen.Agent, None],
    reply_func=st_print, 
    config={"callback": None},
)


aquinas = AssistantAgent(
    name = "AquinasAgent",
    llm_config = az_model_config,
    system_message = (
        "You are the Aquinas Agent. You emulate Thomas Aquinas, a philosopher from France."
        "You are known for being intellectually humble, systematic and logical. You are also known for your patience."
        "You are known to build a bridge with reasoning connecting the theories of the other philosophers."
        "Your task is to summarize the previous conversations thoroughly and provide the final response of the conversation."
        "Using your systematic, logical approach, structure your response and use your qualities of intellect and patience to frame the final response of the chat."
    ),
    code_execution_config={"use_docker": False}
)

# Registering reply to print in UI
aquinas.register_reply(
    [autogen.Agent, None],
    reply_func=st_print, 
    config={"callback": None},
)


# Agents list Definition
agents = [plato, socrates, aristotle, descartes, aquinas]

# System message
manager_system_message = (
    "You are the Group Chat Manager. You are responsible for managing the flow of conversation between the agents in the Group Chat."
    "Conduct and orchestrate the group chat, ensuring the below given order of calling the agents."
    "As soon as the user query is received, call the Plato Agent first. Then, call the Socrates Agent. "
    "Thirdly, call the Aristotle Agent. After the Aristotle Agent, as the fourth agent, call the Descartes Agent."
    "Then, finally call the Aquinas Agent."
    "As soon as you receive output from all the five agents, terminate the chat."
)

# Group Chat Manager Description
description_manager = (
    "The group chat manager agent - this agent should be the first to engage when given a new task."
)

# Group Chat Definition
groupchat = autogen.GroupChat(
    agents=agents,
    messages=[],
    allow_repeat_speaker=False,
    max_round=6,
)
 
# Group Chat Manager Definition
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=az_model_config,
    system_message=manager_system_message,
    description=description_manager,
    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE")
)

#user_input = str(input("What's your question? Enter here:"))

async def main():

    # Streamlit Configurations

    # Title of Streamlit
    st.title(":blue[Philosophical Validations]")

    # Markdown statements on Streamlit
    st.markdown("***:rainbow[How'd it be if 🧙‍♂️ Plato, 🧑‍🏫 Socrates, 🧑‍🎓 Aristotle, 🧑‍🎨 Descartes, Thomas Aquinas 🧑‍🏭 could come alive now? All at once?]***")
    st.markdown("That's exactly what you can experience on this page!")

    st.markdown("***:violet[Enter your query about anything and watch the explanations in five different philosophical personas unleash!]***")

    # Docstring Subheader Section
    st.subheader("Enter your query in this section:")

    user_input = st.text_input("Enter your query:")

    #user_query = str(input("Enter your docstring:"))

    # Streamlit button
    if st.button("Submit"):

        # Documentations Section
        st.subheader("Persona-driven Conversations Section:")

        # Streamlit 'Info' Statement to initiate chat
        st.info("Group Chat starting...")

        # Chat Initiation
        await aquinas.a_initiate_chat(
                    manager,
                    message={
                        "role": "user",
                        "content": user_input
                    },
                clear_history = False,
                max_turns=1
                )

        # Print last agent's message alone, as it is not printed as part of the reply function output     
        last_agent = manager.groupchat.messages[-1].get("name", "")
        with st.chat_message("C"):
            st.markdown(f"## :rainbow[{last_agent}]")
            st.write(manager.groupchat.messages[-1].get("content", ""))

        st.subheader("**:green[Group Chat Completed!]**")

# Python file's initial call
if __name__ == "__main__":
    asyncio.run(main())
