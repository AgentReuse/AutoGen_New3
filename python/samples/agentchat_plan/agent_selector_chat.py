from typing import List, Sequence, cast

import chainlit as cl
import yaml
from Response_reuse import SemanticCache
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent, BaseAgentEvent, BaseChatMessage
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken

# Example usage in another script:
from transit_intent import load_models, predict

#初始化
semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)

import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

@cl.step(type="tool")
async def search_web(query: str) -> str:
    return f"🌐 检索结果：'{query}' 的最新网页摘要如下……"

@cl.step(type="tool")
async def analyze_data(data: str) -> str:
    return f"📊 针对数据'{data}'的初步分析结果：……"


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    MAX_TURNS = 6
    print("message_len")
    print(len(messages))
    if len(messages) == 1:
        return "InputRefiner"
    if len(messages) == MAX_TURNS - 1:
        return "OutputSummarizer"
    return None



@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_cfg)

    input_refiner = AssistantAgent(
        name="InputRefiner",
        system_message="You are good at condensing user input into concise, structured, and information-dense task descriptions. Note: Your responses should be highly summarized, typically no more than 30 words. The input you provide is divided into sentences and keywords. The keywords must appear in the sentences. In the task description you generate, the keywords clearly stated in the input must be included and enclosed in curly braces ({}). When mentioning an entity value in your output sentence, wrap it with curly braces in the format {entity_type:entity_value}. For example, if the entity is {'transport_mode': 'train', 'source': 'jfk airport', 'destination': 'san francisco', 'date': 'next monday'}, you must refer to san francisco as {'destination': 'san francisco'} in your response.",
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    info_retriever = AssistantAgent(
        name="InfoRetriever",
        system_message="You are good at retrieving knowledge, examples and data related to the task. When necessary, you can call the search_web tool.",
        tools=[search_web],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    analyst = AssistantAgent(
        name="Analyst",
        system_message="You are good at conducting clear and organized analyses of given tasks or information, and can call on the analyze_data tool to assist in making judgments.",
        tools=[analyze_data],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    output_summarizer = AssistantAgent(
        name="OutputSummarizer",
        system_message="You do not directly engage in communication with other agents. You only need to make a systematic summary of the outputs given by other team members in the current context, which should be organized and easy to understand.",
        model_client=model_client,


        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    team = SelectorGroupChat(
        [input_refiner, info_retriever, analyst, output_summarizer],
        model_client=model_client,
        # selector_func=selector_func,  # 首尾定序，中间自由
        max_turns=6,
    )

    cl.user_session.set("input_refiner", input_refiner)
    cl.user_session.set("team", team)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Ticket",
            message="Is the train from JFK Airport to San Francisco running next Monday?"
        ),
    ]


@cl.on_message
async def chat(message: cl.Message) -> None:
    user_text = message.content
    embedding = semantic_cache.get_embedding(user_text)             #向量化
    similar_question, score = semantic_cache.search_similar_query(embedding)   #相似性搜索
    input_refiner = cl.user_session.get("input_refiner")
    refined = ""
    async for evt in input_refiner.on_messages_stream(
            messages=[TextMessage(content=user_text, source="user")],
            cancellation_token=CancellationToken(),
    ):
        if isinstance(evt, ModelClientStreamingChunkEvent):
            refined += evt.content

    team: SelectorGroupChat = cl.user_session.get("team")
    msg = cl.Message(content="")

    team = cast(SelectorGroupChat, cl.user_session.get("team"))

    # load_models()  # optional, uses default dirs
    load_models(intent_dir="transit_intent/bert_intent_model",
                slot_dir="transit_intent/bert_slot_model")
    intent = predict(user_text)
    print(intent)

    async for evt in team.run_stream(
        task=refined,
        cancellation_token=CancellationToken(),
    ):
        isReuse = 0 ## 0为不复用，1为计划复用，2为响应复用

        if score<0.75 :
            isReuse=0
        elif 0.75 <= score < 0.90:
            isReuse=1
        else:
            isReuse=2
        if isReuse == 0:
            agent_name = getattr(evt, "source", None) or getattr(getattr(evt, "chat_message", None), "source", None)

            if agent_name == "InputRefiner":
                print("InputRefiner has been selected.")
                if hasattr(evt, "content") and isinstance(evt.content, str):
                    semantic_cache.save_to_cache(user_text, None, evt.content)   #存储响应

        elif isReuse == 1:
            external_content = semantic_cache.cache[user_text]["plan"]  # 读取计划
            msg = TextMessage(source="InputRefiner", content=external_content)
            # team._group_chat_manager._message_thread.append(msg)
            team._group_chat_manager.update_message_thread(msg)

            

        elif isReuse == 2:
            external_content= semantic_cache.cache[user_text]["response"]  # 读取响应
            msg = cl.Message(author="OutputSummarizer", content=external_content)
            if msg is None:
                msg = cl.Message(author="OutputSummarizer", content="")
            if hasattr(evt, "content") and isinstance(evt.content, str):
                await msg.stream_token(evt.content)
            elif hasattr(evt, "content"):
                await msg.send()

        if agent_name == "OutputSummarizer":
            if msg is None:
                msg = cl.Message(author="OutputSummarizer", content="")
            if hasattr(evt, "content") and isinstance(evt.content, str):
                await msg.stream_token(evt.content)
            elif hasattr(evt, "content"):
                await msg.send()