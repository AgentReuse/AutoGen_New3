from typing import List, Sequence, cast

import chainlit as cl
import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent, BaseAgentEvent, BaseChatMessage
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken
from autogen_agentchat.base import Response

# Example usage in another script:
from python.transit_intent import load_models, predict

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
        system_message="You are good at condensing user input into concise, structured, and information-dense task descriptions. Note: Your responses should be highly summarized, typically no more than 20 words. The input you provide is divided into sentences and keywords. The keywords must appear in the sentences. In the task description you generate, the keywords clearly stated in the input must be included and enclosed in curly braces ({}).",
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



    cl.user_session.set("team", team)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="法律咨询",
            message="我最近被公司解雇，对方没有提前一个月通知我，只支付了一个月工资补偿，请问我能否要求更多补偿？有哪些相关的法律依据和案例？我需要注意哪些风险？"
        ),
        cl.Starter(
            label="旅游攻略",
            message="我想去云南自由行5天，能帮我设计一份详细路线和注意事项吗？"
        ),
        cl.Starter(
            label="数据分析",
            message="请帮我分析一份销售数据，给出增长瓶颈和改进建议。原始数据如下：......"
        ),
    ]


@cl.on_message
async def chat(message: cl.Message) -> None:
    user_text = message.content
    input_refiner = cl.user_session.get("input_refiner")
    refined = ""
    async for evt in input_refiner.on_messages_stream(
            messages=[TextMessage(content=user_text, source="user")],
            cancellation_token=CancellationToken(),
    ):
        if isinstance(evt, ModelClientStreamingChunkEvent):
            refined += evt.content

    initial_thread = [TextMessage(source="InputRefiner", content=refined)]

    team = cast(SelectorGroupChat, cl.user_session.get("team"))

    # load_models()  # optional, uses default dirs
    load_models(intent_dir="python/transit_intent/bert_intent_model",
                slot_dir="python/transit_intent/bert_slot_model")
    intent = predict(user_text)
    print(intent)

    async for evt in team.run_stream(
        messages=initial_thread,
        cancellation_token=CancellationToken(),
    ):
        isReuse = 0 ## 0为不复用，1为计划复用，2为响应复用

        if isReuse == 0:
            agent_name = getattr(evt, "source", None) or getattr(getattr(evt, "chat_message", None), "source", None)

            if agent_name == "InputRefiner":
                if hasattr(evt, "content") and isinstance(evt.content, str):
                    with open("input_refiner.txt", "a", encoding="utf-8") as f:
                        f.write(evt.content)

        elif isReuse == 1:
            external_content = "【这是我希望 InputRefiner 说的话，由我外部指定】"
            msg = TextMessage(source="InputRefiner", content=external_content)
            # team._group_chat_manager._message_thread.append(msg)
            team._group_chat_manager.update_message_thread(msg)

        elif isReuse == 2:
            pass

        if agent_name == "OutputSummarizer":
            if msg is None:
                msg = cl.Message(author="OutputSummarizer", content="")
            if hasattr(evt, "content") and isinstance(evt.content, str):
                await msg.stream_token(evt.content)
            elif hasattr(evt, "content"):
                await msg.send()
