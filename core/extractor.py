#I need actionable items, decisions, Questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
import os 

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)


def build_chain(system_prompt: str):

    llm = get_llm()

    return(RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) 
    | llm |StrOutputParser()
    )


def extract_actionable_items(transcript: str) -> str:
    """Extract actionable items from the transcript."""
    system_prompt = (
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke(transcript)



def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions made during the meeting from the transcript."""
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    """Extract questions raised during the meeting from the transcript."""
    prompt = ChatPromptTemplate.from_template (
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
        "Transcript:\n{transcript}"
    )

    llm = get_llm()

    chain = prompt | llm 

    return chain.invoke(transcript).content