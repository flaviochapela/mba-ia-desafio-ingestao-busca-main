from search import search_prompt
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def main():
    
    user_question = input("Qual sua pergunta? ")
    context_prompt = search_prompt(user_question)

    PROMPT_TEMPLATE = """
    CONTEXTO:
    {context}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
    "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {pergunta}

    RESPONDA A "PERGUNTA DO USUÁRIO"
    """
    
    question_template = PromptTemplate(
        input_variables=["pergunta", "context"],
        template=PROMPT_TEMPLATE
    )

    model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

    chain = question_template | model

    result = chain.invoke({"pergunta": user_question, "context": context_prompt})
    print(result.content)


if __name__ == "__main__":
    main()