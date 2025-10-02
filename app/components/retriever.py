from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
You are an expert in Audio Speech Analysis
Answer the following questions in 2-3 lines using the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables= ['context', 'question'])

def create_qa_chain():
    try:
        logger.info('Loading vector store for context')
        db = load_vector_store()

        if db is None:
            raise CustomException('Vector Store not present or empty')

        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,
                       hf_token=HF_TOKEN)
        
        if llm is None:
            raise CustomException('LLM not loaded')
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = 'stuff',
            retriever = db.as_retriever(search_kwargs={'k':3}),
            return_source_documents = False,
            chain_type_kwargs={'prompt':set_custom_prompt()}
        )

        logger.info('Successfully created the QA chain')
        return qa_chain

    except Exception as e:
        error_message = CustomException('Failed to make a QA chain')
        logger.error(str(error_message))
        raise e