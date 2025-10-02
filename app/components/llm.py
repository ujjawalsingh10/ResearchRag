from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info('Loading LLM from HuggingFace')
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            max_new_tokens=256,
            temperature=0.3,
            huggingfacehub_api_token=hf_token,
            return_full_text=False
        )

        llm = ChatHuggingFace(llm=llm_endpoint)
        logger.info('LLM loaded successfully...')
        return llm
    
    except Exception as e:
        error_message = CustomException('Failed to load a LLM', e)
        logger.error(str(error_message))
        return None
