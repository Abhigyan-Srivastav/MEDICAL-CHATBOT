from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import traceback

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM from HuggingFace")

        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.3,
            max_new_tokens=256,
            return_full_text=False,
            huggingfacehub_api_token=hf_token,
            task="conversational"
        )
        
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            # We strictly request text-generation from HF directly
            task="text-generation", 
            # Force a higher timeout in case the serverless model is sleeping
            timeout=300,
            # Optional: adjust model temperature for medical context accuracy
            temperature=0.1,
            max_new_tokens=256,
            return_full_text=False
        )

        logger.info("LLM loaded successfully")
        return llm
    except Exception as e:
        error_message  =CustomException("Failed to load LLM", e)
        logger.error(str(error_message))
        print("--- FULL TRACEBACK ---")
        traceback.print_exc()

