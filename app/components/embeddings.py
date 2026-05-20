from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import HF_TOKEN

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing HF Embedding Model")
        model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs={
        "token": HF_TOKEN
    })

        logger.info("Huggingface embedding model loaded successfully.")
        return model
    except Exception as e:
        error_message = CustomException("Error occurred while loading embedding model", e)
        logger.error(str(error_message))
        raise error_message


    