from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from src.config import Config
from src.data_converter import DataConverter
from langchain_core.documents import Document
 
class DataIngestor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDED_MODEL)
       
        self.vstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name = "legal_bot",
            api_endpoint = Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )
   
    def ingest(self,load_existing=True):
        if load_existing==True:
            return self.vstore
        docs = DataConverter(Config.DOCUMENT_PATHS)
        raw_text = docs.extract_clean_data()
        text_chunks = docs.get_text_chunks(raw_text)
 
        documents = []
        for i, chunk in enumerate(text_chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "legal_documents",
                    "document_type": "legal_code"
                }
            )
            documents.append(doc)
        self.vstore.add_documents(documents)
        return self.vstore
   
# if __name__=="__main__":
#     ingestor = DataIngestor()
#     ingestor.ingest(load_existing=False)
 