from src.config import Config
from src.data_ingestion import DataIngestor
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
 
 
class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}
 
    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
   
    def build_chain(self):
        # Create retriever with appropriate settings for legal documents
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Increased to get more relevant legal context
        )
 
        # Context prompt for reformulating questions based on chat history
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the chat history and the latest user question,
                         reformulate the question to be a standalone question that includes
                         relevant context from the conversation history.
                         Keep legal terminology and section references intact."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")  
        ])
 
        # QA prompt specifically designed for legal assistance
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable legal assistant specializing in Indian law,
                         particularly the Bharatiya Nyaya Sanhita (BNS), Bharatiya Sakshya Adhiniyam (BSA),
                         and Bharatiya Nagarik Suraksha Sanhita (BNSS).
 
                         Instructions:
                         1. Provide accurate legal information based strictly on the context provided
                         2. Cite specific sections, articles, or provisions when applicable
                         3. If the context doesn't contain sufficient information, clearly state this
                         4. Use clear, professional language suitable for legal queries
                         5. Structure your response with relevant legal principles first, then specific provisions
                         6. Do not provide legal advice - only explain what the law states
                         
                         Context from legal documents:
                         {context}
                         
                         Please answer the following legal query based on the provided context:"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")  
        ])
 
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        )
 
        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        )
 
        # Create the main RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
 
        # Return chain with message history
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
 
    def get_response(self, query: str, session_id: str = "default"):
        chain = self.build_chain()
       
        response = chain.invoke(
            {"input": query},
            {"configurable": {"session_id": session_id}}
        )
       
        return {
            "answer": response["answer"],
            "source_documents": response.get("context", []),
            "session_id": session_id
        }
 
    def clear_history(self, session_id: str):
        """Clear conversation history for a specific session"""
        if session_id in self.history_store:
            del self.history_store[session_id]
            return f"History cleared for session: {session_id}"
        return f"No history found for session: {session_id}"
 
 