import gradio as gr
import uuid
from src.data_ingestion import DataIngestor
from src.rag_chain import RAGChainBuilder
 
# Global variables to store the RAG system
vector_store = None
rag_builder = None
 
def initialize_rag_system():
    """Initialize the RAG system once at startup"""
    global vector_store, rag_builder
   
    print("Initializing Legal Bot...")
   
    # Load vector store
    ingestor = DataIngestor()
    vector_store = ingestor.ingest(load_existing=True)
   
    # Build RAG chain
    rag_builder = RAGChainBuilder(vector_store)
   
    print("Legal Bot initialized successfully!")
 
def get_response(message, history, session_id):
    """Get response from the RAG system"""
    try:
        if not rag_builder:
            return " Legal Bot not initialized. Please refresh the page."
       
        # Get response from RAG chain
        response = rag_builder.get_response(message, session_id)
       
        return response["answer"]
   
    except Exception as e:
        return f" Error: {str(e)}"
 
def chat_interface(message, history):
    """Main chat interface function"""
    # Generate a session ID for this conversation
    session_id = "session_" + str(uuid.uuid4())[:8]
   
    # Get response
    response = get_response(message, history, session_id)
   
    # Update history
    history.append((message, response))
   
    return "", history
 
def clear_chat():
    """Clear the chat history"""
    return [], []
 
# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
.chat-message {
    padding: 10px !important;
    margin: 5px !important;
    border-radius: 10px !important;
}
"""
 
def main():
    """Main function to run the Gradio app"""
   
    # Initialize RAG system
    initialize_rag_system()
   
    # Create Gradio interface
    with gr.Blocks(
        css=custom_css,
        title="Indian Legal Bot",
        theme=gr.themes.Soft()
    ) as app:
       
        # Header
        gr.Markdown(
            """
            #  Indian Legal Assistant Bot
           
            Ask questions about **Indian Legal Codes**:
            - **BNS** (Bharatiya Nyaya Sanhita)
            - **BSA** (Bharatiya Sakshya Adhiniyam)
            - **BNSS** (Bharatiya Nagarik Suraksha Sanhita)
           
            ---
            """
        )
       
        # Chat interface
        chatbot = gr.Chatbot(
            height=500,
            placeholder="Hello! I'm your Indian Legal Assistant. Ask me anything about Indian legal codes.",
            label="Legal Assistant Chat"
        )
       
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your legal question here...",
                label="Your Question",
                lines=2,
                scale=4
            )
            send_btn = gr.Button("Send 📤", variant="primary", scale=1)
       
        with gr.Row():
            clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
       
        # Example questions
        gr.Markdown("### 💡 Example Questions:")
       
        example_questions = [
            "What is the punishment for theft under BNS?",
            "Explain the concept of evidence under BSA",
            "What are the procedures for arrest under BNSS?",
            "What constitutes murder under Indian law?",
            "How is bail determined under BNSS?"
        ]
       
        with gr.Row():
            for i in range(0, len(example_questions), 2):
                with gr.Column():
                    if i < len(example_questions):
                        gr.Button(
                            example_questions[i],
                            variant="outline"
                        ).click(
                            lambda x=example_questions[i]: (x, []),
                            outputs=[msg, chatbot]
                        )
                    if i + 1 < len(example_questions):
                        gr.Button(
                            example_questions[i + 1],
                            variant="outline"
                        ).click(
                            lambda x=example_questions[i + 1]: (x, []),
                            outputs=[msg, chatbot]
                        )
       
        gr.Markdown(
            """
            ---
            **Disclaimer**: This is an AI assistant for informational purposes only.
            It does not provide legal advice. Consult a qualified lawyer for legal matters.
            """
        )
       
        msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
        send_btn.click(chat_interface, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_chat, outputs=[chatbot, msg])
   
    app.launch(
        server_name="127.0.0.1",  # 127.0.0.1
        server_port=9095,   #9090
        share=True,
        debug=False,
        inbrowser=True
    )
 
if __name__ == "__main__":
    main()
 