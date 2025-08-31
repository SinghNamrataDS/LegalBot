from PyPDF2 import PdfReader
import os
import re
from utils.custom_exception import CustomException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config
 
 
class DataConverter:
    def __init__(self,file_path:str):
        if isinstance(file_path, str):
            self.file_path = [file_path]
        else:
            self.file_path = file_path
 
    def extract_clean_data(self):
        try:
            combined_text = ""
           
            for file in self.file_path:
                if isinstance(file, str):
                    pdf_reader = PdfReader(file)
                elif hasattr(file, 'pages'):
                    pdf_reader = file
                else:
                    raise CustomException("Input must be either a file path (string) or PdfReader object")
 
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:  
                        text += page_text + "\n"
 
                combined_text += text + "\n"
 
            cleaned_text = self.clean_legal_text(combined_text)
            return cleaned_text
 
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
   
    def clean_legal_text(self,text):
 
        if not text or not isinstance(text, str):
            return ""
 
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
 
            # Remove page numbers and headers/footers
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
            text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text)
 
            # Fix broken words across lines (hyphenation)
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
 
            # Remove excessive dots/underscores (form fields)
            text = re.sub(r'\.{3,}', '...', text)
            text = re.sub(r'_{3,}', '', text)
 
            # Clean up multiple newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
 
            # Remove trailing spaces from lines
            text = re.sub(r' +\n', '\n', text)
 
            # Clean section references for legal documents
            text = re.sub(r'\bSec\.?\s*(\d+)', r'Section \1', text)
            text = re.sub(r'\bArt\.?\s*(\d+)', r'Article \1', text)
 
            return text.strip()
 
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text if isinstance(text, str) else ""
       
 
 
    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
 


# if __name__=="__main__":
#     documents = Config.DOCUMENT_PATHS
#     obj = DataConverter(documents)
#     raw_text = obj.extract_clean_data()
#     text_chunks = obj.get_text_chunks(raw_text)
#     print(len(text_chunks))