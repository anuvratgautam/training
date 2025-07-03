import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Search:
    def __init__(self, dim: int = 384):
        self.dim = dim                    # Embedding size (e.g., 384, 768)
        self.index = faiss.IndexFlatIP(dim)  # IP = Inner Product(dot) = Cosine 
        self.texts : list= []                  # Stores original texts
        self.embeddings : list = []            # Stores embeddings for later use

    def add_documents(self, embeddings: np.ndarray, texts: list[str]):
        """
        embeddings: numpy array of shape (n, dim)
        texts: original text chunks corresponding to each embedding
        """
        self.index.add(embeddings)
        self.embeddings.extend(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        query_embedding: numpy array of shape (1, dim)
        Returns top_k most similar texts
        """
        _, indices = self.index.search(query_embedding, top_k)
        results = [self.texts[i] for i in indices[0]]
        return results
    
    def embed_doc(self,path : str) -> None:
        try:
            loc = path
            if ".pdf" in loc:
                texts = ""
                try:
                    with fitz.open(loc) as doc:
                        for page in doc:  # Iterate through each page
                            texts += page.get_text()
                except FileNotFoundError:
                    print("File Not Found")
                except Exception as e:
                    print(f"Error extracting text: {e}")
                
            elif ".txt" in loc:
                with open(loc) as doc:
                    texts = doc.read()
                
            else:
                raise ValueError(f"The doc type {loc} is not supported")
            
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 200,
                chunk_overlap = 50
            )
            chunks = splitter.split_text(texts)
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(chunks, normalize_embeddings=True)
            
            self.add_documents(np.array(embedding),chunks)
            
        except FileNotFoundError:
            print(f"File Not Found at '{loc}'")
        except Exception as e :
            print(f"Faced an error {e}")
        
    def embed_user(self, user_input : str):
        try:
            user_input = user_input

            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(user_input, normalize_embeddings=True)
            
            return embedding.reshape(1,-1)
        except Exception as e :
            print(f"Faced an error {e}")


