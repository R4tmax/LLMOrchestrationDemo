from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    loader = DirectoryLoader(r"C:\Users\kadle\Documents\PersonalLibrary\VŠE\Magistr\General\Applied deep learning", glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding' : 'utf-8', 'autodetect_encoding' : True})
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)