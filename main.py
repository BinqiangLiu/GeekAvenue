import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

#This function will go through pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]
  
st.set_page_config(layout="centered", page_title="AI智能文档问答")
st.header("支持多语言的AI智能文档问答程序")
st.write("---")
  
#file uploader
#uploaded_files = st.file_uploader("上传文档/Upload documents",accept_multiple_files=True, type=["txt","pdf"])
uploaded_files = st.file_uploader("上传文档/Upload documents",accept_multiple_files=True, type=["pdf"])
st.write("---")

if uploaded_files is None:
  st.info(f"""上传文档开始分析/Upload files to analyse""")
elif uploaded_files:
  st.write("已上传" + str(len(uploaded_files)) + "个文档/" + str(len(uploaded_files)) + " document(s) loaded")
  
  textify_output = read_and_textify(uploaded_files)
  
  documents = textify_output[0]
  sources = textify_output[1]
  
  #extract embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
  #vstore with metadata. Here we will store page numbers.
  vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
  #deciding model
  model_name = "gpt-3.5-turbo"
  # model_name = "gpt-4"

  retriever = vStore.as_retriever()
  retriever.search_kwargs = {'k':2}

  #initiate model
  llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
  model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
  st.header("开始问答/Ask your data")
  user_q = st.text_area("请在此输入您的问题/Enter your questions here")
  
  if st.button("获取答案/Get Response"):
    try:
      with st.spinner("答案获取中，请稍候/Your response is on the way, please wait"):
        result = model({"question":user_q}, return_only_outputs=True)
        st.subheader('AI智能回答/Your AI response')
        st.write(result['answer'])
        st.subheader('出处/Source:')
        st.write(result['sources'])
    except Exception as e:
      st.error(f"噢，出错了！/An error occurred: {e}")
      st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
