# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC TODO change how we can create or point to an existing chatbot
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-ron_joy` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0925-195633-7abao24z/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Data preparation for LLM Chatbot RAG
# MAGIC
# MAGIC ## Building our knowledge base and preparing our documents for Databricks Vector Search
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC In this notebook, we'll prepare data for our Vector Search Index.
# MAGIC
# MAGIC Preparing high quality data is key for your chatbot performance. We recommend taking time implementing this with your own dataset.
# MAGIC
# MAGIC For this example, we will use Databricks documentation from [docs.databricks.com](docs.databricks.com):
# MAGIC - Download the web pages
# MAGIC - Split the pages in small chunks
# MAGIC - Extract the text from the HTML content
# MAGIC
# MAGIC Thankfully, Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale.
# MAGIC
# MAGIC *Note: While some processing in this notebook is specific to our dataset (exmple: splitting chunks around `h2` elements), **we strongly recommend getting familiar with the overall process and replicate that on your own dataset**.*

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC TODO pass to custom catalog to run statement
# MAGIC
# MAGIC TODO make sure to use an ML runtime
# MAGIC
# MAGIC TODO build Volume for dropping data into init

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=akraemer $db=custom_llm_demo $reset_all_data=false

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO add this to setup
# MAGIC -- TODO add instructions to drop in particular folder
# MAGIC CREATE VOLUME IF NOT EXISTS akraemer.custom_llm_demo.sandbox

# COMMAND ----------

pip install langchain pymupdf unstructured python-docx

# COMMAND ----------

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

# COMMAND ----------

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents


# COMMAND ----------

source_directory = "/Volumes/akraemer/custom_llm_demo/sandbox/landing"
# load_documents(vol_dir)

try:
    documents = load_documents(source_directory)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

except:
    print("No new documents to load")

# COMMAND ----------

from pprint import pprint
pprint(documents)

# COMMAND ----------

# DBTITLE 1,Load Delta Tables from Documents
# Filter out None values (URLs that couldn't be fetched or didn't have the specified div)
valid_results = [doc for doc in documents if doc is not None]

#Save the content in a raw table
spark.createDataFrame(documents).write.mode('overwrite').saveAsTable('raw_documentation')
spark.sql("ALTER TABLE raw_documentation SET OWNER TO `account users`;")
display(spark.table('raw_documentation').limit(20))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Splitting documentation pages into small chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC LLM models typically have a maximum input window size, and you won't be able to compute embbeddings for very long texts.
# MAGIC In addition, the bigger your context is, the longer inference will take.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC
# MAGIC - Split document in small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends of your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one.
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference.
# MAGIC
# MAGIC ### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count how many tokens will be created for a given sequence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer and context size limit you'll be using here matches your embedding model. To do so, we'll be using the `tiktoken` library to count GPT-3.5 tokens with its tokenizer: `tiktoken.encoding_for_model("gpt-3.5-turbo")`

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's now split our entire dataset using this function using a pandas UDF.
# MAGIC
# MAGIC We will also extract the title from the page (based on the `h1` tag)

# COMMAND ----------

display(spark.table('raw_documentation').limit(20))

# COMMAND ----------

df = (
    spark.table("raw_documentation")
    .withColumn("title", col("metadata.file_path"))
    .withColumn("url", col("metadata.source"))
    .withColumnRenamed("page_content", "content")
    .drop("metadata")
)
# display(df.limit(5))
# Save back the results to our final table
# Note that we only do it if the table is empty, because it'll trigger an full indexation and we want to avoid this
if (
    not spark.catalog.tableExists(f"{catalog}.{db}.customer_documentation")
    or spark.table("customer_documentation").count() < 50
):
    df.write.mode("overwrite").saveAsTable("customer_documentation")

spark.sql("ALTER TABLE customer_documentation SET OWNER TO `account users`;")

# COMMAND ----------

display(spark.table("customer_documentation"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our dataset is now ready! Let's create our Vector Search Index.
# MAGIC
# MAGIC Our dataset is now ready, and saved as a Delta Lake table.
# MAGIC
# MAGIC We could easily deploy this part as a production-grade job, leveraging Delta Live Table capabilities to incrementally consume and cleanup document updates.
# MAGIC
# MAGIC Remember, this is the real power of the Lakehouse: one unified platform for data preparation, analysis and AI.
# MAGIC
# MAGIC Next: Open the [02-Creating-Vector-Index]($./02-Creating-Vector-Index) notebook and create our embedding endpoint and index.
