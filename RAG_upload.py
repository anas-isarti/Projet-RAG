import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate



os.environ["MISTRAL_API_KEY"] = "mets_ta_cle_api_mistral_ici"
DOSSIER_BDD = "chroma_db_data"

# ============================================
#   CHARGEMENT DU MODELE D'EMBEDDINGS
#   on met en cache pour ne pas recharger a chaque interaction
# ============================================

@st.cache_resource
def get_embeddings():
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vector_store():
    # charge la base existante sur le disque
    return Chroma(
        persist_directory=DOSSIER_BDD,
        embedding_function=get_embeddings()
    )

# ============================================
#   FONCTION D'INDEXATION

# ============================================

def indexer_pdf(pdf_bytes, pdf_name):
    # on sauvegarde le PDF dans un fichier temporaire
    # car PyPDFLoader a besoin d'un vrai fichier sur le disque
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # etape 1 : on charge le PDF
    loader = PyPDFLoader(tmp_path)
    pages  = loader.load()

    # on ajoute le vrai nom du fichier dans les metadonnees
    for page in pages:
        page.metadata["source"] = pdf_name

    # etape 2 : on decoupe en morceaux
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(pages)

    # etape 3 : on vectorise et on stocke dans chromadb
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DOSSIER_BDD
    )

    # on supprime le fichier temporaire
    os.unlink(tmp_path)

    return len(chunks)

# ============================================
#   INTERFACE STREAMLIT
# ============================================

st.title(" Le super Chatbot RAG d'Anas - Question/Reponse sur document")

# --- section upload dans la sidebar ---
with st.sidebar:
    st.header("📁 Charger un document")

    uploaded_file = st.file_uploader("Uploader un PDF", type=["pdf"])

    if uploaded_file:
        if st.button("📥 Indexer ce document"):
            with st.spinner("Indexation en cours... (1-2 min la premiere fois)"):
                nb_chunks = indexer_pdf(uploaded_file.read(), uploaded_file.name)
                # on vide le cache pour recharger la base avec le nouveau document
                load_vector_store.clear()
                st.success(f"✅ {uploaded_file.name} indexé ({nb_chunks} chunks)")

# --- chargement de la base de donnees ---
if not os.path.exists(DOSSIER_BDD):
    st.info("👈 Commencez par uploader un PDF dans la barre latérale.")
    st.stop()

vector_store = load_vector_store()

# le retriever va chercher les 10 morceaux les plus proches de la question
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# LLm et prompt
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

prompt_template = ChatPromptTemplate.from_template("""
Tu es un assistant expert en analyse de documents.
Reponds a la question en utilisant le contexte suivant.

Regles importantes :
- Tu peux faire des deductions logiques a partir du contexte
- Si la question utilise des synonymes ou des reformulations, cherche le concept equivalent dans le contexte
- Si tu trouves une information partiellement liee a la question, utilise-la et precise que c est une deduction
- Dis uniquement "Je ne trouve pas cette information" si le contexte ne contient vraiment aucune information liee au sujet
"

Contexte :
{context}

Question :
{input}
""")

# --- zone de question ---
st.divider()
user_question = st.text_input("Ta question :")

if user_question:
    try:
        # etape 1 : on recupere les morceaux pertinents depuis chroma
        relevant_docs = retriever.invoke(user_question)

        if not relevant_docs:
            st.warning("Aucun document pertinent trouve dans la base.")
        else:
            # etape 2 : on construit le contexte et on envoie a Mistral
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            final_prompt = prompt_template.invoke({
                "context": context_text,
                "input": user_question
            })

            with st.spinner("Mistral réfléchit..."):
                reponse = llm.invoke(final_prompt)
                st.write(reponse.content)

            # on affiche les sources pour verifier que la reponse est bien fondee
            with st.expander("Sources utilisees"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Morceau {i+1}** (page {doc.metadata.get('page', '?') + 1}) :")
                    st.text(doc.page_content)
                    st.divider()

    except Exception as e:
        st.error(f"Erreur : {e}")
