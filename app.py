import os
import streamlit as st

# Configuration Streamlit
st.set_page_config(
    page_title="Chat Intelligent avec Llama ",
    page_icon="💬",
    layout="wide"
)

from dotenv import load_dotenv
import hashlib
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy import MetaData
from typing import Optional, List, Mapping, Any

# Imports LangChain
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.agents import load_tools, initialize_agent
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, TextLoader

# Client Groq
from groq import Groq as GroqClient

# Wrapper personnalisé Groq pour LangChain
class GroqLLM(LLM):
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.7
    max_tokens: int = 1024
    groq_api_key: str
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = GroqClient(api_key=self.groq_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop
        )
        return response.choices[0].message.content
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Charger les variables d'environnement
load_dotenv()

# Configuration SQLAlchemy
metadata = MetaData()
Base = declarative_base(metadata=metadata)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String(100), nullable=False, default="Nouvelle conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model = Column(String(50), nullable=False, default="llama3-8b-8192")
    temperature = Column(Float, nullable=False, default=0.7)
    max_tokens = Column(Integer, nullable=False, default=1024)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation = relationship("Conversation", back_populates="messages")

# Configuration de la base de données
DB_HOST = "localhost"  # Utilise le nom du service Docker
DATABASE_URL = f"postgresql+psycopg2://chat_user:chat_password@{DB_HOST}:5432/chatbotdb"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Connexion à la base de données réussie")
        
        Base.metadata.create_all(bind=engine)
        print("Tables créées avec succès")
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
            print("Tables existantes:", [row[0] for row in result])
            
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de la base de données: {e}")
        raise

# Fonctions d'accès aux données
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user(db, username: str, password: str):
    hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
    try:
        user = User(username=username, password=hashed_pwd)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return None
    except Exception as e:
        db.rollback()
        st.error(f"Erreur lors de la création de l'utilisateur: {e}")
        return None

def authenticate_user(db, username: str, password: str):
    hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
    try:
        user = db.query(User).filter(User.username == username, User.password == hashed_pwd).first()
        return user
    except Exception as e:
        st.error(f"Erreur d'authentification: {e}")
        return None

def create_conversation(db, user_id: int, title: str = "Nouvelle conversation"):
    try:
        conversation = Conversation(user_id=user_id, title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    except Exception as e:
        db.rollback()
        st.error(f"Erreur lors de la création de la conversation: {e}")
        return None

def get_user_conversations(db, user_id: int):
    try:
        return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).all()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des conversations: {e}")
        return []

def get_conversation_messages(db, conversation_id: int):
    try:
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp).all()
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    except Exception as e:
        st.error(f"Erreur lors de la récupération des messages: {e}")
        return []

def save_message(db, conversation_id: int, role: str, content: str):
    try:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        db.add(message)
        db.commit()
    except Exception as e:
        db.rollback()
        st.error(f"Erreur lors de l'enregistrement du message: {e}")

def update_conversation_title(db, conversation_id: int, new_title: str):
    try:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            conversation.title = new_title
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        st.error(f"Erreur lors de la mise à jour du titre: {e}")
        return False

def load_conversation_memory(db, conversation_id: int):
    """Charge l'historique de conversation dans la mémoire LangChain"""
    messages = get_conversation_messages(db, conversation_id)
    st.session_state.memory.clear()
    
    for msg in messages:
        if msg["role"] == "user":
            st.session_state.memory.chat_memory.add_user_message(msg["content"])
        else:
            st.session_state.memory.chat_memory.add_ai_message(msg["content"])

def initialize_rag():
    """Initialise le système RAG avec des documents exemple"""
    try:
        # Exemple avec un document web
        loader = WebBaseLoader("https://fr.wikipedia.org/wiki/Paris")
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
        st.success("Système RAG initialisé avec des données exemple")
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation RAG: {e}")

# Initialisation
try:
    init_db()
    
    # Configuration LangChain avec notre wrapper Groq
    llm = GroqLLM(
        model_name="llama3-8b-8192",
        temperature=0.7,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Prompt template amélioré
    template = """Vous êtes un assistant expert. Répondez de manière précise et utile.
    
Historique de conversation:
{history}

Question: {input}
Réponse utile:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Initialisation de la mémoire et de la chaîne
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=True
        )
    
    # Initialisation des outils (optionnel)
    if 'tools' not in st.session_state:
        try:
            st.session_state.tools = load_tools(["llm-math"], llm=llm)
            st.session_state.agent = initialize_agent(
                st.session_state.tools,
                llm,
                agent="zero-shot-react-description",
                verbose=True
            )
        except Exception as e:
            st.warning(f"Impossible de charger les outils: {e}")
    
    # Initialisation RAG (optionnel)
    if 'vectorstore' not in st.session_state:
        initialize_rag()
    
except Exception as e:
    st.error(f"Erreur d'initialisation: {e}")
    st.stop()



# Authentification
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.current_conversation = None
    st.session_state.messages = []

# Page de connexion/inscription
if not st.session_state.authenticated:
    st.title("Authentification")
    
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])
    
    with tab1:
        with st.form("Connexion"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter")
            
            if submit:
                try:
                    with SessionLocal() as db:
                        user = authenticate_user(db, username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            
                            # Charge la dernière conversation ou en crée une nouvelle
                            conversations = get_user_conversations(db, user.id)
                            if conversations:
                                st.session_state.current_conversation = conversations[0].id
                                st.session_state.messages = get_conversation_messages(db, conversations[0].id)
                                load_conversation_memory(db, conversations[0].id)
                            else:
                                new_conv = create_conversation(db, user.id)
                                st.session_state.current_conversation = new_conv.id
                                st.session_state.messages = []
                            
                            st.rerun()
                        else:
                            st.error("Identifiants incorrects")
                except Exception as e:
                    st.error(f"Erreur de connexion: {e}")
    
    with tab2:
        with st.form("Inscription"):
            new_username = st.text_input("Choisissez un nom d'utilisateur")
            new_password = st.text_input("Choisissez un mot de passe", type="password")
            confirm_password = st.text_input("Confirmez le mot de passe", type="password")
            submit = st.form_submit_button("Créer un compte")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif len(new_password) < 6:
                    st.error("Le mot de passe doit faire au moins 6 caractères")
                else:
                    try:
                        with SessionLocal() as db:
                            user = create_user(db, new_username, new_password)
                            if user:
                                st.success("Compte créé avec succès! Veuillez vous connecter.")
                            else:
                                st.error("Ce nom d'utilisateur est déjà pris")
                    except Exception as e:
                        st.error(f"Erreur lors de la création du compte: {e}")
    
    st.stop()

# Interface principale
st.title("💬 Chat Intelligent avec Llama")
st.write(f"Connecté en tant que: {st.session_state.user.username}")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Mode de fonctionnement
    st.session_state.mode = st.radio(
        "Mode de fonctionnement",
        ["Conversation", "Recherche documentaire (RAG)", "Agent avec outils"],
        index=0
    )
    
    st.header("Mes conversations")
    if st.button("+ Nouvelle conversation"):
        try:
            with SessionLocal() as db:
                conversation = create_conversation(db, st.session_state.user.id)
                st.session_state.current_conversation = conversation.id
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de la création d'une nouvelle conversation: {e}")

    try:
        with SessionLocal() as db:
            conversations = get_user_conversations(db, st.session_state.user.id)
            
            if conversations:
                selected_conv = st.selectbox(
                    "Sélectionnez une conversation",
                    options=[conv.id for conv in conversations],
                    format_func=lambda x: next(conv.title for conv in conversations if conv.id == x),
                    index=0
                )
                
                if selected_conv != st.session_state.current_conversation:
                    st.session_state.current_conversation = selected_conv
                    st.session_state.messages = get_conversation_messages(db, selected_conv)
                    load_conversation_memory(db, selected_conv)
                    st.rerun()
                
                current_title = next(conv.title for conv in conversations if conv.id == selected_conv)
                new_title = st.text_input("Renommer la conversation", value=current_title)
                if new_title != current_title:
                    if update_conversation_title(db, selected_conv, new_title):
                        st.rerun()
            else:
                st.info("Aucune conversation existante")
                conversation = create_conversation(db, st.session_state.user.id)
                st.session_state.current_conversation = conversation.id
                st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du chargement des conversations: {e}")

    # Bouton pour réinitialiser RAG
    if st.button("Recharger les documents RAG"):
        initialize_rag()

# Afficher les messages
if 'messages' not in st.session_state:
    try:
        with SessionLocal() as db:
            st.session_state.messages = get_conversation_messages(db, st.session_state.current_conversation) if st.session_state.current_conversation else []
    except Exception as e:
        st.error(f"Erreur lors du chargement des messages: {e}")
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bouton de déconnexion
if st.button("Déconnexion"):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.current_conversation = None
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        with SessionLocal() as db:
            save_message(db, st.session_state.current_conversation, "user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                # Sélection du mode de fonctionnement
                if st.session_state.mode == "Recherche documentaire (RAG)":
                    # Utilisation du système RAG
                    docs = st.session_state.vectorstore.similarity_search(prompt)
                    context = "\n".join([doc.page_content for doc in docs[:3]])
                    enhanced_prompt = f"Contexte:\n{context}\n\nQuestion: {prompt}\nRéponse:"
                    
                    response = st.session_state.conversation.run(enhanced_prompt)
                elif st.session_state.mode == "Agent avec outils":
                    # Utilisation de l'agent avec outils
                    response = st.session_state.agent.run(prompt)
                else:
                    # Mode conversation standard
                    response = st.session_state.conversation.run(prompt)
                
                st.markdown(response)
            
            save_message(db, st.session_state.current_conversation, "assistant", response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Erreur lors de l'envoi du message: {e}")