# Chatbot RAG — Question/Réponse sur documents PDF

Système de question/réponse intelligent basé sur la technologie RAG (Retrieval Augmented Generation). Permet d'uploader n'importe quel document PDF et de poser des questions en langage naturel dessus. Le système répond en citant précisément ses sources (numéro de page).

---

## Démonstration

```
Utilisateur : "Quelle est la date de naissance mentionnée dans le document ?"

Chatbot : "Selon le document (page 2), la date de naissance est le 15 mars 1998."

Sources utilisées :
  Morceau 1 (page 2) : "...né le 15 mars 1998 à Paris..."
```

---

## Comment ça fonctionne

Le principe du RAG (Retrieval Augmented Generation) est de ne pas demander au modèle de langage de répondre de mémoire, mais de lui fournir les passages pertinents du document avant de lui poser la question. Ça évite les hallucinations et permet de citer les sources.

```
PDF uploadé
    ↓
Découpage en morceaux de texte (chunks)
    ↓
Transformation de chaque morceau en vecteur numérique (embeddings)
    ↓
Stockage dans une base de données vectorielle (ChromaDB)
    ↓
Question posée par l'utilisateur
    ↓
Recherche des morceaux les plus proches sémantiquement
    ↓
Envoi des morceaux + question à Mistral
    ↓
Réponse générée avec citations des sources
```

---

## Stack technique

| Technologie | Rôle | Pourquoi ce choix |
|---|---|---|
| **Streamlit** | Interface web | Prototypage rapide en Python, pas besoin de HTML/CSS |
| **LangChain** | Orchestration | Connecte tous les outils, simplifie le pipeline RAG |
| **PyPDFLoader** | Extraction PDF | Lecture page par page avec conservation des métadonnées |
| **RecursiveCharacterTextSplitter** | Découpage du texte | Coupe aux endroits logiques (fin de paragraphe, fin de phrase) |
| **HuggingFace — all-MiniLM-L6-v2** | Embeddings | Modèle léger (90MB), tourne en local, pas de clé API nécessaire |
| **ChromaDB** | Base vectorielle | 100% local, données qui ne quittent pas le PC, pas d'infrastructure |
| **Mistral AI** | Génération de réponse | Modèle performant, entreprise française, bon rapport qualité/coût |

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- Un compte Mistral AI avec une clé API → [console.mistral.ai](https://console.mistral.ai)

### Étapes

**1. Cloner le projet**
```bash
git clone https://github.com/ton-username/chatbot-rag.git
cd chatbot-rag
```

**2. Créer et activer un environnement virtuel**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Installer les dépendances**
```bash
pip install streamlit langchain langchain-community langchain-huggingface langchain-chroma langchain-mistralai chromadb sentence-transformers
```

**4. Ajouter ta clé API Mistral**

Ouvre le fichier `RAG_upload.py` et remplace à la ligne 13 :
```python
os.environ["MISTRAL_API_KEY"] = "mets_ta_cle_api_mistral_ici"
```
par ta vraie clé :
```python
os.environ["MISTRAL_API_KEY"] = "ta_cle_api"
```

**5. Lancer l'application**
```bash
streamlit run RAG_upload.py
```

L'application s'ouvre automatiquement dans ton navigateur à l'adresse `http://localhost:8501`

---

## Utilisation

**1. Uploader un document**
Dans la barre latérale gauche, clique sur "Browse files" et sélectionne un fichier PDF.

**2. Indexer le document**
Clique sur le bouton "Indexer ce document". La première fois, le modèle d'embeddings se télécharge automatiquement (~90MB, 1-2 minutes). Les fois suivantes c'est instantané.

**3. Poser une question**
Tape ta question dans le champ texte et appuie sur Entrée.

**4. Consulter les sources**
Clique sur "Sources utilisées" pour voir les passages exacts du document qui ont servi à construire la réponse.

---

## Structure du projet

```
chatbot-rag/
├── RAG_upload.py       ← fichier principal, contient tout le code
├── README.md           ← ce fichier
├── chroma_db_data/     ← base vectorielle (créée automatiquement)
└── data/               ← dossier de données (créé automatiquement)
```

---

## Paramètres configurables

Tous les paramètres sont en haut du fichier `RAG_upload.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `MISTRAL_API_KEY` | à renseigner | Clé API Mistral |
| `MISTRAL_MODEL` | `mistral-small-latest` | Modèle Mistral utilisé |
| `k` dans le retriever | `10` | Nombre de chunks envoyés à Mistral |
| `chunk_size` | `1000` | Taille maximale d'un morceau de texte |
| `chunk_overlap` | `200` | Chevauchement entre deux morceaux consécutifs |

---

## Nettoyer la base de données

Pour vider complètement la base et repartir de zéro, supprime le dossier `chroma_db_data/` :

```bash
# Windows
rmdir /s /q chroma_db_data

# Mac / Linux
rm -rf chroma_db_data
```

---

## Limites actuelles

**Compréhension sémantique partielle.** Le modèle d'embeddings `all-MiniLM-L6-v2` est léger et fait des approximations. Si le document parle de "chat" et que tu demandes "félin", le système peut ne pas faire le lien. Les synonymes rares et les périphrases complexes peuvent ne pas être retrouvés.

**Accumulation des documents sans gestion.** Quand plusieurs PDFs sont uploadés, tous leurs chunks s'accumulent dans ChromaDB. Le retriever cherche dans tous les documents simultanément, ce qui peut mélanger des informations de sources différentes sans que ce soit toujours clair.

**Découpage mécanique du texte.** Le découpage se fait par taille fixe de caractères. Les tableaux, les listes, les titres et les formules sont traités comme du texte brut — des informations structurées peuvent se retrouver découpées en morceaux qui n'ont plus de sens séparément.

**Pas de mémoire conversationnelle.** Chaque question est traitée indépendamment. Il n'est pas possible de poser des questions de suivi en référençant la question précédente ("et en 2022 ?", "parle-moi plus de ça").

**PDFs scannés non supportés.** Si le PDF est une image scannée sans couche texte, PyPDFLoader ne peut pas extraire le contenu. Seuls les PDFs avec du texte natif fonctionnent.

**Interface minimaliste.** Streamlit est un outil de prototypage — le design est générique, pas d'historique de conversation affiché, pas d'adaptation mobile.

---

## Améliorations envisagées

**Recherche hybride (Dense + BM25).** Combiner la recherche vectorielle actuelle avec une recherche BM25 par mots-clés, puis fusionner les résultats avec Reciprocal Rank Fusion. Résoudrait en grande partie le problème des synonymes.

**Mémoire conversationnelle.** Intégrer `ConversationBufferMemory` de LangChain pour conserver l'historique des échanges et permettre des questions de suivi naturelles.

**Gestion multi-documents.** Créer une collection ChromaDB distincte par document avec possibilité de choisir sur quel document interroger, ou de comparer plusieurs documents simultanément.

**Découpage sémantique.** Remplacer `RecursiveCharacterTextSplitter` par `SemanticChunker` qui regroupe les phrases par cohérence sémantique plutôt que par taille fixe.

**Évaluation automatique avec Ragas.** Mesurer objectivement la qualité du système — fidélité des réponses, pertinence des chunks récupérés, qualité globale — pour quantifier l'impact de chaque amélioration.

**Interface web complète.** Remplacer Streamlit par FastAPI + React pour une interface sur mesure : historique de conversation, mise en surbrillance des sources dans le PDF, responsive mobile.

**Support OCR pour PDFs scannés.** Intégrer Tesseract ou une API de document intelligence pour traiter les PDFs qui sont des images scannées.

---

## Auteur

ISARTI Anas
