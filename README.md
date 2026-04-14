# KnowRAG
Il progetto KnowRAG è un sistema avanzato di Retrieval-Augmented Generation (RAG) progettato per ottimizzare il recupero e la generazione di informazioni da un corpus di documenti locale. L'architettura si distingue per l'uso di un retrieval ibrido, un sistema di reranking basato su Cross-Encoder e una pipeline di valutazione delle performance integrata.

# 1. Architettura del Sistema

Il sistema è diviso in due macro-fasi: Ingestion (preparazione dei dati) e Query (recupero e generazione).

## A. Pipeline di Ingestion (IngestionPipeline)

Questa fase trasforma i documenti grezzi in una base di conoscenza interrogabile.

1. Document Loading: Caricamento di file da directory locali.

2. Recursive Chunking: Suddivisione dei testi in frammenti (chunk) di dimensioni fisse (es. 1000 caratteri) con un overlap (es. 200 caratteri) per mantenere il contesto semantico tra i chunk adiacenti.

3. Embedding Generation: Utilizzo di modelli Transformer (es. all-MiniLM-L6-v2) tramite la libreria sentence-transformers per convertire il testo in vettori densi (embedding).

4. Vector Indexing: Memorizzazione dei chunk e dei relativi embedding in ChromaDB, un database vettoriale che permette ricerche rapide per similarità coseno.

## B. Pipeline di Query (QueryPipeline)

L'orchestratore che gestisce il flusso dalla domanda dell'utente alla risposta finale.

1. Preprocessing: Pulizia e tokenizzazione della query.

2. Hybrid Retrieval: Combina due approcci per massimizzare la precisione:

        • Dense Retrieval: Ricerca vettoriale su ChromaDB (cattura il significato semantico).

        • Sparse Retrieval (BM25): Ricerca basata su parole chiave (cattura termini specifici, nomi propri o codici).

        • Reciprocal Rank Fusion (RRF): Algoritmo che combina le classifiche dei due metodi per ottenere una lista di candidati ottimizzata.

3. Reranking: I candidati vengono ri-valutati da un modello Cross-Encoder (ms-marco-MiniLM-L-6-v2). A differenza degli embedding (Bi-Encoder), il Cross-Encoder analizza la coppia (query, chunk) simultaneamente, offrendo una precisione molto più elevata nel determinare la rilevanza.

4. Generation: I chunk più rilevanti vengono passati come contesto a un LLM (tramite Ollama o Anthropic) per generare una risposta accurata e fondata sui documenti (grounded).

## 2. Dettagli Tecnici e Scelte Progettuali

### Il Modello di Embedding

È stato scelto all-MiniLM-L6-v2 per il bilanciamento tra velocità e accuratezza. Il sistema supporta modelli multilingua come paraphrase-multilingual-MiniLM-L12-v2 per gestire correttamente l'italiano. Gli embedding vengono normalizzati per permettere l'uso della similarità coseno come metrica di distanza.

### Retrieval Ibrido e RRF

Il retrieval ibrido risolve i limiti della ricerca semantica pura (che a volte ignora parole chiave esatte) e della ricerca testuale pura (che ignora i sinonimi). L'uso di RRF permette di fondere i risultati senza dover normalizzare i punteggi (score) eterogenei di BM25 e ChromaDB.

### Strategia di Reranking

Il reranking è il "collo di bottiglia" qualitativo: riduce il rumore passando all'LLM solo le informazioni strettamente necessarie (Top-N), riducendo il rischio di allucinazioni e ottimizzando l'uso dei token.


## 3. Valutazione delle Performance (Evaluation)

Per garantire l'affidabilità del sistema, è stata implementata una suite di test basata sulla classe BenchmarkMetrics:

Metrica     ||     Descrizione Tecnica

Insert Time     ||      Tempo totale di esecuzione della IngestionPipeline (caricamento + embedding + indexing).

Search Time     ||      Latenza media della fase di retrieval (Hybrid + Reranking).

Accuracy Top-1      ||      Frequenza con cui il chunk più rilevante (ground truth) appare in prima posizione.

Accuracy Top-5      ||      Frequenza con cui il chunk corretto appare tra i primi 5 risultati.

Total Documents     ||      Dimensione del corpus indicizzato, utile per valutare la scalabilità.


## 4. Punti di Forza

1. Modularità: Ogni componente (Embedder, Indexer, Retriever, Generator) è disaccoppiato, permettendo la sostituzione di un modello o di un database senza riscrivere il sistema.

2. Efficienza: L'uso di modelli "Small Language Models" per embedding e reranking permette l'esecuzione su hardware consumer o server leggeri.

3. Robustezza: Il retrieval ibrido garantisce che il sistema funzioni bene sia con query concettuali che con ricerche di termini specifici.

4. Approccio Data-Driven: La presenza di un modulo di valutazione dimostra un approccio ingegneristico volto all'ottimizzazione continua basata sui dati.


## Avvio del server Ollama

Questo progetto utilizza **Ollama** per l'esecuzione locale dei modelli LLM.  
È necessario che il server Ollama sia attivo prima di avviare l'applicativo.

---

### 1. Installazione

Scaricare e installare Ollama da:

https://ollama.com

Oppure via terminale (Linux/Mac):

 bash 
curl -fsSL https://ollama.com/install.sh | sh

### 2. Avvio del server

Avviare il server con:

ollama serve

Il server sarà disponibile all'indirizzo:

http://localhost:11434

Nota: se il server è già attivo, il comando restituirà un errore di porta già in uso.

### 3. Download del modello 

Prima di eseguire l'applicativo, assicurarsi di aver scaricato almeno un modello:

ollama pull llama3.2

### 4. Verifica il funzionamento 

Eseguire un test rapido:

ollama run llama3.2

Se il modello risponde correttamente, il sistema è pronto.