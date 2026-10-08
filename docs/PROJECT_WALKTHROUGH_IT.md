# Data-Mining-Streamlit — Come raccontare il progetto

> Una guida italiana per spiegare **perché nasce**, **come funziona** e **quali scelte tecniche contiene** il sistema di raccomandazione di articoli scientifici.
>
> [System Design con diagrammi Mermaid](SYSTEM_DESIGN.md) · [Repository](../README.md)

## 1. Il problema: trovare articoli davvero correlati

Cercare pubblicazioni scientifiche per parole chiave o titolo non permette sempre di capire quali lavori siano rilevanti per una specifica ricerca.

Due articoli possono essere collegati attraverso il **grafo delle citazioni** anche quando i titoli sono diversi. Ma guardare soltanto alle connessioni tra articoli non basta: possono interessare anche la categoria scientifica, gli autori, il numero di citazioni e i concetti trattati.

La domanda che guida il progetto è quindi:

**Possiamo combinare informazioni apprese da un grafo di citazioni e metadati bibliografici per proporre articoli pertinenti in modo interattivo e personalizzabile?**

Qui è utile mettere in evidenza il collegamento fra **Data Mining, Graph Machine Learning e Recommendation Systems**: non è soltanto una dashboard Streamlit.

## 2. La soluzione, in modo semplice

Il progetto contiene due momenti distinti.

**Prima si prepara la conoscenza offline.** Il lavoro sperimentale riguarda il dataset di articoli e citazioni OGBN-ArXiv, l'analisi del grafo, diversi approcci di Machine Learning e GNN, la generazione di embedding per i nodi e l'arricchimento dei metadati tramite OpenAlex.

**Poi si utilizzano gli embedding online.** L'app Streamlit carica una tabella con i metadati e un tensore di embedding precomputati. Quando l'utente seleziona un articolo, ne recupera il vettore e calcola la similarità con gli altri articoli presenti nell'indice.

Il ranking viene quindi modificato con alcuni segnali bibliografici e filtrato secondo le preferenze espresse nell'interfaccia.

### Schema essenziale

```text
           FASE OFFLINE                      FASE INTERATTIVA
    Grafo delle citazioni                      Titolo paper
              |                                      |
    Esperimenti di Graph ML                   Exact / fuzzy match
              |                                      |
        Node embeddings ----------------> Similarità coseno
                                                     |
    Metadati OpenAlex ------------------> Bonus e PageRank
                                                     |
                                            Filtri e Top-K
                                                     |
                                      Streamlit + export CSV
```

**Punto fondamentale:** durante una ricerca nell'app non viene addestrata né eseguita nuovamente una GNN. Si interrogano gli **embedding già salvati**. Questa scelta consente una fase interattiva molto più leggera.

## 3. Perché un grafo di citazioni?

Un articolo è un nodo; una citazione rappresenta una relazione tra pubblicazioni. Questo consente di studiare il contesto di ogni paper considerando non soltanto il contenuto testuale, ma anche la posizione nella rete scientifica.

Un approccio basato su graph learning può produrre una rappresentazione vettoriale dei nodi che cattura informazioni sulle loro relazioni.

**Motivazione:** ricavare una forma di vicinanza tra articoli che non dipenda soltanto dalla presenza delle stesse parole nel titolo.

**Limite:** essere vicini nel grafo non garantisce rilevanza tematica e la struttura delle citazioni può favorire articoli più noti o centrali.

## 4. Dove entrano in gioco le GNN?

Il README documenta esperimenti con GCN, GraphSAGE, GAT, feature strutturali e modelli classici; il notebook è conservato nella cartella `notebooks/` e sono presenti alcune figure storiche.

Per l'applicazione, invece, il file effettivamente atteso è `deepgcn_node_embeddings.pt`: il nome e il testo dell'interfaccia identificano embedding precomputati di tipo DeepGCN.

È corretto spiegare che:

- **offline** vengono esplorate rappresentazioni e metodi di apprendimento sul grafo;
- **online** il sistema consulta la rappresentazione appresa, invece di rieseguire i layer della rete;
- il README e le figure descrivono un lavoro sperimentale più ampio della sola app Streamlit.

Non presenterei tutte le famiglie GNN come un ensemble attivo in produzione: il codice Streamlit non ne contiene uno.

## 5. La logica di raccomandazione effettiva

Una volta individuato un articolo tramite titolo esatto o fuzzy matching con RapidFuzz, il motore esegue quattro passaggi:

1. Recupera l'identificativo `node_idx` e l'embedding dell'articolo selezionato.
2. Calcola la **similarità coseno** con gli embedding degli altri articoli.
3. Aggiunge un bonus se la categoria coincide, un bonus per gli autori condivisi e un contributo proporzionale al **PageRank**.
4. Esclude l'articolo di partenza, applica i filtri scelti e restituisce i primi `K` risultati.

La formula implementata è:

```text
Score = Cosine Similarity
      + Bonus Categoria (se coincide)
      + Bonus Autori (se presenti in comune)
      + Peso PageRank × PageRank
```

I pesi dei bonus e del PageRank sono modificabili dall'utente tramite slider.

**Dettaglio da sapere al colloquio:** le citazioni minime e i concetti scientifici sono **filtri**, non termini additivi della funzione di score. L'anno di pubblicazione viene visualizzato, ma non entra nella formula attuale.

### Un esempio intuitivo

Se cerco un articolo sulle reti neurali per grafi, due candidati potrebbero avere embedding simili. Il sistema può favorire quello nella stessa categoria o con un autore condiviso, e può assegnare peso maggiore al PageRank. Posso poi restringere l'insieme ai paper che rispettano determinate condizioni.

**Questo è un esempio concettuale**, non il risultato di una ricerca realmente eseguita.

## 6. Perché Streamlit e OpenAlex?

**Streamlit** permette di trasformare un esperimento di Data Mining in un'interfaccia esplorabile: ricerca per titolo, pesi modificabili, filtri, schede dei paper, preferiti di sessione ed esportazione CSV.

**OpenAlex** arricchisce le pubblicazioni con metadati quali titolo, autori, affiliazioni, concetti, citazioni e data di pubblicazione. Nel progetto è presente uno script separato per questa fase: l'app non interroga OpenAlex a ogni ricerca.

È presente anche un'integrazione con **GitHub OAuth** per mostrare l'identità dell'utente. Non la descriverei come un sistema di autorizzazione completo, perché nel codice attuale la ricerca non è chiaramente bloccata per chi non è autenticato. I preferiti non sono persistenti su un database utenti: sono conservati nella sessione Streamlit.

## 7. Presentazione da circa 30 secondi

> Ho sviluppato un sistema di raccomandazione di articoli scientifici che combina Graph Machine Learning e metadati bibliografici. L'idea è sfruttare il grafo delle citazioni per rappresentare gli articoli tramite embedding, invece di basarsi soltanto sulla ricerca testuale. Nell'app Streamlit, un articolo selezionato viene confrontato con gli altri usando similarità coseno; i risultati sono poi riordinati attraverso categoria, autori e PageRank e filtrati secondo le preferenze dell'utente. Il progetto unisce quindi una parte sperimentale offline e un'interfaccia di raccomandazione interattiva.

## 8. Presentazione tecnica da circa 90 secondi

> Il progetto nasce dal problema di aiutare un ricercatore a individuare articoli scientifici correlati sfruttando non solo titoli e parole chiave, ma anche la struttura delle citazioni.
>
> Come base ho utilizzato il contesto del grafo OGBN-ArXiv, in cui i paper sono nodi collegati dalle citazioni. Il repository raccoglie analisi di Data Mining, esperimenti con modelli classici e architetture GNN, e rappresentazioni vettoriali apprese dei nodi.
>
> Dal punto di vista architetturale ho separato la fase offline di analisi e preparazione dalla fase di raccomandazione. L'applicazione Streamlit non riaddestra una GNN quando arriva una richiesta: carica embedding già salvati e una tabella arricchita con metadati bibliografici.
>
> L'utente sceglie un articolo, anche tramite fuzzy matching del titolo. Il sistema recupera il relativo embedding, calcola la similarità coseno rispetto agli altri paper e costruisce un punteggio che aggiunge contributi configurabili per categoria, autori in comune e PageRank. Successivamente applica filtri per categoria, citazioni, autori e concetti, mostrando una classifica Top-K esportabile in CSV.
>
> Il punto interessante è la combinazione di una rappresentazione strutturale appresa con segnali più interpretabili e controllabili. Il principale miglioramento futuro sarebbe una valutazione quantitativa rigorosa della qualità dei ranking, insieme alla riproducibilità degli embedding e alla gestione sicura e robusta dell'applicazione.

## 9. Domande che potrebbero farti

| Domanda | Risposta tecnicamente corretta |
| --- | --- |
| **Perché non usare soltanto una ricerca per titolo?** | Perché la rete delle citazioni contiene informazioni relazionali non sempre espresse dalle stesse parole. |
| **Che cosa rappresenta un node embedding?** | Un vettore appreso per un nodo del grafo, utilizzabile per confrontare rappresentazioni dei paper. La semantica precisa dipende dal training. |
| **La GNN viene eseguita a ogni ricerca?** | No. L'app carica il file degli embedding precomputati e confronta i vettori. |
| **Come calcoli la similarità?** | Normalizzazione del tensore salvato e similarità coseno tra embedding selezionato e quelli degli altri paper. |
| **Qual è il ruolo di PageRank?** | È un segnale strutturale di centralità usato nel ranking; può premiare paper influenti ma introdurre bias verso i più noti. |
| **È un learning-to-rank model?** | No. Il ranking online è una combinazione additiva di similarity score e bonus configurabili. |
| **Citazioni e concetti entrano nel punteggio?** | Non direttamente: sono filtri. La funzione di score contiene similarity, categoria, autori e PageRank. |
| **A cosa serve RapidFuzz?** | A trovare un titolo vicino a quello digitato quando non esiste un match esatto. |
| **A cosa serve OpenAlex?** | Ad arricchire gli articoli con metadati bibliografici in una fase separata. |
| **Come garantisci che l'embedding appartenga al paper giusto?** | Il codice usa `node_idx`; è essenziale che CSV e tensore provengano dallo stesso ordinamento del grafo. Oggi manca un controllo esplicito tramite manifest. |
| **I preferiti vengono salvati nell'account GitHub?** | No. Sono gestiti tramite session state Streamlit. OAuth serve all'identità mostrata nella demo. |
| **Hai misurato Precision@K o NDCG?** | Non rivendico tali metriche per l'app senza un protocollo di rilevanza verificato e una nuova valutazione. |
| **Qual è il miglioramento più importante?** | Riprodurre gli artefatti offline, validare la corrispondenza `node_idx`–embedding e confrontare embedding-only vs metadata-aware con metriche ranking. |

## 10. I limiti che devi conoscere

**Il repository non contiene i due artefatti necessari per far partire la raccomandazione completa:** `data/df_final3.csv` e `checkpoints/deepgcn_node_embeddings.pt`. Sono stati esclusi deliberatamente, ma bisogna fornirli separatamente. Questo significa che la documentazione non può promettere una demo immediatamente riproducibile da un semplice clone.

Il codice contiene inoltre aspetti che andrebbero affrontati prima di un rilascio pubblico: controllo degli input e del mapping dei nodi, bilanciamento dei pesi del ranking, rendering HTML dei metadati, OAuth e affidabilità dei salvataggi incrementali dello script OpenAlex.

Anche le figure storiche di confronto modelli non dimostrano da sole la qualità delle raccomandazioni. **Una buona accuratezza in un esperimento di classificazione di nodi non coincide necessariamente con una buona Precision@K per gli utenti.**

## 11. La frase con cui chiudere

> **Ho usato le rappresentazioni apprese da un grafo di citazioni come base per la similarità tra paper, poi ho aggiunto metadati interpretabili e controlli interattivi per trasformare il risultato di Graph Learning in uno strumento di raccomandazione esplorabile.**
