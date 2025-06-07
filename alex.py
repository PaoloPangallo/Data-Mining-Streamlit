import requests
import pandas as pd
import time
import threading
import os
import logging
from queue import Queue
from datetime import datetime
from tqdm import tqdm

# === CONFIG ===
MAG_IDS_FILE = "mag_ids.csv"
OUTFILE = "openalex_enriched_parallel.csv"
SAVE_EVERY = 50
THREADS = 4
SLEEP_BETWEEN_REQUESTS = 2.5
BACKUP_EVERY_SECONDS = 3600  # ogni 1 ora

# === Logger ===
logging.basicConfig(filename="run_log.txt", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# === Carica ID ===
mag_ids_df = pd.read_csv(MAG_IDS_FILE)
MAG_IDS = mag_ids_df["MAG_ID"].tolist()

# === Carica quelli già fatti ===
if os.path.exists(OUTFILE):
    done_df = pd.read_csv(OUTFILE)
    done_ids = set(done_df["MAG_ID"].tolist())
    logging.info(f"Ripresa: {len(done_ids)} MAG_ID già elaborati")
else:
    done_df = pd.DataFrame()
    done_ids = set()

# === Filtro ===
remaining_ids = [mid for mid in MAG_IDS if mid not in done_ids]
print(f"➡️ Da scaricare: {len(remaining_ids)} MAG_ID")

# === Coda thread-safe ===
q = Queue()
for mid in remaining_ids:
    q.put(mid)

lock = threading.Lock()
results = []
last_backup = time.time()

# === Barra di avanzamento ===
pbar = tqdm(total=len(remaining_ids))


def fetch_info():
    global results, last_backup
    while not q.empty():
        mag_id = q.get()
        url = f"https://api.openalex.org/works/W{mag_id}"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                title = data.get("title", "")
                authors = [a["author"]["display_name"] for a in data.get("authorships", [])]
                institutions = [i["display_name"] for a in data.get("authorships", []) for i in
                                a.get("institutions", [])]
                concepts = [c["display_name"] for c in data.get("concepts", [])]
                citations = data.get("cited_by_count", 0)
                pub_date = data.get("publication_date", "")

                row = {
                    "MAG_ID": mag_id,
                    "Title": title,
                    "Authors": "; ".join(authors),
                    "Institutions": "; ".join(institutions),
                    "Concepts": "; ".join(concepts),
                    "Citations": citations,
                    "Publication_Date": pub_date
                }

                with lock:
                    results.append(row)
                    pbar.update(1)

                    # Salva ogni SAVE_EVERY
                    if len(results) >= SAVE_EVERY:
                        flush_results_to_disk()

                    # Backup ogni ora
                    if time.time() - last_backup >= BACKUP_EVERY_SECONDS:
                        flush_results_to_disk(backup=True)
                        last_backup = time.time()

            else:
                logging.warning(f"Status {r.status_code} per {mag_id}")

        except Exception as e:
            logging.error(f"Errore per {mag_id}: {e}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)
        q.task_done()


def flush_results_to_disk(backup=False):
    """Scrive i risultati accumulati su file"""
    global results
    if not results:
        return
    df_partial = pd.DataFrame(results)
    results = []  # svuota il buffer
    combined = pd.concat([done_df, df_partial])
    combined.drop_duplicates(subset="MAG_ID", inplace=True)

    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_{timestamp}.csv"
        combined.to_csv(backup_file, index=False)
        logging.info(f"Backup salvato: {backup_file}")
    else:
        combined.to_csv(OUTFILE, index=False)
        logging.info(f"Salvataggio normale: {len(combined)} righe totali")


# === Avvio threads ===
threads = []
for _ in range(THREADS):
    t = threading.Thread(target=fetch_info)
    t.start()
    threads.append(t)

# === Attendi completamento ===
for t in threads:
    t.join()

# === Salvataggio finale ===
flush_results_to_disk()
pbar.close()
print("✅ Completato!")
logging.info("Run completata.")

