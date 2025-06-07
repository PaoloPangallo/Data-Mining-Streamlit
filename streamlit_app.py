import ast
import re
import string

import numpy as np
import pandas as pd
import streamlit as st
import torch
from authlib.integrations.requests_client import OAuth2Session
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
from streamlit.runtime.scriptrunner import RerunException



# ====================================
st.set_page_config(page_title="Raccomandazioni Paper", layout="wide")
st.sidebar.subheader("🛠 DEBUG OAUTH")
st.sidebar.write("🔁 query_params", st.query_params)
st.sidebar.write("📦 session_state", dict(st.session_state))

# ✅ INIZIALIZZA SUBITO I SEGRETI
GH            = st.secrets["github"]
CLIENT_ID     = GH["client_id"]
CLIENT_SECRET = GH["client_secret"]
REDIRECT_URI  = GH["redirect_uri"]
AUTH_URL      = "https://github.com/login/oauth/authorize"
TOKEN_URL     = "https://github.com/login/oauth/access_token"

# ✅ SOLO DOPO puoi usare queste variabili
if "oauth_state" not in st.session_state:
    st.session_state.oauth_state = None
if "auth_url" not in st.session_state:
    st.session_state.auth_url = None

if st.session_state.auth_url is None:
    oauth = OAuth2Session(CLIENT_ID, CLIENT_SECRET, redirect_uri=REDIRECT_URI)
    auth_url, state = oauth.create_authorization_url(AUTH_URL)
    st.session_state.oauth_state = state
    st.session_state.auth_url = auth_url






# ====================================
# === 1) CALLBACK: se arriva il codice da GitHub ===
if "code" in st.query_params and "access_token" not in st.session_state:
    try:
        st.sidebar.write("📤 fetch_token input", {
            "code": st.query_params["code"],
            "state": st.session_state.oauth_state
        })

        oauth = OAuth2Session(CLIENT_ID, CLIENT_SECRET, redirect_uri=REDIRECT_URI, state=st.session_state.oauth_state)
        token = oauth.fetch_token(
            TOKEN_URL,
            code=st.query_params["code"],  # ✅ rimuovi [0]
            client_secret=CLIENT_SECRET
        )
        st.session_state.access_token = token["access_token"]
        st.query_params.clear()

        st.markdown("""
            <meta http-equiv="refresh" content="0; url=/" />
            <script>window.location.replace(window.location.pathname);</script>
        """, unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.error(f"OAuth failed: {e}")

# === 2) GENERA LOGIN se non ho ancora il token ===
elif "access_token" not in st.session_state:
    if not st.session_state.oauth_state or not st.session_state.auth_url:
        oauth = OAuth2Session(CLIENT_ID, CLIENT_SECRET, redirect_uri=REDIRECT_URI)
        auth_url, state = oauth.create_authorization_url(AUTH_URL)
        st.session_state.oauth_state = state
        st.session_state.auth_url = auth_url

    if st.session_state.auth_url:  # ✅ solo se è valido
        st.markdown(f"[🔐 Login con GitHub]({st.session_state.auth_url})", unsafe_allow_html=True)
        st.sidebar.write("DEBUG auth_url:", st.session_state.auth_url)
    else:
        st.warning("⚠️ auth_url non inizializzato correttamente.")



# ====================================
# 6) HEADER con Login / Logout
# ====================================
col1, col2 = st.columns([8,1])
col1.title("📚 Motore di Raccomandazione Articoli OGBN-ArXiv")

if "access_token" in st.session_state:


    oauth_sess = OAuth2Session(
        CLIENT_ID,
        CLIENT_SECRET,
        token={"access_token": st.session_state.access_token}
    )
    user = oauth_sess.get("https://api.github.com/user").json()
    st.sidebar.write("👤 GitHub user info", user)

    try:
        user = oauth_sess.get("https://api.github.com/user").json()
        col2.image(user.get("avatar_url", ""), width=40)
        col2.write(f"👤 {user.get('login')}")

        # ✅ Mostra badge in sidebar
        with st.sidebar:
            st.markdown("### 🔐 Autenticato come:")
            st.image(user.get("avatar_url", ""), width=64)
            st.markdown(f"**[{user.get('login')}]({user.get('html_url')})**", unsafe_allow_html=True)
            st.markdown("---")

    except:
        col2.error("GitHub user fetch failed")


    if col2.button("Logout"):
        for k in ("access_token", "oauth_state"):
            st.session_state.pop(k, None)
        st.query_params.clear()
        st.markdown("""
            <meta http-equiv="refresh" content="0; url=/" />
            <script>window.location.replace(window.location.pathname);</script>
        """, unsafe_allow_html=True)
        st.stop()


else:
    st.write("🔑 Autenticati con GitHub per continuare.")






# ====================================
# 7) SESSION STATE PER APP
# ====================================
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_favorite(idx: int):
    favs = st.session_state.favorites.copy()
    if idx in favs: favs.remove(idx)
    else:          favs.append(idx)
    st.session_state.favorites = favs

def toggle_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

if st.button("🌙 Dark mode" if not st.session_state.dark_mode else "☀️ Light mode"):
    toggle_mode()
if st.session_state.dark_mode:
    st.markdown(
        "<style>body{background:#15202B;color:#D1D5DB} .css-1d391kg{background:#010409}</style>",
        unsafe_allow_html=True
    )

# ====================================
# 8) CARICAMENTO E CACHE DEI DATI
# ====================================
@st.cache_data(show_spinner=False)
def load_embeddings(path="checkpoints/deepgcn_node_embeddings.pt") -> np.ndarray:
    embs = torch.load(path)
    embs = embs / embs.norm(dim=1, keepdim=True)
    return embs.numpy()

@st.cache_data(show_spinner=False)
def load_dataframe(path="df_final3.csv") -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates("Title").reset_index(drop=True)
    # Publication Year
    if "Publication_Date" in df.columns:
        df["Publication_Year"] = (
            pd.to_datetime(df["Publication_Date"], errors="coerce")
              .dt.year.fillna(0).astype(int)
        )
    else:
        df["Publication_Year"] = 0
    # Abstract
    df["Abstract"] = df.get("Abstract","")
    # Concepts_List
    def clean_concepts(c):
        if isinstance(c,str) and c.startswith("[") and c.endswith("]"):
            try: parts = ast.literal_eval(c)
            except: parts = []
        else:
            parts = str(c).split(";")
        seen,out = set(),[]
        for p in parts:
            t = p.strip()
            if len(t)<2: continue
            if not re.search(r"[A-Za-z0-9]",t): continue
            if all(ch in string.punctuation or ch.isspace() for ch in t): continue
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
        return out
    if "Concepts" in df.columns:
        df["Concepts_List"] = df["Concepts"].apply(clean_concepts)
    # Institutions_List
    def clean_insts(c):
        parts = str(c).split(";")
        seen,out = set(),[]
        for p in parts:
            t = p.strip()
            if len(t)<2: continue
            if not re.search(r"[A-Za-z0-9]",t): continue
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
        return out
    if "Institutions" in df.columns:
        df["Institutions_List"] = df["Institutions"].apply(clean_insts)
    return df

embeddings_np = load_embeddings()
df_final      = load_dataframe()
node_idx_list = df_final["node_idx"].tolist()

# Build lookups
base_cols   = ["Authors","categoria","pagerank","node_idx","Publication_Year","Abstract"]
opt_cols    = [c for c in ["Citations","Concepts_List"] if c in df_final.columns]
cols_utili  = base_cols + opt_cols
idx_to_info = df_final.set_index("Title")[cols_utili].to_dict("index")
title_to_idx= dict(zip(df_final["Title"], df_final.index))

# ====================================
# 9) SIDEBAR FILTRI
# ====================================
st.sidebar.header("🔧 Parametri di ricerca")
input_title      = st.sidebar.text_input("Titolo (anche parziale):")
top_k            = st.sidebar.slider("Numero risultati",3,20,5)
category_boost   = st.sidebar.slider("Bonus categoria",0.0,0.5,0.05)
author_boost     = st.sidebar.slider("Bonus autore",0.0,0.5,0.05)
pagerank_weight  = st.sidebar.slider("Peso PageRank",0.0,20.0,10.0)
cards_per_row    = st.sidebar.slider("Card per riga",1,5,3)
max_cit          = int(df_final["Citations"].max()) if "Citations" in df_final.columns else 0
min_citations    = st.sidebar.slider("Citazioni minime",0,max_cit,0)
all_cats         = sorted(df_final["categoria"].unique())
sel_cats         = st.sidebar.multiselect("Categorie",all_cats,default=all_cats)
only_authors     = st.sidebar.checkbox("Solo autori in comune")
all_cons         = sorted({c for sub in df_final.get("Concepts_List",[]) for c in sub})
sel_cons         = st.sidebar.multiselect("🔖 Filtra per concetti", all_cons)

# ====================================
# 10) LOGICA DI RACCOMANDAZIONE
# ====================================
if input_title:
    # match titolo
    if input_title in title_to_idx:
        matched = input_title
        st.success(f"✅ Match esatto: **{matched}**")
    else:
        best = process.extractOne(input_title,list(title_to_idx),score_cutoff=75)
        if not best:
            st.error("❌ Nessun titolo simile.") ; st.stop()
        matched = best[0]
        st.warning(f"🔁 Fuzzy match: **{matched}** ({best[1]:.0f})")

    info     = idx_to_info[matched]
    emb      = embeddings_np[info["node_idx"]].reshape(1,-1)
    sims     = cosine_similarity(emb, embeddings_np[node_idx_list])[0]
    df_final["similarity"] = sims

    auth_in   = set(str(info["Authors"]).split(";"))
    cat_in    = info["categoria"]

    def score_fn(r):
        s = r["similarity"]
        if r["categoria"]==cat_in: s+=category_boost
        if any(a in auth_in for a in str(r["Authors"]).split(";")): s+=author_boost
        return s + pagerank_weight*r["pagerank"]

    df_final["score"] = df_final.apply(score_fn,axis=1)

    filt = df_final[df_final["Title"]!=matched]
    filt = filt[filt["categoria"].isin(sel_cats)]
    filt = filt[filt["Citations"].fillna(0)>=min_citations]
    if only_authors:
        filt = filt[filt["Authors"].apply(lambda a:any(x in auth_in for x in str(a).split(";")))]
    if sel_cons:
        filt = filt[filt["Concepts_List"].apply(lambda L:any(c in L for c in sel_cons))]

    topN = filt.nlargest(top_k,"score").copy()

    # === visualizzazione a schede ===
    st.subheader(f"🃏 Raccomandazioni per: _{matched}_")
    rows = (len(topN)+cards_per_row-1)//cards_per_row
    CARD = """<div style="border:1px solid #CCC;border-radius:8px;padding:12px;margin:6px;box-shadow:2px 2px 8px rgba(0,0,0,0.1);">"""
    for i in range(rows):
        cols = st.columns(cards_per_row)
        for j in range(cards_per_row):
            idx = i*cards_per_row+j
            if idx>=len(topN): break
            row = topN.iloc[idx]
            with cols[j]:
                html = CARD
                q = row["Title"].replace(" ","+").replace("&","and")
                html+=f"<h4><a href='https://www.google.com/search?q={q}' target='_blank'>{row['Title']}</a></h4>"
                auth = str(row["Authors"])
                html+=f"<p><strong>Autori:</strong> {auth[:80]}{'...' if len(auth)>80 else ''}</p>"
                ab = row["Abstract"] or ""
                if ab:
                    html+=f"<p>{ab[:120]}{'...' if len(ab)>120 else ''}</p>"
                    html+=f"<details><summary>Mostra Abstract</summary><p>{ab}</p></details>"
                cit=int(row.get("Citations",0)); yr=row["Publication_Year"]
                html+=f"<p><em>Citations:</em> {cit} • <em>Anno:</em> {yr}</p>"
                tags=row.get("Concepts_List",[])[:5]
                if tags:
                    html+="<p>"+ "".join(f"<code style='margin-right:4px'>{t}</code>" for t in tags) +"</p>"
                bonus="";
                if row["categoria"]==cat_in: bonus+="🎯"
                if any(a in auth_in for a in str(row["Authors"]).split(";")): bonus+="👥"
                new_ic=" 🆕" if row["Publication_Year"]>=2017 else ""
                html+="<hr>"+f"<p>⭐ {row['score']:.3f} {bonus}{new_ic}</p></div>"
                st.markdown(html,unsafe_allow_html=True)

                key=f"fav_{row['node_idx']}"
                if st.button(
                    "❌ Rimuovi dai preferiti" if row["node_idx"] in st.session_state.favorites else "⭐ Salva nei preferiti",
                    key=key
                ):
                    toggle_favorite(row["node_idx"])

    # === download CSV, preferiti, FAQ ===
    st.download_button("📁 Scarica CSV", topN.to_csv(index=False).encode("utf-8"),
                       "raccomandazioni.csv","text/csv")

    if st.session_state.favorites:
        st.sidebar.subheader("⭐ I miei preferiti")
        favs = df_final[df_final["node_idx"].isin(st.session_state.favorites)]
        for _,f in favs.iterrows():
            q=f["Title"].replace(" ","+").replace("&","and")
            st.sidebar.markdown(f"- [{f['Title']}](https://www.google.com/search?q={q})")

    with st.expander("❓ FAQ – Come funziona?"):
        st.markdown("""
**Q1. Embeddings?** DeepGCN sul grafo OGBN-ArXiv, normalizzati.  
**Q2. Punteggio?** cosine similarity + bonus categoria + bonus autori + peso PageRank.  
**Q3. Filtri?** categoria, autori in comune, citazioni minime, concetti.  
**Q4. Preferiti?** salva con ⭐, vedi in sidebar.  
**Q5. Esporta?** usa “Scarica CSV”.
""")
