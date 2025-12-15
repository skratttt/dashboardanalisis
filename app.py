import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

# ==========================================
# 1. CONFIGURACIÓN ESTILO
# ==========================================
st.set_page_config(page_title="Analisis De Datos FabricaChile", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1 {color: #0e1117;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
    .stTabs [aria-selected="true"] {background-color: #ffffff; border-bottom: 2px solid #000000;}
</style>
""", unsafe_allow_html=True)

st.title("Analisis De Datos FabricaChile")
st.markdown("Plataforma para análisis.")

# ==========================================
# 2. FUNCIONES DE CARGA (CACHÉ)
# ==========================================

@st.cache_resource
def cargar_spacy():
    try: return spacy.load("es_core_news_sm")
    except: return None

@st.cache_resource
def cargar_modelo_sentimiento():
    nombre = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    return BertTokenizer.from_pretrained(nombre), BertForSequenceClassification.from_pretrained(nombre)

@st.cache_resource
def cargar_modelo_embeddings():
    # Modelo ligero para búsqueda semántica
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Stopwords
STOPWORDS_ES = list(cargar_spacy().Defaults.stop_words) + ['chile', 'chileno', 'chilena', 'noticia', 'hoy', 'ayer', 'tras', 'hacia', 'según', 'dice', 'ser', 'parte', 'nuevo']

# ==========================================
# 3. BARRA LATERAL
# ==========================================
with st.sidebar:
    st.header("Carga de Datos")
    archivo = st.file_uploader("Sube Dataset (CSV)", type=["csv"])
    
    col_texto = None
    col_medio = None

    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.success(f"Registros cargados: {len(df)}")
            cols = df.columns.tolist()
            idx_txt = cols.index('titulo') if 'titulo' in cols else 0
            col_texto = st.selectbox("Columna Texto", cols, index=idx_txt)
            idx_med = cols.index('medio') if 'medio' in cols else 0
            col_medio = st.selectbox("Columna Medio (Opcional)", ["No disponible"] + cols, index=idx_med + 1)
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 4. LÓGICA PRINCIPAL
# ==========================================
if archivo and col_texto:
    df = df.dropna(subset=[col_texto])
    df[col_texto] = df[col_texto].astype(str)

    # Definir pestañas (Sin Emojis)
    tabs = st.tabs(["Resumen", "Sentimiento", "Lenguaje", "Temas y Anomalias", "Buscador Semantico", "Analisis de Redes"])

    # --- TAB 1: RESUMEN ---
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.metric("Volumen Total", len(df))
        if col_medio != "No disponible":
            c2.metric("Medios Monitoreados", df[col_medio].nunique())
            conteo = df[col_medio].value_counts().reset_index()
            conteo.columns = ['Medio', 'Count']
            fig = px.bar(conteo.head(15), x='Count', y='Medio', orientation='h', title="Distribución por Medio")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: SENTIMIENTO ---
    with tabs[1]:
        if st.button("Ejecutar Analisis de Sentimiento", type="primary"):
            with st.spinner("Procesando..."):
                tok, mod = cargar_modelo_sentimiento()
                def predict(txts):
                    inp = tok(txts, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cpu")
                    with torch.no_grad(): out = mod(**inp)
                    return ["Positivo" if p[1]>p[0] else "Negativo" for p in torch.softmax(out.logits, dim=1)]
                
                res = []
                bs = 32
                prog = st.progress(0)
                txts = df[col_texto].tolist()
                for i in range(0, len(txts), bs):
                    res.extend(predict(txts[i:i+bs]))
                    prog.progress(min((i+bs)/len(txts), 1.0))
                df['Sentimiento'] = res
                
                c1, c2 = st.columns([1, 2])
                with c1: st.plotly_chart(px.pie(df, names='Sentimiento', title="Global", color='Sentimiento', color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'}), use_container_width=True)
                with c2:
                    if col_medio != "No disponible":
                        df_f = df[df[col_medio].isin(df[col_medio].value_counts()[lambda x: x>2].index)]
                        st.plotly_chart(px.imshow(pd.crosstab(df_f[col_medio], df_f['Sentimiento'], normalize='index')*100, text_auto='.0f', aspect="auto", color_continuous_scale="RdBu", origin='lower'), use_container_width=True)

    # --- TAB 3: LENGUAJE ---
    with tabs[2]:
        if st.button("Ejecutar Analisis de Entidades"):
            nlp = cargar_spacy()
            doc = nlp(" ".join(df[col_texto].tolist())[:1000000])
            
            ents = [e.text for e in doc.ents if len(e.text)>2 and e.label_ in ["PER", "ORG", "LOC"]]
            counts = pd.Series(ents).value_counts().head(15).sort_values()
            st.plotly_chart(px.bar(x=counts.values, y=counts.index, orientation='h', title="Top Entidades Detectadas"), use_container_width=True)

    # --- TAB 4: TEMAS Y OUTLIERS ---
    with tabs[3]:
        st.subheader("Modelado de Topicos (Clustering) + Deteccion de Anomalias")
        if st.button("Detectar Temas y Anomalias", type="primary"):
            with st.spinner("Entrenando modelo..."):
                topic_model = BERTopic(language="multilingual", min_topic_size=5)
                topics, probs = topic_model.fit_transform(df[col_texto].tolist())
                df['Tema'] = topics
                
                # SECCIÓN 1: TEMAS PRINCIPALES
                freq = topic_model.get_topic_info()
                freq_clean = freq[freq['Topic'] != -1].head(10)
                freq_clean['Nombre'] = freq_clean['Name'].apply(lambda x: " ".join(x.split("_")[1:4]))
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.plotly_chart(px.bar(freq_clean, x='Nombre', y='Count', title="Temas Principales", color='Count'), use_container_width=True)
                with c2:
                    st.info("Mapa de Distancia Inter-Topicos")
                    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

                st.markdown("---")
                
                # SECCIÓN 2: DETECTOR DE OUTLIERS
                st.subheader("Detector de Anomalias (Outliers)")
                st.markdown("""
                Las noticias clasificadas como **Topico -1** son ruido o **anomalias**: textos que no encajan en ningun patron comun. 
                """)
                
                outliers = df[df['Tema'] == -1]
                st.metric("Cantidad de Anomalias Detectadas", len(outliers))
                
                with st.expander(f"Ver lista de las {len(outliers)} noticias anomalas"):
                    st.dataframe(outliers[[col_texto] + ([col_medio] if col_medio != "No disponible" else [])])

    # --- TAB 5: BUSCADOR SEMÁNTICO ---
    with tabs[4]:
        st.subheader("Buscador Semantico Neural")
        st.markdown("Busqueda por contexto y significado.")
        
        query = st.text_input("Ingrese consulta:", placeholder="Ej: Crisis de seguridad, corrupcion, innovacion tecnologica...")
        top_k = st.slider("Resultados", 3, 20, 5)

        if query:
            with st.spinner("Calculando similitud..."):
                model_emb = cargar_modelo_embeddings()
                # Encoding
                corpus_emb = model_emb.encode(df[col_texto].tolist(), convert_to_tensor=True)
                query_emb = model_emb.encode(query, convert_to_tensor=True)
                
                # Cosine Similarity
                hits = util.semantic_search(query_emb, corpus_emb, top_k=top_k)[0]
                
                st.markdown("### Resultados Relevantes")
                for hit in hits:
                    idx = hit['corpus_id']
                    score = hit['score']
                    txt = df.iloc[idx][col_texto]
                    src = df.iloc[idx][col_medio] if col_medio != "No disponible" else "N/A"
                    
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #333;">
                        <small><b>Relevancia: {score:.2f}</b> | {src}</small><br>
                        {txt}
                    </div>
                    """, unsafe_allow_html=True)

    # --- TAB 6: ANÁLISIS DE REDES ---
    with tabs[5]:
        st.subheader("Grafo de Conexiones (Network Analysis)")
        st.markdown("Visualizacion de relaciones entre entidades (Personas y Organizaciones).")

        if st.button("Generar Red de Conexiones"):
            with st.spinner("Construyendo grafo..."):
                nlp = cargar_spacy()
                
                # 1. Extraer entidades
                docs = list(nlp.pipe(df[col_texto].head(1000).astype(str), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]))
                
                entity_list = []
                for doc in docs:
                    ents = sorted(list(set([e.text for e in doc.ents if e.label_ in ["PER", "ORG"] and len(e.text) > 3])))
                    if len(ents) > 1:
                        entity_list.append(ents)

                # 2. Construir Grafo
                G = nx.Graph()
                co_occurrences = {}
                node_counts = {}

                for ents in entity_list:
                    for i in range(len(ents)):
                        node_counts[ents[i]] = node_counts.get(ents[i], 0) + 1
                        for j in range(i + 1, len(ents)):
                            pair = tuple(sorted((ents[i], ents[j])))
                            co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

                # Filtro Top 40
                top_nodes = sorted(node_counts, key=node_counts.get, reverse=True)[:40]
                
                for node in top_nodes:
                    G.add_node(node, size=node_counts[node]*2, title=f"Menciones: {node_counts[node]}", group=1)
                
                for (source, target), weight in co_occurrences.items():
                    if source in top_nodes and target in top_nodes:
                        G.add_edge(source, target, value=weight, title=f"Co-ocurrencias: {weight}")

                # 3. Visualizar
                if len(G.nodes) > 0:
                    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                    net.from_nx(G)
                    net.force_atlas_2based()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                        net.save_graph(tmp.name)
                        tmp.seek(0)
                        html_bytes = tmp.read()
                    
                    st.success(f"Grafo generado: {len(G.nodes)} nodos.")
                    components.html(html_bytes.decode(), height=600, scrolling=True)
                    os.unlink(tmp.name)
                else:
                    st.warning("No hay suficientes datos para generar el grafo.")

else:
    st.info("Sube tu archivo CSV para comenzar.")
