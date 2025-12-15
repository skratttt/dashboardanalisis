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


st.set_page_config(page_title="Analizador Semantico", layout="wide")


st.markdown("""
<style>
    /* 1. Ajuste general del contenedor */
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {font-family: 'Sans-serif'; color: #202124;}
    

    
    
    div.stButton > button[kind="primary"] {
        background-color: #1A73E8; /* Azul Google */
        color: white;
        border: none;
        border-radius: 24px; /* Borde muy redondeado (Pastilla) */
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #174EA6; /* Azul más oscuro al pasar mouse */
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
        transform: translateY(-2px);
    }

    
    div.stButton > button[kind="secondary"] {
        background-color: white;
        color: #1A73E8;
        border: 1px solid #dadce0;
        border-radius: 24px;
        font-weight: 600;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #F8F9FA;
        border-color: #1A73E8;
        color: #174EA6;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 20px;
        gap: 1px;
        padding: 0px 20px;
        color: #5f6368;
        border: 1px solid #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe;
        color: #1A73E8;
        border: 1px solid #1A73E8;
    }
</style>
""", unsafe_allow_html=True)

st.title("Analizador Semantico")
st.markdown("Suite para análisis de texto, detección de patrones y redes.")

#cache

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
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


with st.sidebar:
    st.header("Configuracion del Dataset")
    archivo = st.file_uploader("1. Subir Archivo (CSV)", type=["csv"])
    
    col_texto = None
    col_cat = None
    custom_stopwords = []

    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.success(f"Registros cargados: {len(df)}")
            cols = df.columns.tolist()
            
            idx_txt = 0
            for possible in ['texto', 'text', 'comentario', 'mensaje', 'descripcion', 'titulo']:
                if possible in cols:
                    idx_txt = cols.index(possible)
                    break
            col_texto = st.selectbox("2. Columna de TEXTO a analizar", cols, index=idx_txt)
            
            st.info("Opcional: Selecciona una columna para agrupar.")
            col_cat = st.selectbox("3. Columna de AGRUPACIÓN", ["No aplicar"] + cols)

            st.header("Filtros de Texto")
            stopwords_input = st.text_area("Palabras a ignorar (separadas por coma)", "el, la, los, un, una, de, del, y, o, que, por, para, con, se, su")
            custom_stopwords = [x.strip() for x in stopwords_input.split(",")]
            
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

def get_stopwords():
    base = list(cargar_spacy().Defaults.stop_words) if cargar_spacy() else []
    return base + custom_stopwords

#maincode
if archivo and col_texto:
    df = df.dropna(subset=[col_texto])
    df[col_texto] = df[col_texto].astype(str)

    tabs = st.tabs(["Resumen Global", "Analisis de Sentimiento", "Frecuencia y Entidades", "Clusterizacion (Temas)", "Busqueda Inteligente", "Analisis de Redes"])

#1
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.metric("Total de Documentos", len(df))
        
        if col_cat != "No aplicar":
            c2.metric(f"Total de {col_cat} únicos", df[col_cat].nunique())
            conteo = df[col_cat].value_counts().reset_index()
            conteo.columns = ['Categoria', 'Count']
            
            fig = px.bar(conteo.head(20), x='Count', y='Categoria', orientation='h', 
                         title=f"Distribución por {col_cat}", text_auto=True)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona una columna de agrupación en la barra lateral para ver estadísticas comparativas.")
#2
    with tabs[1]:
        st.subheader("Clasificacion Automatica de Tono")
        if st.button("Ejecutar Modelo de Sentimiento", type="primary"):
            with st.spinner("Procesando textos con IA..."):
                tok, mod = cargar_modelo_sentimiento()
                
                def predict(txts):
                    inp = tok(txts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cpu")
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
                with c1: 
                    st.plotly_chart(px.pie(df, names='Sentimiento', title="Distribucion Global", 
                                    color='Sentimiento', color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'}), use_container_width=True)
                with c2:
                    if col_cat != "No aplicar":
                        val_counts = df[col_cat].value_counts()
                        top_cats = val_counts[val_counts > 1].index
                        df_f = df[df[col_cat].isin(top_cats)]
                        
                        st.subheader(f"Sentimiento por {col_cat}")
                        st.plotly_chart(px.imshow(pd.crosstab(df_f[col_cat], df_f['Sentimiento'], normalize='index')*100, 
                                        text_auto='.0f', aspect="auto", color_continuous_scale="RdBu", origin='lower',
                                        labels=dict(x="Sentimiento", y=col_cat, color="%")), use_container_width=True)

    #3
    with tabs[2]:
        
        if st.button("Ejecutar Analisis de Frecuencia"):
            nlp = cargar_spacy()
            full_text = " ".join(df[col_texto].tolist())[:1000000]
            
            st.subheader("Mapa de Conceptos Clave")
            wc = WordCloud(width=800, height=300, background_color='white', stopwords=get_stopwords(), colormap='viridis').generate(full_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            st.subheader("Entidades Detectadas")
            doc = nlp(full_text)
            ents = [e.text for e in doc.ents if len(e.text)>2 and e.label_ in ["PER", "ORG", "LOC"]]
            if ents:
                counts = pd.Series(ents).value_counts().head(15).sort_values()
                st.plotly_chart(px.bar(x=counts.values, y=counts.index, orientation='h', title="Top Entidades"), use_container_width=True)
            else:
                st.warning("No se detectaron entidades nombradas suficientes.")

  #4
    with tabs[3]:
        st.subheader("Deteccion de Patrones")
        if st.button("Detectar Clusters y Outliers", type="primary"):
            with st.spinner("Entrenando modelo de clustering..."):
                topic_model = BERTopic(language="multilingual", min_topic_size=5)
                topics, probs = topic_model.fit_transform(df[col_texto].tolist())
                df['Cluster_ID'] = topics
                
                freq = topic_model.get_topic_info()
                freq_clean = freq[freq['Topic'] != -1].head(10)
                freq_clean['Nombre'] = freq_clean['Name'].apply(lambda x: " ".join(x.split("_")[1:4]))
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.plotly_chart(px.bar(freq_clean, x='Nombre', y='Count', title="Grupos Principales", color='Count'), use_container_width=True)
                with c2:
                    st.info("Mapa de Similitud")
                    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

                st.markdown("---")
                st.subheader("Registros Anomalos (Outliers)")
                st.markdown("Textos que no encajan en ningun cluster identificado.")
                outliers = df[df['Cluster_ID'] == -1]
                st.metric("Cantidad de Outliers", len(outliers))
                with st.expander("Ver datos anomalos"):
                    cols_to_show = [col_texto] + ([col_cat] if col_cat != "No aplicar" else [])
                    st.dataframe(outliers[cols_to_show])
#5
    with tabs[4]:
        st.subheader("Motor de Busqueda Semantica")
        
        query = st.text_input("Consulta:", placeholder="Ej: Problemas de infraestructura, retrasos...")
        
        if query:
            with st.spinner("Calculando similitud..."):
                model_emb = cargar_modelo_embeddings()
                corpus_emb = model_emb.encode(df[col_texto].tolist(), convert_to_tensor=True)
                query_emb = model_emb.encode(query, convert_to_tensor=True)
                
                hits = util.semantic_search(query_emb, corpus_emb, top_k=5)[0]
                
                st.markdown("### Resultados")
                for hit in hits:
                    idx = hit['corpus_id']
                    score = hit['score']
                    txt = df.iloc[idx][col_texto]
                    cat_val = df.iloc[idx][col_cat] if col_cat != "No aplicar" else "-"
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #e0e0e0;">
                        <small style="color: #1A73E8;"><b>Relevancia: {score:.2f}</b> | {cat_val}</small><br>
                        <span style="color: #202124;">{txt}</span>
                    </div>
                    """, unsafe_allow_html=True)

   #6
    with tabs[5]:
        st.subheader("Relaciones entre Entidades")
        st.markdown("Grafo de co-ocurrencia: Entidades que aparecen juntas en el mismo texto.")

        if st.button("Generar Grafo", type="primary"):
            with st.spinner("Procesando red..."):
                nlp = cargar_spacy()
                sample_df = df.head(1000)
                docs = list(nlp.pipe(sample_df[col_texto].astype(str), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]))
                
                entity_list = []
                for doc in docs:
                    ents = sorted(list(set([e.text for e in doc.ents if e.label_ in ["PER", "ORG"] and len(e.text) > 3])))
                    if len(ents) > 1:
                        entity_list.append(ents)

                G = nx.Graph()
                co_occurrences = {}
                node_counts = {}

                for ents in entity_list:
                    for i in range(len(ents)):
                        node_counts[ents[i]] = node_counts.get(ents[i], 0) + 1
                        for j in range(i + 1, len(ents)):
                            pair = tuple(sorted((ents[i], ents[j])))
                            co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

                top_nodes = sorted(node_counts, key=node_counts.get, reverse=True)[:40]
                
                for node in top_nodes:
                    G.add_node(node, size=node_counts[node]*2, title=f"Frecuencia: {node_counts[node]}")
                
                for (source, target), weight in co_occurrences.items():
                    if source in top_nodes and target in top_nodes:
                        G.add_edge(source, target, value=weight, title=f"Co-ocurrencias: {weight}")

                if len(G.nodes) > 0:
                    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                    net.from_nx(G)
                    net.force_atlas_2based()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                        net.save_graph(tmp.name)
                        tmp.seek(0)
                        html_bytes = tmp.read()
                    
                    components.html(html_bytes.decode(), height=600, scrolling=True)
                    os.unlink(tmp.name)
                else:
                    st.warning("No se encontraron suficientes relaciones para generar el grafo.")

else:
    st.info("Sube un archivo CSV para comenzar. Asegurate de que tenga una columna de texto.")
