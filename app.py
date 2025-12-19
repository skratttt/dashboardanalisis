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
# 1. CONFIGURACIÓN ESTILO (GEMINI UI)
# ==========================================
st.set_page_config(page_title="Analisis de texto", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {font-family: 'Sans-serif'; color: #202124;}
    
    /* BOTONES PRIMARIOS */
    div.stButton > button[kind="primary"] {
        background-color: #1A73E8;
        color: white;
        border: none;
        border-radius: 24px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        transition: all 0.3s;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #174EA6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }

    /* BOTONES SECUNDARIOS */
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

    /* PESTAÑAS (Corrección de contraste) */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 20px;
        padding: 0px 20px;
        color: #5f6368; /* Color de texto más oscuro para que se lea bien */
        border: 1px solid #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe;
        color: #1A73E8;
        border: 1px solid #1A73E8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("Analisis de texto")
st.markdown("Suite de analisis de Sentimiento, Entidades, N-Gramas, Temas y Redes.")
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

# Stopwords
def get_stopwords(custom_list):
    base = list(cargar_spacy().Defaults.stop_words) if cargar_spacy() else []
    return base + custom_list


def get_top_ngrams(corpus, n=2, top_k=10, stopwords=[]):
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=['Frase', 'Frecuencia'])

#sidebar
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
            col_texto = st.selectbox("2. Columna de TEXTO", cols, index=idx_txt)
            
            st.info("Opcional: Agrupador (ej: Medio, Fecha)")
            col_cat = st.selectbox("3. Columna de AGRUPACIÓN", ["No aplicar"] + cols)

            st.header("Filtros de Texto")
            stopwords_input = st.text_area("Palabras a ignorar (separadas por coma)", "el, la, los, un, una, de, del, y, o, que, por, para, con, se, su, noticia, tras, segun, hace, puede")
            custom_stopwords = [x.strip() for x in stopwords_input.split(",")]
            
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
#main 
if archivo and col_texto:
    df = df.dropna(subset=[col_texto])
    df[col_texto] = df[col_texto].astype(str)
    all_stopwords = get_stopwords(custom_stopwords)

    tabs = st.tabs(["Resumen Global", "Analisis de Sentimiento", "Lenguaje Profundo", "Clusterizacion (Temas)", "Busqueda", "Redes"])

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
            st.info("Selecciona una columna de agrupación para ver estadísticas.")

    #2
    with tabs[1]:
        st.subheader("Clasificacion de Tono")
        if st.button("Ejecutar Modelo de Sentimiento", type="primary"):
            with st.spinner("Procesando..."):
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
                        st.plotly_chart(px.imshow(pd.crosstab(df_f[col_cat], df_f['Sentimiento'], normalize='index')*100, 
                                        text_auto='.0f', aspect="auto", color_continuous_scale="RdBu", origin='lower',
                                        labels=dict(x="Sentimiento", y=col_cat, color="%")), use_container_width=True)
#3
    with tabs[2]:
        if st.button("Ejecutar Analisis Completo de Lenguaje"):
            nlp = cargar_spacy()
            full_text = " ".join(df[col_texto].tolist())[:1000000]
            
            # 1. NUBE DE PALABRAS
            st.subheader(" Nube de Conceptos")
            wc = WordCloud(width=800, height=300, background_color='white', stopwords=all_stopwords, colormap='viridis').generate(full_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            st.markdown("---")

            
            st.subheader("Deteccion de Entidades (NER)")
            doc = nlp(full_text)
            
            per = [e.text for e in doc.ents if e.label_ == "PER" and len(e.text)>3]
            org = [e.text for e in doc.ents if e.label_ in ["ORG", "MISC"] and len(e.text)>2]
            loc = [e.text for e in doc.ents if e.label_ in ["LOC", "GPE"] and len(e.text)>2]

            def plot_entity(lista, titulo, color):
                if lista:
                    counts = pd.Series(lista).value_counts().head(10).sort_values()
                    fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=titulo, 
                                 color_discrete_sequence=[color])
                    fig.update_layout(showlegend=False, height=450, margin=dict(l=150))
                    return fig
                return None

            col_a, col_b = st.columns(2)
            
            with col_a: 
                fig = plot_entity(per, "Top Personas", "#4285F4") # Azul
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("Sin Personas")

            with col_b: 
                fig = plot_entity(org, "Top Organizaciones", "#EA4335") # Rojo
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("Sin Organizaciones")

            fig_loc = plot_entity(loc, "Top Lugares", "#34A853") # Verde
            if fig_loc: st.plotly_chart(fig_loc, use_container_width=True)
            else: st.info("Sin Lugares")

            st.markdown("---")

            st.subheader("Frases Recurrentes (N-Gramas)")
            c_bi, c_tri = st.columns(2)
            
            with c_bi:
                try:
                    df_bi = get_top_ngrams(df[col_texto], n=2, top_k=10, stopwords=all_stopwords)
                    fig_bi = px.bar(df_bi, x='Frecuencia', y='Frase', orientation='h', title="Top Bigramas (2 palabras)", color='Frecuencia')
                    fig_bi.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bi, use_container_width=True)
                except: st.warning("No hay suficientes datos para Bigramas")

            with c_tri:
                try:
                    df_tri = get_top_ngrams(df[col_texto], n=3, top_k=10, stopwords=all_stopwords)
                    fig_tri = px.bar(df_tri, x='Frecuencia', y='Frase', orientation='h', title="Top Trigramas (3 palabras)", color='Frecuencia')
                    fig_tri.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_tri, use_container_width=True)
                except: st.warning("No hay suficientes datos para Trigramas")

 #4
    # 4. CLUSTERIZACION 
    with tabs[3]:
        st.subheader("Detección de Temas (Topic Modeling)")
        
        # Opciones de configuración para el usuario
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            n_topics_aprox = st.slider("Número esperado de temas (aprox)", 2, 50, 10, help="Esto ajusta la sensibilidad del modelo.")
        with col_opts2:
            force_assign = st.checkbox("Forzar asignación de todos los registros", value=True, help="Si se activa, el sistema intentará eliminar todos los 'Outliers' (-1) reasignándolos al tema más cercano.")

        if st.button("Ejecutar Clustering", type="primary"):
            with st.spinner("Generando Embeddings y Clusters..."):
                try:
                    # 1. Cargar modelo de embeddings 
                    embedding_model = cargar_modelo_embeddings()
                    docs = df[col_texto].tolist()
                    
                    # Pre-calcular embeddings para mayor control
                    embeddings = embedding_model.encode(docs, show_progress_bar=True)

                    #Configurar BERTopic
                    # Ajustamos min_topic_size dinámicamente según el tamaño del dataset
                    min_size = max(5, int(len(docs) * 0.005)) 
                    
                    topic_model = BERTopic(
                        language="multilingual", 
                        min_topic_size=min_size,
                        nr_topics=n_topics_aprox if n_topics_aprox > 5 else "auto", # Auto reducción si el usuario pide pocos
                        calculate_probabilities=True, # Necesario para reasignar outliers
                        verbose=True
                    )
                    
                    # 3. Entrenar
                    topics, probs = topic_model.fit_transform(docs, embeddings)
                    
                    # 4. REASIGNACIÓN DE OUTLIERS (La solución a tu problema)
                    if force_assign:
                        # Estrategia: "c-tf-idf" o "embeddings". 'embeddings' suele ser más preciso para similitud semántica.
                        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
                        topic_model.update_topics(docs, topics=new_topics)
                        topics = new_topics
                        st.success("✅ Outliers reasignados a sus temas más cercanos.")
                    
                    df['Cluster_ID'] = topics
                    
                    # 5. Mostrar Resultados
                    freq = topic_model.get_topic_info()
                    # Filtramos el tema -1 solo si no forzamos asignación o si quedó alguno residual
                    freq_clean = freq[freq['Topic'] != -1].head(15)
                    
                    # Limpieza de nombres para el gráfico
                    freq_clean['Nombre_Tema'] = freq_clean['Name'].apply(lambda x: " ".join(x.split("_")[1:4]))
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("#### Distribución de Temas")
                        fig_bar = px.bar(freq_clean, x='Count', y='Nombre_Tema', orientation='h', 
                                         text_auto=True, title="Temas Principales Detectados")
                        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    with c2:
                        st.markdown("#### Mapa de Distancia")
                        try:
                            st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)
                        except:
                            st.warning("No se generaron suficientes temas para el mapa.")

                    st.markdown("---")
                    
                    st.subheader("Palabras Clave por Grupo")
                    top_clusters = freq_clean['Topic'].tolist()[:6] # Top 6 temas
                    
                    cols_wc = st.columns(3)
                    for i, topic_id in enumerate(top_clusters):
                        # Obtener palabras clave del modelo directamente (más preciso que wordcloud crudo)
                        keywords_dict = {word: score for word, score in topic_model.get_topic(topic_id)}
                        
                        wc_cluster = WordCloud(width=400, height=250, background_color='white', 
                                             colormap='viridis').generate_from_frequencies(keywords_dict)
                        
                        with cols_wc[i % 3]:
                            st.markdown(f"**Grupo {topic_id}: {freq_clean[freq_clean['Topic']==topic_id]['Nombre_Tema'].values[0]}**")
                            fig_wc, ax_wc = plt.subplots(figsize=(4, 3))
                            ax_wc.imshow(wc_cluster, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                            plt.close()

                    # 7. Tabla de Outliers 
                    outliers = df[df['Cluster_ID'] == -1]
                    if len(outliers) > 0:
                        st.markdown("---")
                        st.warning(f"Aún quedan {len(outliers)} registros sin clasificar (ruido extremo).")
                        with st.expander("Ver registros no clasificados"):
                            st.dataframe(outliers[[col_texto]])
                    else:
                        st.balloons()
                        st.success("¡Todos los documentos fueron clasificados exitosamente!")

                except Exception as e:
                    st.error(f"Error en el clustering: {str(e)}")
                    st.info("Intenta reducir el número de temas o aumentar el tamaño de la muestra.")
        # ... código anterior dentro de tabs[3] ...

                    st.markdown("---")
                    st.subheader("🌲 Jerarquía de Temas (Para decidir el número óptimo)")
                    st.markdown("""
                    **¿Cómo leer esto?**
                    - Este árbol agrupa temas similares. 
                    - Si cortas el árbol por la izquierda, tendrás menos temas (más generales).
                    - Si lo cortas por la derecha, tendrás más temas (más específicos).
                    - **Tip:** Busca la línea vertical donde se unen los grandes bloques para estimar el número ideal.
                    """)
                    
                    try:
                        # Hierarchical clustering plot
                        fig_hierarchy = topic_model.visualize_hierarchy(top_n_topics=30)
                        st.plotly_chart(fig_hierarchy, use_container_width=True)
                    except:
                        st.info("Se necesitan más temas para generar la jerarquía.")
                        
    
    #5
    with tabs[4]:
        st.subheader("Motor de Busqueda Semantica")
        query = st.text_input("Consulta:", placeholder="Ej: Problemas de infraestructura...")
        
        if query:
            with st.spinner("Buscando..."):
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
                    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#202124")
                    net.from_nx(G)
                    net.force_atlas_2based()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                        net.save_graph(tmp.name)
                        tmp.seek(0)
                        html_bytes = tmp.read()
                    components.html(html_bytes.decode(), height=600, scrolling=True)
                    os.unlink(tmp.name)
                else:
                    st.warning("No se encontraron suficientes relaciones.")

else:
    st.info("Sube un archivo CSV para comenzar.")




