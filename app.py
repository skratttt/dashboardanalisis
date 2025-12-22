import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
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
import zipfile 
import io      

# ==========================================
# 0. CONFIGURACIÓN GLOBAL Y RECOLECTOR
# ==========================================
pio.templates.default = "plotly_white"
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

if 'figures_to_export' not in st.session_state:
    st.session_state.figures_to_export = {}

def mostrar_y_guardar(fig, nombre_archivo, use_container_width=True):
    """
    1. Muestra el gráfico en la app.
    2. Lo guarda en la memoria permanente para el ZIP.
    """
    st.plotly_chart(fig, use_container_width=use_container_width)
    
    clean_name = "".join(x for x in nombre_archivo if x.isalnum() or x in " -_").strip()
    st.session_state.figures_to_export[clean_name] = fig

# ==========================================
# 1. CONFIGURACIÓN ESTILO STREAMLIT
# ==========================================
st.set_page_config(page_title="Analisis de texto", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {font-family: 'Sans-serif'; color: #202124;}
    .stApp {background-color: white;}
    
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
    div.stButton > button[kind="secondary"] {
        background-color: white;
        color: #1A73E8;
        border: 1px solid #dadce0;
        border-radius: 24px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 20px;
        padding: 0px 20px;
        color: #5f6368;
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

# ==========================================
# 2. FUNCIONES CACHEADAS
# ==========================================
@st.cache_resource
def cargar_spacy():
    try: 
        # Intentamos cargar el modelo MEDIUM (Equilibrio perfecto)
        return spacy.load("es_core_news_md")
    except: 
        try:
            # Si falla, intentamos el pequeño (backup)
            return spacy.load("es_core_news_sm")
        except: return None

@st.cache_resource
def cargar_modelo_sentimiento():
    nombre = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    return BertTokenizer.from_pretrained(nombre), BertForSequenceClassification.from_pretrained(nombre)

@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

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

# ==========================================
# 3. SIDEBAR Y CARGA DE DATOS
# ==========================================
with st.sidebar:
    st.header("Configuracion del Dataset")
    archivo = st.file_uploader("1. Subir Archivo (CSV)", type=["csv"])
    
    col_texto = None
    col_cat = None
    col_fecha = None 
    custom_stopwords = []

    if archivo:
        # Limpieza de memoria al cambiar archivo
        file_id = f"{archivo.name}-{archivo.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            st.session_state.figures_to_export = {}
            st.session_state.last_file_id = file_id
            st.session_state.sentimiento_data = None

        try:
            try:
                df = pd.read_csv(archivo, sep=';')
                if df.shape[1] < 2:
                    archivo.seek(0)
                    df = pd.read_csv(archivo, sep=',')
            except UnicodeDecodeError:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=';', encoding='latin-1')

            st.success(f"Registros cargados: {len(df)}")
            cols = df.columns.tolist()
            
            idx_txt = 0
            for possible in ['texto', 'text', 'comentario', 'mensaje', 'descripcion', 'titulo']:
                match = next((c for c in cols if possible.lower() == c.lower()), None)
                if match:
                    idx_txt = cols.index(match)
                    break
            
            col_texto = st.selectbox("2. Columna de TEXTO", cols, index=idx_txt)
            
            if col_texto:
                def limpiar_texto_duro(txt):
                    if not isinstance(txt, str): return str(txt)
                    t = txt.lower()
                    t = t.replace("ee.uu.", "eeuu").replace("ee.uu", "eeuu")
                    t = t.replace("ee. uu.", "eeuu").replace("ee. uu", "eeuu")
                    t = t.replace("&quot;", "").replace("quot", "")
                    return t
                df[col_texto] = df[col_texto].apply(limpiar_texto_duro)

            st.info("Opcional: Agrupador")
            col_cat = st.selectbox("3. Columna de AGRUPACIÓN", ["No aplicar"] + cols)
            
            st.info("Opcional: Análisis Temporal")
            col_fecha = st.selectbox("4. Columna de FECHA", ["No aplicar"] + cols, 
                                   help="Debe ser una columna con fechas (ej: 2023-10-25)")

            st.header("Filtros de Texto")
            lista_defecto = "el, la, los, un, una, de, del, y, o, que, qué, quien, quién, por, para, con, se, su, sus, lo, las, al, como, cómo, mas, más, noticia, tras, segun, según, hace, puede, ser, es, son, fue, eran, era, habia, hay"
            stopwords_input = st.text_area("Palabras a ignorar (separadas por coma)", lista_defecto, height=100)
            custom_stopwords = [x.strip().lower() for x in stopwords_input.split(",")]
            
            st.markdown("---")
            st.header("🔍 Filtro Global")
            filtro_palabra = st.text_input("Filtrar análisis por palabra clave:", placeholder="Ej: litio...")
            
            if filtro_palabra:
                mask = df[col_texto].str.contains(filtro_palabra, case=False, na=False)
                df = df[mask]
                st.success(f"Filtrado: {len(df)} registros contienen '{filtro_palabra}'")
            
            st.markdown("---")
            st.header("📦 Descarga Masiva")
            
            if st.button("Generar Reporte Visual (ZIP)"):
                graficos = st.session_state.figures_to_export
                if not graficos:
                    st.warning("⚠️ No hay gráficos en memoria. Navega por las pestañas para generarlos primero.")
                else:
                    with st.spinner(f"📸 Procesando {len(graficos)} gráficos..."):
                        try:
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                for nombre, fig in graficos.items():
                                    img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
                                    zf.writestr(f"{nombre}.png", img_bytes)
                            
                            st.download_button(
                                label=f"📥 Descargar ZIP ({len(graficos)} gráficos)",
                                data=zip_buffer.getvalue(),
                                file_name="reporte_graficos.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.error(f"Error generando ZIP (¿Instalaste kaleido?): {e}")

        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

# ==========================================
# 4. APLICACIÓN PRINCIPAL
# ==========================================
if archivo and col_texto:
    df = df.dropna(subset=[col_texto])
    df[col_texto] = df[col_texto].astype(str)
    all_stopwords = get_stopwords(custom_stopwords)

    tabs = st.tabs(["Resumen Global", "Analisis de Sentimiento", "Lenguaje Profundo", "Clusterizacion (Temas)", "Busqueda", "Redes", "Monitor de Tendencias"])

    # ---------------- TAB 1: RESUMEN ----------------
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.metric("Total de Documentos", len(df))
        
        if col_cat != "No aplicar":
            c2.metric(f"Total de {col_cat} únicos", df[col_cat].nunique())
            
            conteo = df[col_cat].value_counts().reset_index()
            conteo.columns = ['Categoria', 'Count']
            
            fig = px.bar(conteo.head(20), x='Count', y='Categoria', orientation='h', 
                         title=f"Distribución por {col_cat}", text_auto=True)
            
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                yaxis_tickmode='linear',
                margin=dict(l=10, r=10, t=50, b=10)
            )
            fig.update_yaxes(automargin=True)
            mostrar_y_guardar(fig, f"Resumen_Distribucion_{col_cat}")
        else:
            st.info("Selecciona una columna de agrupación para ver estadísticas.")

    # ---------------- TAB 2: SENTIMIENTO ----------------
    with tabs[1]:
        st.subheader("Clasificacion de Tono")
        
        if 'sentimiento_data' not in st.session_state:
            st.session_state.sentimiento_data = None
        
        if st.session_state.sentimiento_data is None:
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
                    
                    st.session_state.sentimiento_data = res
                    st.rerun()

        if st.session_state.sentimiento_data is not None:
            df['Sentimiento'] = st.session_state.sentimiento_data
            
            if st.button("🔄 Reiniciar Análisis", type="secondary"):
                st.session_state.sentimiento_data = None
                st.rerun()

            c1, c2 = st.columns([1, 2])
            with c1: 
                st.markdown("##### Distribución Global")
                fig_pie = px.pie(df, names='Sentimiento', color='Sentimiento', color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'})
                mostrar_y_guardar(fig_pie, "Sentimiento_Global_Pie")
            
            with c2:
                if col_cat != "No aplicar":
                    st.markdown(f"##### Comparativa por {col_cat}")
                    c_fil, _ = st.columns([1, 1])
                    with c_fil:
                        opcion_top = st.selectbox("Filtrar por volumen:", ["Top 3", "Top 5", "Top 10", "Top 20", "Todos"], index=2)
                    
                    conteo_total = df[col_cat].value_counts()
                    if opcion_top != "Todos":
                        n_top = int(opcion_top.split(" ")[1])
                        cats_to_keep = conteo_total.head(n_top).index
                        df_f = df[df[col_cat].isin(cats_to_keep)]
                    else:
                        df_f = df.copy()

                    df_grouped = df_f.groupby([col_cat, 'Sentimiento']).size().reset_index(name='Conteo')
                    df_grouped['Porcentaje'] = df_grouped.groupby(col_cat)['Conteo'].transform(lambda x: 100 * x / x.sum())
                    
                    n_categorias = df_grouped[col_cat].nunique()
                    alto_grafico = max(350, n_categorias * 40) 

                    fig_bar = px.bar(df_grouped, x="Porcentaje", y=col_cat, color="Sentimiento", orientation='h',
                        text_auto='.0f', hover_data={'Porcentaje':':.1f', 'Conteo':True},
                        color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'}, height=alto_grafico 
                    )
                    fig_bar.update_layout(xaxis_title="% del Total", yaxis_title="", legend_title=dict(text=""), yaxis={'categoryorder':'total ascending'})
                    fig_bar.update_traces(textposition='inside', textfont_color='white')
                    mostrar_y_guardar(fig_bar, f"Sentimiento_Detalle_{col_cat}")
                    
                    with st.expander("📊 Ver Tabla de Datos Exactos", expanded=True):
                        tabla = pd.crosstab(df_f[col_cat], df_f['Sentimiento'])
                        tabla['Total'] = tabla.sum(axis=1)
                        tabla = tabla.sort_values('Total', ascending=False)
                        st.dataframe(tabla, use_container_width=True)
                else:
                    st.info(f"Selecciona una columna de agrupación en la barra lateral.")

    # ---------------- TAB 3: LENGUAJE PROFUNDO (NER MEJORADO) ----------------
    with tabs[2]:
        if st.button("Ejecutar Análisis Completo de Lenguaje"):
            nlp = cargar_spacy()
            
            # Unimos texto
            full_text = " ".join(df[col_texto].tolist())[:1000000]
            
            # 1. NUBE DE PALABRAS (CORREGIDA - SOLUCIÓN DEFINITIVA)
            st.subheader("☁️ Nube de Conceptos")
            wc = WordCloud(width=800, height=300, background_color='white', stopwords=all_stopwords, colormap='viridis').generate(full_text)
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
            
            # --- CORRECCIÓN AQUÍ: .to_image() funciona con NumPy nuevo ---
            ax.imshow(wc.to_image(), interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            
            # 2. DETECCIÓN DE ENTIDADES (NER) - VERSIÓN MEJORADA
            st.subheader("🕵️ Detección de Entidades (NER)")
            
            with st.spinner("Analizando gramática y entidades..."):
                doc = nlp(full_text)
                
                # --- FUNCIÓN DE FILTRADO INTELIGENTE ---
                def es_entidad_valida(entidad):
                    txt = entidad.text.lower().strip()
                    if txt in all_stopwords or len(txt) < 3: return False
                    if "(" in txt or ")" in txt or "http" in txt or "%" in txt: return False
                    if len(entidad) == 1: 
                        pos = entidad[0].pos_
                        if pos in ["VERB", "ADV", "ADJ", "NUM", "AUX", "SCONJ", "DET"]: return False
                    return True
                # ----------------------------------------

                per = []
                org = []
                loc = []
                
                for e in doc.ents:
                    if es_entidad_valida(e):
                        if e.label_ == "PER": per.append(e.text)
                        elif e.label_ in ["ORG", "MISC"]: org.append(e.text)
                        elif e.label_ in ["LOC", "GPE"]: loc.append(e.text)

            def plot_entity(lista, titulo, color):
                if lista:
                    counts = pd.Series(lista).value_counts().head(10).sort_values()
                    if not counts.empty:
                        fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=titulo, color_discrete_sequence=[color])
                        fig.update_layout(showlegend=False, height=450, margin=dict(l=150))
                        fig.update_yaxes(automargin=True)
                        return fig
                return None

            col_a, col_b = st.columns(2)
            with col_a: 
                fig_per = plot_entity(per, "Top Personas", "#4285F4")
                if fig_per: mostrar_y_guardar(fig_per, "Entidades_Personas")
                else: st.info("Sin Personas detectadas")

            with col_b: 
                fig_org = plot_entity(org, "Top Organizaciones", "#EA4335")
                if fig_org: mostrar_y_guardar(fig_org, "Entidades_Organizaciones")
                else: st.info("Sin Organizaciones detectadas")

            fig_loc = plot_entity(loc, "Top Lugares", "#34A853")
            if fig_loc: mostrar_y_guardar(fig_loc, "Entidades_Lugares")
            else: st.info("Sin Lugares detectados")

            st.markdown("---")
            
            # 3. N-GRAMAS
            st.subheader("🔠 Frases Recurrentes (N-Gramas)")
            c_bi, c_tri = st.columns(2)
            with c_bi:
                try:
                    df_bi = get_top_ngrams(df[col_texto], n=2, top_k=10, stopwords=all_stopwords)
                    fig_bi = px.bar(df_bi, x='Frecuencia', y='Frase', orientation='h', title="Top Bigramas", color='Frecuencia')
                    fig_bi.update_layout(yaxis={'categoryorder':'total ascending'})
                    mostrar_y_guardar(fig_bi, "Bigramas")
                except: st.warning("No hay suficientes datos")
            with c_tri:
                try:
                    df_tri = get_top_ngrams(df[col_texto], n=3, top_k=10, stopwords=all_stopwords)
                    fig_tri = px.bar(df_tri, x='Frecuencia', y='Frase', orientation='h', title="Top Trigramas", color='Frecuencia')
                    fig_tri.update_layout(yaxis={'categoryorder':'total ascending'})
                    mostrar_y_guardar(fig_tri, "Trigramas")
                except: st.warning("No hay suficientes datos")

    # ---------------- TAB 4: CLUSTERIZACION ----------------
    with tabs[3]:
        st.subheader("Detección de Patrones (Topic Modeling)")
        
        c_controls_1, c_controls_2 = st.columns(2)
        with c_controls_1:
            n_topics_aprox = st.slider("Número de Temas Deseados", 2, 50, 5)
        with c_controls_2:
            force_assign = st.checkbox("Forzar asignación de Outliers", value=True)

        if st.button("Ejecutar Clustering", type="primary"):
            with st.spinner("Generando Embeddings y Clusters..."):
                try:
                    embedding_model = cargar_modelo_embeddings()
                    docs = df[col_texto].tolist()
                    embeddings = embedding_model.encode(docs, show_progress_bar=False)
                    vectorizer_model = CountVectorizer(stop_words=all_stopwords, min_df=2)
                    min_size = max(5, int(len(docs) * 0.005))
                    
                    topic_model = BERTopic(language="multilingual", min_topic_size=min_size, nr_topics=n_topics_aprox,
                                          vectorizer_model=vectorizer_model, calculate_probabilities=True, verbose=True)
                    
                    topics, probs = topic_model.fit_transform(docs, embeddings)
                    
                    if force_assign:
                        try:
                            new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
                            topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
                            topics = new_topics
                        except: pass
                    
                    df['Cluster_ID'] = topics
                    freq = topic_model.get_topic_info()
                    freq_clean = freq[freq['Topic'] != -1].head(20)
                    freq_clean['Nombre_Tema'] = freq_clean['Name'].apply(lambda x: " ".join(x.split("_")[1:4]))
                    
                    col_res1, col_res2 = st.columns([2, 1])
                    with col_res1:
                        st.markdown("#### Distribución de Temas")
                        fig_bar = px.bar(freq_clean, x='Count', y='Nombre_Tema', orientation='h', 
                                         text_auto=True, title="Temas Detectados", color='Count')
                        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                        mostrar_y_guardar(fig_bar, "Cluster_Distribucion_Temas")
                        
                    with col_res2:
                        st.markdown("#### Mapa Intertópico")
                        try:
                            fig_inter = topic_model.visualize_topics()
                            mostrar_y_guardar(fig_inter, "Cluster_Mapa_Intertopico")
                        except:
                            st.info("Se necesitan más temas.")

                    st.markdown("---")
                    st.subheader("Palabras Clave por Grupo")
                    top_clusters = freq_clean['Topic'].tolist()[:6] 
                    cols_wc = st.columns(3)
                    for i, topic_id in enumerate(top_clusters):
                        topic_words = topic_model.get_topic(topic_id)
                        if topic_words:
                            keywords_dict = {word: score for word, score in topic_words}
                            wc_cluster = WordCloud(width=400, height=250, background_color='white', colormap='viridis').generate_from_frequencies(keywords_dict)
                            with cols_wc[i % 3]:
                                st.markdown(f"**Grupo {topic_id}**")
                                fig_wc, ax_wc = plt.subplots(figsize=(4, 3), facecolor='white')
                                # --- CORRECCIÓN AQUÍ TAMBIÉN: .to_image() ---
                                ax_wc.imshow(wc_cluster.to_image(), interpolation='bilinear')
                                ax_wc.axis('off')
                                st.pyplot(fig_wc)
                                plt.close()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------------- TAB 5: BUSQUEDA ----------------
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
                    st.markdown(f"""<div style="background-color:#f8f9fa;padding:15px;margin-bottom:10px;">
                        <small><b>Relevancia: {score:.2f}</b> | {cat_val}</small><br>{txt}</div>""", unsafe_allow_html=True)

    # ---------------- TAB 6: REDES ----------------
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
                    G.add_node(node, size=node_counts[node]*2)
                for (source, target), weight in co_occurrences.items():
                    if source in top_nodes and target in top_nodes:
                        G.add_edge(source, target, value=weight)
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

    # ---------------- TAB 7: MONITOR DE TENDENCIAS (SIN SLIDER) ----------------
    with tabs[6]:
        st.subheader("⏳ Monitor de Tendencias y Agenda")
        
        if col_fecha != "No aplicar":
            try:
                df_time = df.copy()
                df_time[col_fecha] = pd.to_datetime(df_time[col_fecha], errors='coerce')
                df_time = df_time.dropna(subset=[col_fecha])
                
                if len(df_time) > 0:
                    # Configuración Fija por defecto: DÍA
                    intervalo = "D" 
                    
                    # 1. RANKING DE ACTORES
                    st.subheader("🏆 La Carrera de la Agenda")
                    st.caption("Visualiza quién domina la conversación.")
                    
                    tipo_tendencia = st.radio("Analizar:", ["Personas", "Organizaciones", "Temas (Clave)"], horizontal=True)
                    
                    if st.button(f"Generar Evolución de {tipo_tendencia}"):
                        with st.spinner("Analizando tendencias históricas..."):
                            nlp = cargar_spacy()
                            def extraer_items(texto):
                                doc = nlp(str(texto))
                                items = []
                                if tipo_tendencia == "Personas":
                                    items = [e.text for e in doc.ents if e.label_ == "PER" and len(e.text) > 3]
                                elif tipo_tendencia == "Organizaciones":
                                    items = [e.text for e in doc.ents if e.label_ in ["ORG", "MISC"] and len(e.text) > 2]
                                else:
                                    items = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and len(t.text) > 3]
                                return [i for i in items if i.lower() not in all_stopwords]

                            df_time['Items'] = df_time[col_texto].apply(extraer_items)
                            df_exploded = df_time.explode('Items').dropna(subset=['Items'])
                            top_global = df_exploded['Items'].value_counts().head(6).index.tolist()
                            df_top = df_exploded[df_exploded['Items'].isin(top_global)]
                            pivot_trend = df_top.groupby([pd.Grouper(key=col_fecha, freq=intervalo), 'Items']).size().reset_index(name='Menciones')
                            
                            if not pivot_trend.empty:
                                pivot_trend['Ranking'] = pivot_trend.groupby(col_fecha)['Menciones'].rank(method='first', ascending=False)
                                
                                # Ranking (Bump Chart)
                                fig_bump = px.line(pivot_trend, x=col_fecha, y='Ranking', color='Items', markers=True,
                                                  title=f"Ranking de {tipo_tendencia} (Top 6)", line_shape='spline')
                                fig_bump.update_yaxes(autorange="reversed", dtick=1)
                                mostrar_y_guardar(fig_bump, f"Ranking_{tipo_tendencia}")
                                
                                # Volumen Tendencia
                                fig_line = px.line(pivot_trend, x=col_fecha, y='Menciones', color='Items',
                                                  title=f"Volumen de menciones", line_shape='spline')
                                mostrar_y_guardar(fig_line, f"Volumen_Tendencia_{tipo_tendencia}")
                            else:
                                st.warning("Datos insuficientes para tendencias.")

                    # 2. MATRIZ DE CALOR
                    if col_cat != "No aplicar":
                        st.markdown("---")
                        st.subheader(f"🔥 Matriz de Intensidad: {col_cat} vs Tiempo")
                        heatmap_data = df_time.groupby([pd.Grouper(key=col_fecha, freq=intervalo), col_cat]).size().reset_index(name='Cantidad')
                        
                        top_fuentes = heatmap_data.groupby(col_cat)['Cantidad'].sum().nlargest(15).index.tolist()
                        heatmap_pivot = heatmap_data[heatmap_data[col_cat].isin(top_fuentes)].pivot(index=col_cat, columns=col_fecha, values='Cantidad').fillna(0)
                        heatmap_pivot = heatmap_pivot.reindex(top_fuentes)
                        
                        fig_heat = px.imshow(heatmap_pivot, aspect="auto", color_continuous_scale="Reds", text_auto=True if len(heatmap_pivot.columns)<20 else False)
                        fig_heat.update_layout(yaxis_nticks=len(top_fuentes))
                        mostrar_y_guardar(fig_heat, "Matriz_Intensidad_Medios")

                    # 3. SENTIMIENTO
                    if 'Sentimiento' in df_time.columns:
                        st.markdown("---")
                        st.subheader("Sentimiento Acumulado")
                        sent_time = df_time.groupby([pd.Grouper(key=col_fecha, freq=intervalo), 'Sentimiento']).size().reset_index(name='Conteo')
                        fig_sent = px.area(sent_time, x=col_fecha, y='Conteo', color='Sentimiento',
                                           color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'})
                        mostrar_y_guardar(fig_sent, "Evolucion_Sentimiento_Area")

                else:
                    st.error("No se encontraron fechas válidas.")
            except Exception as e:
                st.error(f"Error procesando fechas: {e}")
        else:
            st.warning("Selecciona una columna de FECHA en la barra lateral.")

else:
    st.info("Sube un archivo CSV para comenzar.")
