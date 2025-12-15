import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bertopic import BERTopic

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
st.set_page_config(page_title="Dashboard de Prensa IA", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard de Análisis")
st.markdown("Herramienta de inteligencia de datos para auditoría de medios y detección de tendencias.")

# ==========================================
# 2. FUNCIONES DE CARGA Y PROCESAMIENTO
# ==========================================

@st.cache_resource
def cargar_spacy():
    try:
        return spacy.load("es_core_news_sm")
    except:
        return None

@st.cache_resource
def cargar_modelo_sentimiento():
    nombre = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    return BertTokenizer.from_pretrained(nombre), BertForSequenceClassification.from_pretrained(nombre)

# Función para N-Gramas (Frases repetidas)
def get_top_ngrams(corpus, n=2, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words=STOPWORDS_ES).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=['Frase', 'Frecuencia'])

# Stopwords personalizadas
# Puedes agregar más palabras aquí si ves que ensucian la nube
STOPWORDS_ES = list(cargar_spacy().Defaults.stop_words) + [
    'chile', 'chileno', 'chilena', 'tras', 'hacia', 'según', 'foto', 'video', 'clic', 'aquí', 'noticia', 
    'hoy', 'ayer', 'mañana', 'año', 'años', 'dice', 'ser', 'dijo', 'señaló', 'millones', 'peso', 'pesos',
    'parte', 'gran', 'nuevo', 'nueva', 'frente', 'hacer', 'hace', 'puede', 'forma'
]

# ==========================================
# 3. BARRA LATERAL (CONFIGURACIÓN)
# ==========================================
with st.sidebar:
    st.header("📂 Configuración de Datos")
    archivo = st.file_uploader("1. Sube tu Dataset (CSV)", type=["csv"])
    
    col_texto = None
    col_medio = None

    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.success(f"Cargado: {len(df)} registros")
            
            # Mapeo de columnas
            st.subheader("2. Mapeo de Columnas")
            cols = df.columns.tolist()
            
            # Intenta adivinar la columna de texto
            idx_texto = cols.index('titulo') if 'titulo' in cols else 0
            col_texto = st.selectbox("Columna de TEXTO (Titular/Bajada)", cols, index=idx_texto)
            
            # Intenta adivinar la columna de medio
            idx_medio = cols.index('medio') if 'medio' in cols else 0
            col_medio = st.selectbox("Columna de MEDIO (Opcional)", ["No disponible"] + cols, index=idx_medio + 1)
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")

# ==========================================
# 4. LÓGICA PRINCIPAL DEL DASHBOARD
# ==========================================

if archivo and col_texto:
    # Limpieza inicial
    df = df.dropna(subset=[col_texto])
    df[col_texto] = df[col_texto].astype(str)
    
    # --- PESTAÑAS ---
    tab_resumen, tab_sentimiento, tab_lenguaje, tab_clusters = st.tabs([
        " Resumen General", "Radiografía Emocional", "Análisis de Lenguaje", "Temas (Clustering)"
    ])

    # === PESTAÑA 1: RESUMEN GENERAL ===
    with tab_resumen:
        st.subheader("Datos Generales del Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Noticias Analizadas", len(df))
        
        if col_medio != "No disponible":
            c2.metric("Medios Monitoreados", df[col_medio].nunique())
            try:
                top_medio = df[col_medio].mode()[0]
                c3.metric("Medio más activo", top_medio)
            except:
                c3.metric("Medio más activo", "N/A")
            
            st.markdown("---")
            st.subheader("Participación por Medio")
            conteo_medios = df[col_medio].value_counts().reset_index()
            conteo_medios.columns = ['Medio', 'Noticias']
            fig_medios = px.bar(conteo_medios.head(15), x='Noticias', y='Medio', orientation='h', 
                                title="Top 15 Medios con más noticias", color='Noticias', color_continuous_scale='Blues')
            fig_medios.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_medios, use_container_width=True)
        else:
            c2.metric("Medios", "N/A")
            st.info("Para ver estadísticas por medio, selecciona la columna correspondiente en la barra lateral.")

    # === PESTAÑA 2: SENTIMIENTO ===
    with tab_sentimiento:
        st.subheader("Análisis de Sentimiento (IA)")
        
        if st.button("▶️ Ejecutar Análisis de Sentimiento", type="primary"):
            with st.spinner('Analizando tono de las noticias...'):
                tokenizer, model = cargar_modelo_sentimiento()
                
                def predecir_batch(textos):
                    inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cpu")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    return ["Positivo" if p[1] > p[0] else "Negativo" for p in probs]

                batch_size = 32
                todos_sents = []
                textos = df[col_texto].tolist()
                
                progreso = st.progress(0)
                for i in range(0, len(textos), batch_size):
                    batch = textos[i:i+batch_size]
                    todos_sents.extend(predecir_batch(batch))
                    progreso.progress(min((i + batch_size) / len(textos), 1.0))
                
                df['Sentimiento'] = todos_sents
                st.success("Análisis completado")

                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_pie = px.pie(df, names='Sentimiento', title="Distribución Global", 
                                     color='Sentimiento', color_discrete_map={'Positivo':'#2ecc71', 'Negativo':'#e74c3c'})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    if col_medio != "No disponible":
                        st.subheader(" Mapa de Calor: Línea Editorial")
                        conteo_minimo = 2
                        medios_validos = df[col_medio].value_counts()
                        medios_validos = medios_validos[medios_validos > conteo_minimo].index
                        df_filtrado = df[df[col_medio].isin(medios_validos)]
                        cruce = pd.crosstab(df_filtrado[col_medio], df_filtrado['Sentimiento'], normalize='index') * 100
                        fig_heat = px.imshow(cruce, text_auto='.1f', aspect="auto",
                                             labels=dict(x="Sentimiento", y="Medio", color="%"),
                                             color_continuous_scale="RdBu", origin='lower')
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.info("Selecciona columna 'Medio' para ver el Mapa de Calor.")

                st.download_button("Descargar CSV con Sentimiento", df.to_csv(index=False), "datos_sentimiento.csv")

    # === PESTAÑA 3: LENGUAJE (AQUÍ ESTÁ LA NUEVA NUBE) ===
    with tab_lenguaje:
        if st.button("▶️ Analizar Texto y Entidades"):
            nlp = cargar_spacy()
            
            with st.spinner("Generando nube y extrayendo entidades..."):
                # 1. Preparar texto completo
                texto_total = " ".join(df[col_texto].astype(str).tolist())
                
                # --- NUEVA SECCIÓN: NUBE DE PALABRAS GENERAL ---
                st.subheader(" Nube de Palabras Global")
                st.markdown("Conceptos más repetidos en todo el dataset:")
                
                # Generamos la nube
                wc_general = WordCloud(
                    width=800, height=400, 
                    background_color='white', 
                    stopwords=STOPWORDS_ES,
                    colormap='viridis',
                    max_words=100
                ).generate(texto_total[:2000000]) # Limitamos caracteres para proteger memoria

                # Mostramos con Matplotlib
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wc_general, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
                plt.close() # Cerramos figura para liberar memoria
                # -----------------------------------------------

                # 2. PROCESAMIENTO NLP (ENTIDADES)
                doc = nlp(texto_total[:1000000]) 

                personas = [ent.text for ent in doc.ents if ent.label_ == "PER" and len(ent.text) > 3]
                orgs = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "MISC"] and len(ent.text) > 2]
                lugares = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"] and len(ent.text) > 2]

                def plot_top_ent(lista, titulo, color):
                    if not lista: return
                    counts = pd.Series(lista).value_counts().head(10).sort_values(ascending=True)
                    fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=titulo,
                                 labels={'x':'Menciones', 'y':''}, color_discrete_sequence=[color])
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🕵️ Detección de Entidades")
                c1, c2, c3 = st.columns(3)
                with c1: plot_top_ent(personas, "Top Personajes", "#e74c3c")
                with c2: plot_top_ent(orgs, "Top Organizaciones", "#3498db")
                with c3: plot_top_ent(lugares, "Top Lugares", "#27ae60")

                st.markdown("---")

                # 3. ANÁLISIS DE N-GRAMAS
                st.subheader("🗣️ Frases más repetidas")
                c_bi, c_tri = st.columns(2)
                
                with c_bi:
                    df_bi = get_top_ngrams(df[col_texto], n=2, top_k=10)
                    fig_bi = px.bar(df_bi, x='Frecuencia', y='Frase', orientation='h', 
                                    title="Top Bigramas (2 palabras)", color='Frecuencia', color_continuous_scale='Viridis')
                    fig_bi.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bi, use_container_width=True)

                with c_tri:
                    df_tri = get_top_ngrams(df[col_texto], n=3, top_k=10)
                    fig_tri = px.bar(df_tri, x='Frecuencia', y='Frase', orientation='h', 
                                     title="Top Trigramas (3 palabras)", color='Frecuencia', color_continuous_scale='Magma')
                    fig_tri.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_tri, use_container_width=True)

    # === PESTAÑA 4: CLUSTERING ===
    with tab_clusters:
        st.subheader("Descubrimiento de Temas (BERTopic)")
        
        if st.button("▶️ Generar Clusters (Puede tardar)", type="primary"):
            with st.spinner("Entrenando modelo de clustering..."):
                topic_model = BERTopic(language="multilingual", min_topic_size=5)
                docs = df[col_texto].tolist()
                topics, probs = topic_model.fit_transform(docs)
                
                freq = topic_model.get_topic_info()
                df['Tema_ID'] = topics
                
                st.subheader("Cantidad de Noticias por Tema")
                freq_clean = freq[freq['Topic'] != -1].head(10)
                freq_clean['Nombre_Corto'] = freq_clean['Name'].apply(lambda x: "_".join(x.split("_")[1:3]))
                
                fig_bar = px.bar(freq_clean, x='Nombre_Corto', y='Count', 
                                 title="Temas Principales Detectados", 
                                 text_auto=True,
                                 color='Count', color_continuous_scale='Purples')
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Mapa de Inter-distancia de Temas")
                fig_map = topic_model.visualize_topics()
                st.plotly_chart(fig_map, use_container_width=True)

                st.subheader("Nubes de Palabras por Grupo")
                cols = st.columns(3)
                top_clusters = freq_clean['Topic'].tolist()[:6] 
                
                for idx, topic_id in enumerate(top_clusters):
                    text_cluster = " ".join(df[df['Tema_ID'] == topic_id][col_texto].tolist())
                    wc = WordCloud(width=400, height=200, background_color='white', stopwords=STOPWORDS_ES).generate(text_cluster)
                    
                    with cols[idx % 3]:
                        st.markdown(f"**Tema {topic_id}**")
                        plt.figure(figsize=(5, 3))
                        plt.imshow(wc, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                        plt.close()

else:
    st.info("👈 Comienza subiendo un archivo CSV en la barra lateral.")             
