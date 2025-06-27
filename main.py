import streamlit as st
from streamlit_option_menu import option_menu
import time
import glob
import graphviz as gv
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def menu_descricao():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🧠 Hidrodinâmica Cerebral e Segmentação de Tecidos
        """)
        col_int1, col_int2 = st.columns([1, 1])
        with col_int1:
            st.markdown("""
            **Problema Central:**  
            A análise convencional dos tecidos cerebrais — substância cinzenta (GM), substância branca (WM) e líquido cerebrospinal (CSF) — em diferentes modalidades de imagem demanda:
            - Transformações espaciais não lineares entre diferentes domínios de imagem
            - Processamento computacional intensivo
            - Elevada capacidade de armazenamento, devido ao grande volume de dados gerados tanto pelas imagens quanto pelos arquivos intermediários do processamento
            """)
        with col_int2:
            st.markdown("""
            **Motivação:**  
            Estudos crescentes envolvendo o efeito e a relção de processos figiológicos (respiração e pulsação) na hidrodinâmica cerebral com medidas não invasivas por imagens de ressonância magnética de difusão dinâmica.
            - Reduzir o tempo no processamento dos mapas 
            - Simplificação de processos       
            """)

    with col2:
        try:
            # Carrega diretamente a imagem específica
            img = Image.open("imagens/descricao.png")
            st.image(
                img, 
                # caption="Diagrama do Processamento",
                use_container_width=True
            )
        except FileNotFoundError:
            st.error("Imagem 'descricao.png' não encontrada na pasta imagens_menu!")
        except Exception as e:
            st.error(f"Erro ao carregar imagem: {str(e)}")
    # img = Image.open("imagens/dados.png")
    col1, col2, col3 = st.columns([1, 6, 1])  # Proporções que criam margens laterais
    with col2:
        st.image("imagens/dados.png", use_container_width=True)
        st.caption("""
        <div style='text-align: center'>
        <b>Figura 1:</b> Exemplo de imagens dynDWI - b0 (esquerda), b0 inverso (centro) e imagens ponderadas (direita)
        </div>
        """, unsafe_allow_html=True)

def menu_objetivos(): 
    col1, col2, col3 = st.columns([1, 6, 1])  # Proporções que criam margens laterais
    with col2:
        st.markdown("""
        ### O que queremos fazer?  
        - **Simplificar a análise**: Reduzir ou eliminiar a necessidade de trasnformar imagens entre espaços (nativo/difusão <-> estrutural <-> padrão).
        - **Nossa Solução**:  Avaliar 3 algoritmos de **Machine Learning** para segmentação **direta no espaço nativo**:
            - XGBoost (Otimizado com GridSearch)
            - GMM (Modelo de Mistura Gaussiana)
            - MLP (Perceptron Multicamada)
        - **Comparar modelos**: Mostrar diferenças entre cada um deles.  
        - **Validar resultados**: Destacar correlações entre séries temporais de ADC (segmentação com modelo e tradicional) e variabilidade entre sujeitos.  
        """)
        st.image("imagens/treinamento.png", use_container_width=True)
        st.caption("""
        <div style='text-align: center'>
        <b>Figura 1:</b> Exemplo de mapa de ADC (esquerda) e, respectivamente, mascaras de CSF, WM e GM
        </div>
        """, unsafe_allow_html=True)
    

def menu_dados():
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Imagens de MRI de três adultos saudáveis, nas seguintes modalidades:")
        st.markdown("""
        - **dynDWI.nii**: Imagem de difusão dinâmica 5D na matriz 128x128x30x50x16 — três dimensões espaciais, uma temporal e uma de difusão (50 repetições, b = 150 s/mm²).  
        - **T1.nii**: Imagem estrutural T1 3D na matriz 208x320x320.  
        - **b0_inv.nii**: Imagem sem ponderação em difusão, com codificação de fase de leitura oposta à da dynDWI.nii.  
        """)
        st.image("imagens/histograma.png", width=700)
    
    with col2:
        st.subheader("O que precisamos antes de aplicar os modelos?")
        st.markdown("""
        - **ADC.nii**: Mapa do coeficiente aparente de difusão (ADC) no espaço estrutural e nativo (3D).
        - **CSF, WM e GM .nii**: Máscaras binárias de líquido cefalorraquidiano, matéria branca e matéria cinzenta no espaço estrutural (3D).
        - **CSV**: Arquivo contendo a série temporal do valor médio do ADC para cada um dos tecidos.  
        """)
        
        col2_1, col2_2 = st.columns([1, 1])
        with col2_1:
            st.markdown("""
                ##### Processamento das imagens  
                """)
            st.image("imagens/fluxo.png", width=700)

        with col2_2:
            st.markdown("""
                ##### Pré-processamento → Modelos → Avaliação
                """)
            st.image("imagens/fluxo_modelo.png", width=700)

def display_metrics(df):
    # Filtrar apenas linhas com métricas por classe (remover médias)
    df_classes = df[~df['classe'].str.contains('avg|accuracy')]
    # Mostrar dataframe interativo
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("Selecione a métrica:")
    with col2:
        metric =st.selectbox(
                    label="Selecione a métrica:",
                    options=['precision', 'recall', 'f1-score'],
                    key='metric',
                    label_visibility="collapsed"
                )
    
    with st.expander("Comparação de Métricas entre Modelos", expanded=True):
        fig = px.bar(df_classes, 
                 x='classe', 
                 y=metric, 
                 color='modelo',
                 barmode='group',
                 color_discrete_map={
                     'GMM': '#440154',
                     'MLP': '#21918c',
                     'XGB': '#fde725'
                 },
                 labels={'classe': 'Classe', metric: metric.capitalize()})
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Métricas Gerais", expanded=True):
        st.dataframe(
        df_classes.style.format({
            'precision': '{:.2f}',
            'recall': '{:.2f}',
            'f1-score': '{:.4f}',
            'support': '{:.0f}'
        }).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
    )

def display_confusion_matrix(df):
    # Selecionar modelo
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("Selecione o modelo:")
    with col2:
        model = st.selectbox(
                        label="Selecione o modelo:",
                        options= df['modelo'].unique(),
                        key='model_selector',
                        label_visibility="collapsed"
                    )
    
    # Filtrar dados
    df_model = df[df['modelo'] == model]
    
    # Preparar matriz de confusão em porcentagem
    classes = ['Fundo', 'CSF', 'GM', 'WM']
    conf_matrix = df_model[['Pred_0', 'Pred_1', 'Pred_2', 'Pred_3']].values
    true_counts = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percent = (conf_matrix / true_counts * 100).round(1)
    
    # Criar heatmap com porcentagens
    fig = px.imshow(
        conf_matrix_percent,
        labels=dict(x="Predito", y="Verdadeiro", color="%"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale='Viridis',
        aspect="auto",
        zmin=0,
        zmax=100
    )
    
    annotations = []

    
    fig.update_layout(
        title=f'Matriz de Confusão - {model}',
        xaxis_title='Classe Predita',
        yaxis_title='Classe Verdadeira',
        annotations=annotations,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Ajustes adicionais
    fig.update_coloraxes(colorbar_title="%")
    fig.update_traces(
        hovertemplate="<br>".join([
            "Verdadeiro: %{y}",
            "Predito: %{x}",
            "Contagem: %{customdata[0]}",
            "Porcentagem: %{z}%"
        ]),
        customdata=np.dstack([conf_matrix, conf_matrix_percent])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def menu_analises():
    tab1, tab2= st.tabs(["Métricas de Treinamento", "Comparação Inter-sujeito"])
    with tab1:
        df_metrics = pd.read_csv("datas/resultados_modelos.csv")
        df_confusion = pd.read_csv("datas/matrizes_confusao_modelos.csv")
        
        # Container principal com duas colunas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            display_metrics(df_metrics)
        
        with col2:
            display_confusion_matrix(df_confusion)
    with tab2:
        try:
            df_series = pd.read_csv("datas/series_adc.csv")
        except FileNotFoundError:
            st.error("Arquivo 'datas/series_adc.csv' não encontrado.")
            return
        # Verificar colunas necessárias
        required_columns = ["tempo", "metodo", "sujeito", "CSF", "WM", "GM"]
        if not all(col in df_series.columns for col in required_columns):
            st.error(f"CSV deve conter as colunas: {required_columns}")
            return
        df_melted = df_series.melt(
        id_vars=['sujeito', 'tempo', 'metodo'],
        value_vars=['CSF', 'WM', 'GM'],
        var_name='tecido',
        value_name='adc'
        )

        # Dividir em colunas
        col2, col3 = st.columns([1,1])

        # --- Coluna 2 (Séries Temporais de ADC) ---
        with col2:
            st.markdown("### Séries Temporais de ADC")

            col_sel1, col_sel2 = st.columns([1, 1])
            
            with col_sel1:
                sujeito_selecionado = st.selectbox(
                    label="Sujeito",
                    options=df_series['sujeito'].unique(),
                    key='sujeito_selector',
                    label_visibility="collapsed"
                )

            with col_sel2:
                tecido_selecionado = st.selectbox(
                    label="Tecido",
                    options=['CSF', 'WM', 'GM'],
                    key='tecido_selector',
                    label_visibility="collapsed"
                )

            # Filtrar dados
            df_filtrado = df_melted[
                (df_melted['sujeito'] == sujeito_selecionado) &
                (df_melted['tecido'] == tecido_selecionado)
            ]

            # Criar gráfico com paleta personalizada
            fig = px.line(
                df_filtrado,
                x='tempo',
                y='adc',
                color='metodo',
                color_discrete_sequence=['red'],  # Cor padrão vermelha
                color_discrete_map={
                    'GMM': '#440154',  # Roxo escuro
                    'MLP': '#21918c',  # Verde-azulado
                    'XGB': '#fde725'  # Amarelo
                },
                title=f'Série de ADC - Sujeito {sujeito_selecionado} | Tecido {tecido_selecionado}',
                labels={'tempo': 'Tempo', 'adc': 'Valor ADC', 'metodo': 'Método'}
            )

            # Ajustes no layout
            fig.update_layout(
                hovermode='x unified',
                legend_title_text='Método',
                xaxis_title='Tempo (s)',
                yaxis_title='ADC (mm²/s)',
                plot_bgcolor='rgba(0,0,0,0)',  # Fundo transparente
                paper_bgcolor='rgba(0,0,0,0)'  # Fundo transparente
            )
            
            # Personalizar linha de grid
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')

            st.plotly_chart(fig, use_container_width=True)
        with col3:
            st.markdown("### Variação de correlação entre séries do modelo e padrão")
            # Carregar os dados (exemplo com dados fictícios)
            try:
                # Simulando a leitura do CSV - substitua por pd.read_csv("correlacoes_adc_pivo.csv")
                data = {
                    'model': ['GMM', 'GMM', 'MLP', 'MLP', 'XGB', 'XGB'],
                    'subject': ['sub01', 'sub02', 'sub01', 'sub02', 'sub01', 'sub02'],
                    'CSF': [-0.016272, 0.364770, 0.260044, 0.321358, -0.099664, 0.656352],
                    'GM': [0.933462, 0.750561, -0.040625, 0.295746, 0.748532, 0.812538],
                    'WM': [0.833970, 0.932338, 0.705744, 0.900663, 0.800396, 0.942485]
                }
                # df = pd.DataFrame(data).set_index(['model', 'subject'])
                df = pd.read_csv("datas/correlacoes_adc_pivot.csv")
            except Exception as e:
                st.error(f"Erro ao carregar dados: {str(e)}")
                return

            # Transformar os dados para formato longo (tidy)
            df_reset = df.reset_index()
            df_long = pd.melt(df_reset, 
                                id_vars=['model', 'subject'],
                                value_vars=['CSF', 'GM', 'WM'],
                                var_name='tissue',
                                value_name='correlation')

            # Criar o boxplot interativo
            fig = px.box(df_long, 
                            x='tissue', 
                            y='correlation',
                            color='model',
                            color_discrete_map={
                                'GMM': '#440154',  # Cores da paleta Viridis
                                'MLP': '#21918c',
                                'XGB': '#fde725'
                            },
                            title='Variação da Correlação por Tecido e Método',
                            labels={
                                'tissue': 'Tecido Cerebral',
                                'correlation': 'Coeficiente de Correlação',
                                'model': 'Método'
                            },
                            hover_data=['subject'])

            # Melhorar a formatação do gráfico
            fig.update_layout(
                boxmode='group',
                yaxis_range=[-0.2, 1.1],
                hovermode='x unified',
                legend_title_text='Método',
                xaxis_title='Tecido',
                yaxis_title='Correlação',
                margin=dict(l=20, r=20, t=60, b=20)
            )

            # Adicionar linha de referência em y=0
            fig.add_hline(y=0, line_dash="dot", line_color="gray")

            # Mostrar o gráfico
            st.plotly_chart(fig, use_container_width=True)

            # Adicionar tabela com os dados brutos
            with st.expander("📊 Estatísticas Descritivas"):
                stats = df_long.groupby(['model', 'tissue'])['correlation'].describe()
                st.dataframe(stats.unstack(level=0).style.format("{:.3f}"))
            
def create_summary_chart():
    # Carregar os dados do CSV
    try:
        df_metrics = pd.read_csv("datas/resultados_modelos.csv")
        
        # Processar os dados para obter as métricas médias por modelo e tecido
        summary = df_metrics[~df_metrics['classe'].str.contains('avg|accuracy')].groupby('modelo').agg({
            'precision': 'mean',
            'recall': 'mean',
            'f1-score': 'mean'
        }).reset_index()
        
        # Obter acurácia (extrair dos registros que contêm 'accuracy')
        accuracy = df_metrics[df_metrics['classe'] == 'accuracy'].groupby('modelo')['f1-score'].mean().reset_index()
        accuracy = accuracy.rename(columns={'f1-score': 'accuracy'})
        
        # Juntar os dados
        df = pd.merge(summary, accuracy, on='modelo')
        
        # Renomear para o gráfico
        df = df.rename(columns={
            'modelo': 'Modelo',
            'f1-score': 'F1-Score',
            'precision': 'Precisão',
            'recall': 'Recall',
            'accuracy': 'Acurácia'
        })
        
        # Criar o gráfico de barras agrupadas
        fig = px.bar(df, 
                     x='Modelo', 
                     y=['Precisão', 'Recall', 'F1-Score'],
                     barmode='group',
                     color_discrete_map={
                         'Precisão': '#440154',  # Roxo escuro
                         'Recall': '#21918c',    # Verde-azulado
                         'F1-Score': '#fde725'   # Amarelo
                     },
                     labels={'value': 'Valor', 'variable': 'Métrica'},
                     )
        
        # Configurações do layout
        fig.update_layout(
            yaxis_range=[0, 1],
            hovermode='x unified',
            legend_title_text='Métricas',
            xaxis_title='Modelo',
            yaxis_title='Valor Médio',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Adicionar acurácia como linha
        fig.add_scatter(
            x=df['Modelo'], 
            y=df['Acurácia'],
            mode='markers+lines',
            name='Acurácia',
            line=dict(color='red', width=3),
            marker=dict(size=10, symbol='diamond'))
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao carregar dados para o gráfico de resumo: {str(e)}")
        return px.bar()  # Retorna um gráfico vazio em caso de erro
def menu_conclusoes():  
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.markdown("""
        ##### Desempenho Comparativo dos Modelos
        - GMM: Excelente para GM/WM (corr. até 0.93), mas pobre em CSF
        - MLP: Melhor estrutura anatômica no WM, mas inconsistente com GM
        -XGBoost:  Mais equilibrado (acurácia 53%) melhor em CSF (corr. 0.65)
        """)
        # Gráfico resumo de desempenho
        st.plotly_chart(create_summary_chart(), use_container_width=True)

    with col2:
        col21, col22 = st.columns([1,1])
        with col21:
            st.markdown("""
                ##### Fatores Limitantes
                - Tamanho da amostra: 3 sujeitos (inicialmente previstos 5)
                -- Problemas técnicos:Manutenção do equipamento Philips impossibilitou novas aquisições
                - Resolução espacial: Limitações de aquisição EPI
                - Variabilidade inter-sujeito: Posicionamento de cortes
            """)
    
        with col22:
            # with col11:
            st.markdown("""
            ##### Direções Futuras
            - Arquiteturas Híbridas: Combinação de modelos
            - Redes Neurais Profundas: CNNs para capturar padrões complexos
            - Dados Adicionais: Ampliar base
            """)
            st.markdown("""
                ##### Melhorias Processuais
                - Integração T1: Uso da imagem estrutural no espaço de difusão
                - Pré-processamento: Avaliat outras técnicas de correção (eddy do FSL, por exemplo)
                """)
                
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <p style="font-size: 1.1em;"><b>Conclusão:</b> O XGBoost emerge como a abordagem mais promissora para análises equilibradas, enquanto o GMM é ideal para estudos focados em GM/WM.</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide")  # Layout mais largo para melhor visualização
    # CSS para reduzir o espaço superior do título
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem !important;  /* Reduz o espaço superior */
            }
        </style>
    """, unsafe_allow_html=True)
    # TÍTULO no topo em todas as abas
    st.title("Análise por clusterização de tecidos em mapas de ADC")

    # Segunda linha: Menu interativo de abas
    aba = option_menu(
        menu_title=None,
        options=["Descrição","Objetivos", "Dados", "Análises", "Conclusões"],
        icons = ["info-circle", "bullseye", "folder", "bar-chart", "check-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#005580", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "color": "black",  # <--- color del texto normal
                "--hover-color": "#eee",
                "padding": "10px",
                "margin": "0px",
            },
            "nav-link-selected": {"background-color": "#cceeff"},
        }
    )

    # Conteúdo principal baseado na aba selecionada
    if aba == "Descrição":
        menu_descricao()
    elif aba == "Objetivos":
        menu_objetivos()
    elif aba == "Dados":
        menu_dados()
    elif aba == "Análises":
        menu_analises()
    elif aba == "Conclusões":
        menu_conclusoes()

if __name__ == "__main__":
    main()