import streamlit as st
import pandas as pd
import pickle
import os
import joblib
import altair as alt
import requests


# Estilo HTML CSS

st.markdown(
    """
    <style>
/* Ajuste na Sidebar */
[data-testid="stSidebar"] {
    background-color: #222222;
    padding: 20px;
    border-radius: 10px;
    color: #FF9100;
}

/* Melhorando os botões */
button {
    background-color: #444444 !important;
    color: #FFD700 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    padding: 10px !important;
    border: 1px solid #FF9100 !important;
}
button:hover {
    background-color: #666666 !important;
}

/* Cabeçalho mais elegante */
.header {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #FFD700;
    margin-bottom: 20px;
}

/* Imagem de fundo */
.stApp {
    background-image: url('fundo.png');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: #111111; /* Cor de fundo fallback */
}

/* Texto padrão */
body, .stText, .stMarkdown, .stTitle, .stSubtitle {
    color: #BDB9A7;
}
</style>

""", unsafe_allow_html=True
)

# Downloaded the model archive via request
model_url = 'https://github.com/sid-almeida/pc_calculator/raw/refs/heads/main/model.pkl'

response = requests.get(model_url)

# Loaded the model.pkl from files
model = pickle.loads(response.content)

# Criei um discionário com as previsões e número e depois o label que corresponde

pc_label = {0 : 'Baixo', 1 : 'Médio', 2 : 'Alto', 3 : 'Muito Alto'}

# Create a sidebar header
with st.sidebar:
    st.image("logo.png", width=250)
    st.title("Potencial de Crescimento")
    choice = st.radio("**Navegação:**", ("Sobre", "Previsão em Lote"))
    st.info('**Nota:** Por favor, esteja ciente de que este aplicativo é destinado exclusivamente para fins educacionais. É fortemente desaconselhável utilizar esta ferramenta para tomar quaisquer decisões financeiras.')


if choice == "Sobre":
    # Create a title and sub-title
    st.write("""
    # Previsão de Potencial de Crescimento Empresarial
    Este app busca prever o Potencial de Crescimento de Empresas do Continente Americano!
    """)
    st.write('---')
    st.write('**Sobre o App:**')
    st.write('Este aplicativo foi desenvolvido com o objetivo de classificar empresas com base em seu Potencial de Crescimento, utilizando o modelo de aprendizado de máquina **XGBoost Classifier**. O modelo foi treinado com um conjunto de dados abrangente, que inclui diversos parâmetros econômicos e macroeconômicos de empresas do continente americano.')
    st.info('**Nota:** Por favor, esteja ciente de que este aplicativo é destinado exclusivamente para fins educacionais. É fortemente desaconselhável utilizar esta ferramenta para tomar quaisquer decisões financeiras.')
    st.write('---')
    st.write('**Sobre os Dados**')
    st.write('O dataset utilizado neste trabalho foi desenvolvido por meio da coleta e integração de dados públicos provenientes de diversas fontes. Ele reúne uma série de parâmetros econômicos e macroeconômicos sobre empresas do continente americano, proporcionando uma visão abrangente sobre o desempenho e as características dessas organizações. A combinação dessas informações em um único conjunto de dados permite a análise de tendências econômicas e o comportamento das empresas, possibilitando a aplicação de técnicas avançadas de aprendizado de máquina para prever diversos aspectos relacionados ao seu crescimento e performance no mercado. O dataset abrange variáveis como indicadores financeiros, taxas de crescimento econômico, e outros parâmetros que impactam diretamente o setor empresarial americano.')
    # Adding a button to download the dataset data.csv
    pre_data = pd.read_csv('https://raw.githubusercontent.com/sid-almeida/pc_calculator/refs/heads/main/data.csv')
    if st.download_button(label = "Baixar Dataset de Teste", data = pre_data.to_csv(index=False), file_name = "data.csv", mime = "text/csv"):
        # Mensagem de sucesso
        st.success('O arquivo foi baixado com sucesso!')
    else:
        # Mensagem de erro
        st.write('-----------')
        st.info('Clique no botão de Download para baixar o Dataset de Teste!')


    st.write('---')

if choice == 'Previsão em Lote':
    # Create a title and sub-title
    st.write("""
    # Previsão de Potencial de Crescimento Empresarial
    Este app busca prever o Potencial de Crescimento de Empresas do Continente Americano!
    """)
    st.write('---')
    st.info('**Guia:** Por favor, envie o dataset com os parâmetros de previsão.')
    st.write('---')
    # Create a file uploader to upload the dataset of predicting features
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write('---')
        st.write('**Dataset:**')
        st.write(df_pred)
        # Create a button to predict the probability of bankruptcy
        if st.button("Prever"):
            # Predict the growth potential of the companies in the uploaded dataset
            df_pred['Potencial PC'] = model.predict(df_pred)
            # Convert 'Potencial PC' que será numérico nas labels correspondentesda 'pc_label'
            df_pred['Potencial PC'] = df_pred['Potencial PC'].map(pc_label)
            # Create a success message
            st.success('O Potencial de Crescimento foi Calculado com Sucesso!')
            st.write('---')
            st.write('**Dataset Calculado:**')
            st.write(df_pred)
            # Criei uma simples análise da coluna 'Potencial PC' no altair
            count_df = df_pred['Potencial PC'].value_counts().reset_index()
            count_df.columns = ['Potencial PC', 'Count']
            # Criando o gráfico de barras
            chart = alt.Chart(count_df).mark_bar().encode(
                x=alt.X('Potencial PC:N', title='Potencial PC'),
                y=alt.Y('Count:Q', title='Contagem'),
                color='Potencial PC:N',
                tooltip=['Potencial PC', 'Count']
            ).properties(
                title='Distribuição de Potencial de Crescimento (PC)'
            )

            # Exibindo o gráfico no Streamlit
            st.altair_chart(chart, use_container_width=True)
            # Create a button to download the dataset with the predicted probability of bankruptcy
            if st.download_button(label='Baixar o Dataset Calculado', data=df_pred.to_csv(index=False), file_name='predicted.csv', mime='text/csv'):
                pass
        else:
            st.write('---')
            st.info('Clique no botão para prever o Potencial de Crescimento!')
    else:
        st.write('---')



st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')
