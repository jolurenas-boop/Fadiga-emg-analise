import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64
from weasyprint import HTML

# Configuração da página
st.set_page_config(page_title="Análise EMG", layout="wide")
st.title("Análise de Fadiga Eletromiográfica (MDF)")

# 1. Interface de Upload e Metadados
col1, col2, col3 = st.columns(3)

with col1:
    arquivo_txt = st.file_uploader("Upload do sinal (.txt)", type=["txt"])
with col2:
    condicao = st.selectbox("Condição", ["Controle", "Intervenção (Liberação Miofascial)"])
with col3:
    momento = st.selectbox("Momento", ["Pré-teste", "Pós-teste"])

# Variáveis globais para o relatório
if 'pdf_gerado' not in st.session_state:
    st.session_state.pdf_gerado = None

# 2. Processamento do Sinal
if arquivo_txt is not None:
    try:
        # Leitura dos dados
        sinal_bruto = np.loadtxt(arquivo_txt)
        fs = 2000 # Frequência de amostragem
        
        # Filtro Butterworth (20-450Hz)
        nyq = 0.5 * fs
        low = 20 / nyq
        high = 450 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        sinal_filtrado = signal.filtfilt(b, a, sinal_bruto)
        
        # Janelamento (1s de janela com 50% de sobreposição)
        tamanho_janela = 1 * fs
        sobreposicao = int(0.5 * fs)
        
        frequencias, tempos, espectro = signal.spectrogram(
            sinal_filtrado, fs, nperseg=tamanho_janela, noverlap=sobreposicao
        )
        
        # Cálculo da Frequência Mediana (MDF) para cada janela
        mdf_array = []
        for i in range(espectro.shape[1]):
            densidade = espectro[:, i]
            area_total = np.sum(densidade)
            area_acumulada = np.cumsum(densidade)
            # Encontra o índice onde a área acumulada atinge 50% da área total
            indice_mediana = np.where(area_acumulada >= area_total / 2)[0][0]
            mdf_array.append(frequencias[indice_mediana])
            
        mdf_array = np.array(mdf_array)
        tempo_janelas = tempos
        
        # Métricas Clínicas
        coef = np.polyfit(tempo_janelas, mdf_array, 1)
        slope = coef[0]
        tendencia = np.poly1d(coef)
        
        # Queda absoluta (Média das 3 primeiras vs 3 últimas janelas)
        mdf_inicial = np.mean(mdf_array[:3])
        mdf_final = np.mean(mdf_array[-3:])
        queda_abs = mdf_inicial - mdf_final
        
        # 3. Visualização de Resultados
        st.subheader("Declínio da Frequência Mediana (MDF)")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tempo_janelas, mdf_array, marker='o', linestyle='', color='blue', alpha=0.5, label="MDF por Janela")
        ax.plot(tempo_janelas, tendencia(tempo_janelas), color='red', linewidth=2, label=f"Slope: {slope:.2f} Hz/s")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência Mediana (Hz)")
        ax.legend()
        
        st.pyplot(fig)
        
        m1, m2 = st.columns(2)
        m1.metric("Taxa de Fadiga (Slope)", f"{slope:.2f} Hz/s")
        m2.metric("Queda Absoluta Estimada", f"{queda_abs:.2f} Hz")
        
        # 4. Geração do Relatório PDF (Weasyprint)
        st.subheader("Exportação")
        
        # Converter o gráfico para base64 para embutir no HTML
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Template HTML estruturado
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #2C3E50; border-bottom: 2px solid #34495E; padding-bottom: 10px; }}
                h2 {{ color: #2980B9; margin-top: 30px; }}
                .metric-box {{ background-color: #F8F9F9; padding: 15px; border-radius: 5px; border-left: 5px solid #3498DB; margin-bottom: 15px; }}
                .chart {{ text-align: center; margin-top: 30px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; }}
                .footer {{ margin-top: 50px; font-size: 10px; text-align: center; color: #7F8C8D; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Análise Eletromiográfica</h1>
            
            <h2>Parâmetros da Avaliação</h2>
            <ul>
                <li><strong>Condição:</strong> {condicao}</li>
                <li><strong>Momento:</strong> {momento}</li>
                <li><strong>Frequência de Amostragem:</strong> {fs} Hz</li>
            </ul>
            
            <h2>Métricas de Fadiga</h2>
            <div class="metric-box">
                <strong>Taxa de Declínio (Slope):</strong> {slope:.2f} Hz/s<br>
                <strong>Queda Absoluta da MDF:</strong> {queda_abs:.2f} Hz
            </div>
            
            <div class="chart">
                <h3>Comportamento Espectral</h3>
                <img src="data:image/png;base64,{img_base64}">
            </div>
            
            <h2>Observações Clínicas</h2>
            <p>Registro das alterações na mecânica tecidual avaliadas após os testes isométricos.</p>
            
            <div class="footer">Gerado automaticamente pelo Sistema de Análise EMG</div>
        </body>
        </html>
        """
        
        if st.button("Gerar Relatório PDF"):
            with st.spinner("A compilar o documento PDF..."):
                pdf_bytes = HTML(string=html_content).write_pdf()
                st.session_state.pdf_gerado = pdf_bytes
                
        if st.session_state.pdf_gerado:
            st.download_button(
                label="Baixar Documento PDF",
                data=st.session_state.pdf_gerado,
                file_name=f"Relatorio_EMG_{condicao}_{momento}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo. Verifique se o .txt contém apenas números organizados em uma coluna ou matriz. Detalhes: {e}")
