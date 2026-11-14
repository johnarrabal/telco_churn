"""
TELCO CHURN PREDICTOR
Aplicação Streamlit do Projeto Integrador: predição de churn em
clientes de telecomunicações.

A aplicação carrega o pipeline completo (preparação + modelo) salvo
em models/churn_model.joblib e estima a probabilidade de cancelamento
a partir das características informadas pelo usuário.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DIR_RAIZ = Path(__file__).resolve().parent
CAMINHO_MODELO = DIR_RAIZ / "models" / "churn_model.joblib"

COLUNAS_MODELO = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

MAPA_GENERO = {"Feminino": "Female", "Masculino": "Male"}
MAPA_SIM_NAO = {"Sim": "Yes", "Não": "No"}
MAPA_CONTRATO = {
    "Mês a mês": "Month-to-month",
    "Um ano": "One year",
    "Dois anos": "Two year",
}
MAPA_INTERNET = {"DSL": "DSL", "Fibra óptica": "Fiber optic", "Não": "No"}
MAPA_PAGAMENTO = {
    "Cheque eletrônico": "Electronic check",
    "Cheque enviado": "Mailed check",
    "Transferência bancária (automática)": "Bank transfer (automatic)",
    "Cartão de crédito (automático)": "Credit card (automatic)",
}


def carregar_modelo(caminho: Path | str = CAMINHO_MODELO):
    """Carrega o pipeline completo (pré-processamento + classificador)."""
    return joblib.load(caminho)


def classificar_risco(probabilidade: float) -> str:
    """
    Converte a probabilidade em uma faixa de risco didática.

    0% a 30%  -> BAIXO
    30% a 60% -> MODERADO
    acima de 60% -> ALTO
    """
    if probabilidade <= 0.30:
        return "BAIXO"
    if probabilidade <= 0.60:
        return "MODERADO"
    return "ALTO"


def texto_interpretacao(probabilidade: float, risco: str) -> str:
    """Produz um texto de apoio à análise humana, sem recomendar ações comerciais."""
    percentual = f"{probabilidade * 100:.1f}%"
    if risco == "BAIXO":
        return (
            f"A probabilidade estimada de churn é de {percentual}, "
            "classificada como risco BAIXO. O perfil informado se aproxima "
            "de clientes que, no histórico analisado, tendem a permanecer. "
            "Ainda assim, o resultado é apenas um sinal de apoio: a decisão "
            "sobre eventual acompanhamento cabe à área responsável."
        )
    if risco == "MODERADO":
        return (
            f"A probabilidade estimada de churn é de {percentual}, "
            "classificada como risco MODERADO. Há sinais mistos no perfil "
            "informado. Vale a pena a área de retenção olhar o caso com "
            "atenção, cruzando esta estimativa com o contexto real do "
            "cliente. Esta aplicação não recomenda descontos nem ofertas."
        )
    return (
        f"A probabilidade estimada de churn é de {percentual}, "
        "classificada como risco ALTO. O perfil se aproxima de clientes "
        "que, nos dados históricos, cancelaram com mais frequência. "
        "Trata-se de um caso que merece atenção da área responsável por "
        "retenção. A previsão apoia a análise humana e não determina, "
        "automaticamente, qualquer ação comercial."
    )


def montar_entrada(
    gender: str,
    SeniorCitizen: int,
    Partner: str,
    Dependents: str,
    tenure: int,
    PhoneService: str,
    MultipleLines: str,
    InternetService: str,
    OnlineSecurity: str,
    OnlineBackup: str,
    DeviceProtection: str,
    TechSupport: str,
    StreamingTV: str,
    StreamingMovies: str,
    Contract: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: float,
) -> pd.DataFrame:
    """Monta um DataFrame de uma linha com as colunas esperadas pelo pipeline."""
    dados = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    return pd.DataFrame([dados], columns=COLUNAS_MODELO)


def prever(modelo, entrada: pd.DataFrame) -> tuple[float, str]:
    """Retorna a probabilidade da classe Churn (1) e a faixa de risco."""
    probabilidade = float(modelo.predict_proba(entrada)[0, 1])
    return probabilidade, classificar_risco(probabilidade)


def _adicionar_css() -> None:
    st.markdown(
        """
        <style>
        .risco-baixo {background:#e8f5e9;color:#1b5e20;padding:0.8rem 1rem;
                      border-radius:8px;font-size:1.4rem;font-weight:700;
                      text-align:center;border:1px solid #a5d6a7;}
        .risco-moderado {background:#fff8e1;color:#e65100;padding:0.8rem 1rem;
                         border-radius:8px;font-size:1.4rem;font-weight:700;
                         text-align:center;border:1px solid #ffe082;}
        .risco-alto {background:#ffebee;color:#b71c1c;padding:0.8rem 1rem;
                     border-radius:8px;font-size:1.4rem;font-weight:700;
                     text-align:center;border:1px solid #ef9a9a;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def executar_interface() -> None:
    st.set_page_config(
        page_title="TELCO CHURN PREDICTOR",
        page_icon="📶",
        layout="wide",
    )
    _adicionar_css()

    st.title("TELCO CHURN PREDICTOR")
    st.caption(
        "Estimativa de probabilidade de cancelamento (churn) a partir de "
        "características do cliente. Ferramenta acadêmica de apoio à análise."
    )

    modelo = carregar_modelo()

    st.header("Dados pessoais")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        genero_ui = st.selectbox("Gênero", list(MAPA_GENERO.keys()), key="genero")
    with col_b:
        idoso_ui = st.selectbox("Cliente idoso (Senior Citizen)", ["Não", "Sim"], key="idoso")
    with col_c:
        parceiro_ui = st.selectbox("Possui parceiro(a)", ["Não", "Sim"], key="parceiro")
    with col_d:
        dependentes_ui = st.selectbox("Possui dependentes", ["Não", "Sim"], key="dependentes")

    st.header("Contrato e serviços")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input(
            "Tempo como cliente (meses)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            key="tenure",
        )
        contrato_ui = st.selectbox("Tipo de contrato", list(MAPA_CONTRATO.keys()), key="contrato")
        fatura_digital_ui = st.selectbox("Fatura digital (Paperless billing)", ["Sim", "Não"], key="fatura")
        pagamento_ui = st.selectbox("Método de pagamento", list(MAPA_PAGAMENTO.keys()), key="pagamento")
    with col2:
        telefone_ui = st.selectbox("Serviço de telefone", ["Sim", "Não"], key="telefone")
        if telefone_ui == "Não":
            linhas_ui = "Sem serviço de telefone"
            st.selectbox(
                "Múltiplas linhas",
                ["Sem serviço de telefone"],
                disabled=True,
                key="linhas",
            )
        else:
            linhas_ui = st.selectbox("Múltiplas linhas", ["Não", "Sim"], key="linhas")

        internet_ui = st.selectbox(
            "Serviço de internet",
            list(MAPA_INTERNET.keys()),
            key="internet",
        )
    with col3:
        opcoes_adicional = ["Não", "Sim"]
        sem_internet = internet_ui == "Não"

        def adicional(rotulo: str, chave: str) -> str:
            if sem_internet:
                st.selectbox(
                    rotulo,
                    ["Sem serviço de internet"],
                    disabled=True,
                    key=chave,
                )
                return "Sem serviço de internet"
            return st.selectbox(rotulo, opcoes_adicional, key=chave)

        seguranca_ui = adicional("Segurança online", "seguranca")
        backup_ui = adicional("Backup online", "backup")
        protecao_ui = adicional("Proteção de dispositivo", "protecao")
        suporte_ui = adicional("Suporte técnico", "suporte")
        tv_ui = adicional("Streaming de TV", "tv")
        filmes_ui = adicional("Streaming de filmes", "filmes")

    st.header("Dados financeiros")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mensal = st.number_input(
            "Valor mensal (MonthlyCharges)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.05,
            key="mensal",
        )
    with col_f2:
        total = st.number_input(
            "Valor total acumulado (TotalCharges)",
            min_value=0.0,
            max_value=10000.0,
            value=float(min(840.0, 72 * 70.0)),
            step=0.05,
            key="total",
        )
        st.caption(
            "Para clientes com tenure igual a zero, o valor total tende a ser zero "
            "(ainda não houve faturamento acumulado)."
        )

    analisar = st.button("ANALISAR RISCO DE CHURN", type="primary", use_container_width=True, key="analisar")

    if analisar:
        def sim_nao_ou_servico(valor_ui: str, telefone: bool = False) -> str:
            if valor_ui == "Sem serviço de telefone":
                return "No phone service"
            if valor_ui == "Sem serviço de internet":
                return "No internet service"
            return MAPA_SIM_NAO[valor_ui]

        entrada = montar_entrada(
            gender=MAPA_GENERO[genero_ui],
            SeniorCitizen=1 if idoso_ui == "Sim" else 0,
            Partner=MAPA_SIM_NAO[parceiro_ui],
            Dependents=MAPA_SIM_NAO[dependentes_ui],
            tenure=int(tenure),
            PhoneService=MAPA_SIM_NAO[telefone_ui],
            MultipleLines=sim_nao_ou_servico(linhas_ui),
            InternetService=MAPA_INTERNET[internet_ui],
            OnlineSecurity=sim_nao_ou_servico(seguranca_ui),
            OnlineBackup=sim_nao_ou_servico(backup_ui),
            DeviceProtection=sim_nao_ou_servico(protecao_ui),
            TechSupport=sim_nao_ou_servico(suporte_ui),
            StreamingTV=sim_nao_ou_servico(tv_ui),
            StreamingMovies=sim_nao_ou_servico(filmes_ui),
            Contract=MAPA_CONTRATO[contrato_ui],
            PaperlessBilling=MAPA_SIM_NAO[fatura_digital_ui],
            PaymentMethod=MAPA_PAGAMENTO[pagamento_ui],
            MonthlyCharges=float(mensal),
            TotalCharges=float(total),
        )

        probabilidade, risco = prever(modelo, entrada)

        st.subheader("Resultado")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Probabilidade de churn", f"{probabilidade * 100:.1f}%")
        with col_r2:
            classe_css = {
                "BAIXO": "risco-baixo",
                "MODERADO": "risco-moderado",
                "ALTO": "risco-alto",
            }[risco]
            st.markdown(
                f'<div class="{classe_css}">Risco: {risco}</div>',
                unsafe_allow_html=True,
            )

        st.write(texto_interpretacao(probabilidade, risco))

    st.divider()
    st.markdown(
        """
**Nota importante.** Os limiares de risco (0–30% = BAIXO, 30–60% = MODERADO
e acima de 60% = ALTO) são **critérios didáticos desta aplicação** e **não
foram estabelecidos por uma operadora real**. A previsão deve **apoiar
decisões humanas** e não determina automaticamente promoções, preços ou
outras ações comerciais. O projeto tem finalidade acadêmica e demonstrativa.
        """
    )


if __name__ == "__main__":
    executar_interface()
