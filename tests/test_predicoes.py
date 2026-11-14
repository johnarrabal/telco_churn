"""
Testes mínimos do Projeto Integrador — predição de churn.

1. Cliente aparentemente estável.
2. Cliente com características associadas a churn.
3. Valores extremos permitidos.
4. Todos os campos da interface / do modelo.
5. Recarga do modelo a partir do disco em um processo novo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Garante import de app.py a partir da raiz do projeto
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app import (  # noqa: E402
    CAMINHO_MODELO,
    COLUNAS_MODELO,
    carregar_modelo,
    classificar_risco,
    montar_entrada,
    prever,
)


def cliente_estavel() -> pd.DataFrame:
    """Perfil associado, na EDA, a menor risco de cancelamento."""
    return montar_entrada(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=72,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=45.0,
        TotalCharges=3200.0,
    )


def cliente_propenso() -> pd.DataFrame:
    """Perfil associado, na EDA, a maior risco de cancelamento."""
    return montar_entrada(
        gender="Male",
        SeniorCitizen=1,
        Partner="No",
        Dependents="No",
        tenure=1,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=95.0,
        TotalCharges=95.0,
    )


@pytest.fixture(scope="module")
def modelo():
    assert CAMINHO_MODELO.exists(), (
        f"Modelo não encontrado em {CAMINHO_MODELO}. "
        "Execute o caderno 02_modelagem.ipynb antes dos testes."
    )
    return carregar_modelo()


def test_cliente_estavel_tem_risco_baixo(modelo):
    """Teste 1: cliente aparentemente estável deve ter probabilidade baixa."""
    probabilidade, risco = prever(modelo, cliente_estavel())
    assert 0.0 <= probabilidade <= 1.0
    assert probabilidade <= 0.30
    assert risco == "BAIXO"
    assert classificar_risco(probabilidade) == "BAIXO"


def test_cliente_propenso_tem_risco_alto(modelo):
    """Teste 2: cliente com características associadas a churn deve ter risco alto."""
    probabilidade, risco = prever(modelo, cliente_propenso())
    assert 0.0 <= probabilidade <= 1.0
    assert probabilidade > 0.60
    assert risco == "ALTO"
    assert classificar_risco(probabilidade) == "ALTO"


def test_valores_extremos_permitidos(modelo):
    """Teste 3: extremos válidos do domínio não quebram a previsão."""
    casos = [
        montar_entrada(
            gender="Male",
            SeniorCitizen=0,
            Partner="No",
            Dependents="No",
            tenure=0,
            PhoneService="No",
            MultipleLines="No phone service",
            InternetService="No",
            OnlineSecurity="No internet service",
            OnlineBackup="No internet service",
            DeviceProtection="No internet service",
            TechSupport="No internet service",
            StreamingTV="No internet service",
            StreamingMovies="No internet service",
            Contract="Month-to-month",
            PaperlessBilling="No",
            PaymentMethod="Mailed check",
            MonthlyCharges=18.25,
            TotalCharges=0.0,
        ),
        montar_entrada(
            gender="Female",
            SeniorCitizen=1,
            Partner="Yes",
            Dependents="Yes",
            tenure=72,
            PhoneService="Yes",
            MultipleLines="Yes",
            InternetService="Fiber optic",
            OnlineSecurity="Yes",
            OnlineBackup="Yes",
            DeviceProtection="Yes",
            TechSupport="Yes",
            StreamingTV="Yes",
            StreamingMovies="Yes",
            Contract="Two year",
            PaperlessBilling="Yes",
            PaymentMethod="Bank transfer (automatic)",
            MonthlyCharges=118.75,
            TotalCharges=8500.0,
        ),
    ]
    for entrada in casos:
        probabilidade, risco = prever(modelo, entrada)
        assert 0.0 <= probabilidade <= 1.0
        assert risco in {"BAIXO", "MODERADO", "ALTO"}
        assert entrada.isna().sum().sum() == 0


def test_todos_os_campos_presentes(modelo):
    """Teste 4: a entrada contém exatamente os campos esperados pelo pipeline."""
    entrada = cliente_estavel()
    assert list(entrada.columns) == COLUNAS_MODELO
    assert entrada.shape == (1, len(COLUNAS_MODELO))
    probabilidade, risco = prever(modelo, entrada)
    assert isinstance(probabilidade, float)
    assert risco in {"BAIXO", "MODERADO", "ALTO"}

    # Também valida as faixas didáticas de risco nas fronteiras
    assert classificar_risco(0.00) == "BAIXO"
    assert classificar_risco(0.30) == "BAIXO"
    assert classificar_risco(0.3000001) == "MODERADO"
    assert classificar_risco(0.60) == "MODERADO"
    assert classificar_risco(0.6000001) == "ALTO"
    assert classificar_risco(1.00) == "ALTO"


def test_recarga_do_modelo_em_processo_novo():
    """
    Teste 5: o artefato em disco deve funcionar em um processo Python novo,
    como se o ambiente tivesse sido fechado e reaberto.
    """
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(%r)))
from app import carregar_modelo, montar_entrada, prever

modelo = carregar_modelo()
entrada = montar_entrada(
    gender="Female",
    SeniorCitizen=0,
    Partner="Yes",
    Dependents="Yes",
    tenure=72,
    PhoneService="Yes",
    MultipleLines="Yes",
    InternetService="DSL",
    OnlineSecurity="Yes",
    OnlineBackup="Yes",
    DeviceProtection="Yes",
    TechSupport="Yes",
    StreamingTV="No",
    StreamingMovies="No",
    Contract="Two year",
    PaperlessBilling="No",
    PaymentMethod="Credit card (automatic)",
    MonthlyCharges=45.0,
    TotalCharges=3200.0,
)
probabilidade, risco = prever(modelo, entrada)
assert 0.0 <= probabilidade <= 1.0, probabilidade
assert risco in {"BAIXO", "MODERADO", "ALTO"}, risco
print(f"PROCESSO_NOVO_OK {probabilidade:.6f} {risco}")
""" % str(RAIZ)
    resultado = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(RAIZ),
    )
    assert resultado.returncode == 0, (
        "Falha ao recarregar o modelo em um processo novo.\n"
        f"stdout:\n{resultado.stdout}\n"
        f"stderr:\n{resultado.stderr}"
    )
    assert "PROCESSO_NOVO_OK" in resultado.stdout
