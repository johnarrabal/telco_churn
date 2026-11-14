"""Teste da interface Streamlit: a aplicação sobe e o botão dispara a análise."""

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def test_botao_analisar_risco_exibe_resultado():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Erro ao iniciar a aplicação: {at.exception}"

    # Deve existir o botão com o rótulo pedido no enunciado
    rotulos = [botao.label for botao in at.button]
    assert any("ANALISAR RISCO DE CHURN" in str(rotulo) for rotulo in rotulos)

    at.button[0].click().run()
    assert not at.exception, f"Erro ao clicar no botão: {at.exception}"

    # Após o clique, algum elemento deve exibir probabilidade ou risco
    textos = " ".join(str(item.value) for item in at.metric) if at.metric else ""
    markdowns = " ".join(str(getattr(item, "value", item)) for item in at.markdown)
    combinado = (textos + " " + markdowns + " " + str(at.title)).upper()
    assert (
        "CHURN" in combinado
        or "RISCO" in combinado
        or "PROBABILIDADE" in combinado
        or at.metric
    )
