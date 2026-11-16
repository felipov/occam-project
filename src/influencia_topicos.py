"""
Influência entre tópicos do BERTopic: sobreposição, substituição,
antecedência e co-ocorrência.

Uso (ordem importa: a sobreposição define quais tópicos juntar):

    from influencia_topicos import *

    # docs, timestamps e msgs vêm do MESMO DataFrame, na mesma ordem do treino
    df = pd.DataFrame({
        "topic": topic_model.topics_,              # do ajuste, nunca do transform
        "date":  pd.to_datetime(timestamps),
        "msg":   msgs,                             # original_message_id de cada doc
    })

    pares_sob, grupos = sobreposicao(topic_model, limiar=0.80)
    cont, share, clr  = preparar(df, grupos, freq="MS")
    topicos = maiores(cont, n=15)

    subs = substituicao(clr, topicos)
    ante = antecedencia(clr, topicos, max_lag=3)
    coo  = coocorrencia(df, grupos)
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 0. Sobreposição semântica (rode primeiro)
# ------------------------------------------------------------

def sobreposicao(topic_model, limiar=0.80, top_n_words=10):
    """
    Similaridade entre tópicos (embeddings + palavras-chave).

    Devolve (pares, grupos). `grupos` mapeia cada tópico para o menor id do
    seu grupo, só para grupos com mais de um tópico. Use em `preparar`.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    ids_ordem = sorted(topic_model.get_topics())            # inclui -1 se existir
    linha = {t: i for i, t in enumerate(ids_ordem)}
    ids = [t for t in ids_ordem if t != -1]

    emb = np.array(topic_model.topic_embeddings_)[[linha[t] for t in ids]]
    sim = cosine_similarity(emb)

    palavras = {t: {w for w, _ in topic_model.get_topic(t)[:top_n_words]}
                for t in ids}

    linhas = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            inter = len(palavras[a] & palavras[b])
            uni = len(palavras[a] | palavras[b])
            linhas.append((a, b, sim[i, j], inter / uni if uni else 0.0))
    pares = (pd.DataFrame(linhas, columns=["A", "B", "similaridade", "jaccard_palavras"])
               .sort_values("similaridade", ascending=False)
               .reset_index(drop=True))

    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    try:
        agg = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                      linkage="average",
                                      distance_threshold=1 - limiar)
    except TypeError:                                        # sklearn < 1.2
        agg = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                      linkage="average",
                                      distance_threshold=1 - limiar)
    lab = agg.fit_predict(dist)

    grupos = {}
    for g in set(lab):
        membros = [ids[k] for k in np.where(lab == g)[0]]
        if len(membros) > 1:
            for m in membros:
                grupos[m] = min(membros)
    return pares, grupos


# ------------------------------------------------------------
# 1. Preparação: contagens, participação e CLR por período
# ------------------------------------------------------------

def preparar(df, grupos=None, freq="W", pseudo=0.5):
    """Devolve (cont, share, clr).

    cont  : documentos por período x tópico (outliers fora, grupos somados)
    share : participação de cada tópico no período (soma 1)
    clr   : log-razão centrada, a escala certa para correlações entre
            participações (evita a correlação negativa artificial)
    """
    d = df[df.topic != -1].copy()
    d["t"] = d.topic.map(lambda x: (grupos or {}).get(x, x))

    cont = (d.groupby([pd.Grouper(key="date", freq=freq), "t"])
              .size().unstack(fill_value=0))
    cont = cont.reindex(pd.date_range(cont.index.min(), cont.index.max(), freq=freq),
                        fill_value=0)

    vazios = int((cont.sum(axis=1) == 0).sum())
    if vazios:
        print(f"[aviso] {vazios} período(s) sem nenhum documento foram removidos.")
        cont = cont[cont.sum(axis=1) > 0]

    share = cont.div(cont.sum(axis=1), axis=0)

    adj = cont + pseudo
    logp = np.log(adj.div(adj.sum(axis=1), axis=0))
    clr = logp.sub(logp.mean(axis=1), axis=0)

    print(f"[info] {len(cont)} períodos, {cont.shape[1]} tópicos/grupos.")
    if len(cont) < 30:
        print("[aviso] menos de 30 períodos: trate testes e correlações como "
              "indicativos e confie mais nos gráficos.")
    return cont, share, clr


def maiores(cont, n=15):
    return cont.sum().sort_values(ascending=False).head(n).index.tolist()


# ------------------------------------------------------------
# 2. Substituição (canibalização)
# ------------------------------------------------------------

def substituicao(clr, topicos, k=20):
    """Correlação das variações (diff do CLR). Negativa = um sobe quando o
    outro desce. Devolve os k pares mais negativos e os k mais positivos."""
    d = clr[topicos].diff().dropna()
    c = d.corr()
    linhas = [(a, b, c.loc[a, b], len(d))
              for i, a in enumerate(topicos) for b in topicos[i + 1:]]
    out = (pd.DataFrame(linhas, columns=["A", "B", "corr_diff", "n_periodos"])
             .sort_values("corr_diff").reset_index(drop=True))
    return {"mais_negativos": out.head(k),
            "mais_positivos": out.tail(k).iloc[::-1].reset_index(drop=True),
            "matriz": c}


# ------------------------------------------------------------
# 3. Antecedência
# ------------------------------------------------------------

def antecedencia(clr, topicos, max_lag=3, granger=True):
    """Para cada par ordenado (A, B): melhor defasagem em que a variação de A
    correlaciona com a de B `lag` períodos depois, e o p-valor de Granger
    (A ajuda a prever B?), corrigido por defasagens (Bonferroni) e por pares
    (Benjamini-Hochberg).

    Escolher o melhor lag dentre vários infla a correlação: leia como
    pista para investigar, não como prova.
    """
    d = clr[topicos].diff().dropna()
    n = len(d)
    pode_granger = granger and n > 3 * max_lag + 10
    if granger and not pode_granger:
        print("[aviso] poucos períodos para Granger; só correlação defasada.")
    if pode_granger:
        from statsmodels.tsa.stattools import grangercausalitytests

    linhas = []
    for a in topicos:
        for b in topicos:
            if a == b:
                continue
            cs = {l: d[a].corr(d[b].shift(-l)) for l in range(1, max_lag + 1)}
            lag = max(cs, key=lambda l: abs(cs[l]) if pd.notna(cs[l]) else -1)
            p = np.nan
            if pode_granger:
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                            warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = grangercausalitytests(d[[b, a]], maxlag=max_lag)
                    p = min(1.0, min(res[l][0]["ssr_ftest"][1]
                                     for l in res) * max_lag)
                except Exception:
                    p = np.nan
            linhas.append((a, b, lag, cs[lag], p))

    out = pd.DataFrame(linhas, columns=["A_antecede", "B", "lag", "corr", "p_granger"])
    if pode_granger and out.p_granger.notna().any():
        from statsmodels.stats.multitest import multipletests
        ok = out.p_granger.notna()
        out.loc[ok, "p_ajustado"] = multipletests(out.loc[ok, "p_granger"],
                                                  method="fdr_bh")[1]
    out["abs_corr"] = out["corr"].abs()
    return out.sort_values("abs_corr", ascending=False).drop(columns="abs_corr") \
              .reset_index(drop=True)


# ------------------------------------------------------------
# 4. Co-ocorrência dentro do mesmo relatório
# ------------------------------------------------------------

def coocorrencia(df, grupos=None, min_msgs=20, k=30):
    """Pares de tópicos que aparecem juntos no mesmo relatório.

    lift  > 1: aparecem juntos mais do que o acaso.
    npmi  em [-1, 1]: força da associação, menos sensível à frequência.
    """
    d = df[df.topic != -1].copy()
    d["t"] = d.topic.map(lambda x: (grupos or {}).get(x, x))
    pres = (d.groupby(["msg", "t"]).size().unstack(fill_value=0) > 0).astype(int)
    pres = pres.loc[:, pres.sum() >= min_msgs]

    n = len(pres)
    p = pres.mean()
    co = (pres.T @ pres)
    cols = list(pres.columns)

    linhas = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pab = co.loc[a, b] / n
            if pab == 0:
                continue
            lift = pab / (p[a] * p[b])
            npmi = np.log(pab / (p[a] * p[b])) / -np.log(pab) if pab < 1 else np.nan
            linhas.append((a, b, int(co.loc[a, b]), lift, npmi))
    out = (pd.DataFrame(linhas, columns=["A", "B", "relatorios_juntos", "lift", "npmi"])
             .sort_values("npmi", ascending=False).reset_index(drop=True))
    return {"mais_associados": out.head(k),
            "menos_associados": out.tail(k).iloc[::-1].reset_index(drop=True)}