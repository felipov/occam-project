import json
import re
import pandas as pd
from recovery import limpar_padroes


# ------------------------------------------------------------
# 1. Recuperar as linhas originais, com a posição de cada uma
# ------------------------------------------------------------

def recuperar_linhas(caminho_jsonl):
    """Devolve uma linha por trecho do relatório, com o número do
    parágrafo e o número da linha (mesma contagem usada no HDBSCAN)."""
    registros = []

    padrao_metadado = re.compile(
        r'^\s*\*{0,2}(source|focus|general comment|theme|markets|author|date)\s*:',
        flags=re.IGNORECASE
    )
    padrao_header = re.compile(r'^\s*#{1,6}\s+')

    with open(caminho_jsonl, 'r', encoding='utf-8') as arquivo:
        for linha_json in arquivo:
            if not linha_json.strip():
                continue

            registro = json.loads(linha_json)
            id_original = registro.get('message_id')
            summary = registro.get('summary')

            if not isinstance(summary, str):
                continue

            # Divide o relatório em parágrafos originais.
            paragrafos = re.split(r'\n\s*\n+', summary.strip())

            numero_paragrafo = 0
            numero_linha = 0
            inicio = True

            for paragrafo in paragrafos:
                textos = []
                header_pendente = ''

                for linha in paragrafo.splitlines():
                    linha = linha.strip()

                    if not linha:
                        continue

                    if inicio and re.match(r'^#\s+', linha):
                        continue

                    if inicio and padrao_metadado.match(linha):
                        continue

                    inicio = False

                    # Headers internos são anexados ao próximo conteúdo.
                    if padrao_header.match(linha):
                        header_pendente = re.sub(
                            r'^#{1,6}\s+', '', linha
                        ).strip()
                        continue

                    texto = (
                        f'{header_pendente}. {linha}'
                        if header_pendente else linha
                    )

                    textos.append(texto)
                    header_pendente = ''

                if not textos:
                    continue

                numero_paragrafo += 1

                for texto in textos:
                    numero_linha += 1
                    registros.append({
                        'original_message_id': id_original,
                        'paragraph_number': numero_paragrafo,
                        'summary_line_number': numero_linha,
                        'texto_linha': texto,
                    })

    return pd.DataFrame(registros)


# ------------------------------------------------------------
# 2. Recuperar as decisões dos clusters
# ------------------------------------------------------------

revisao = pd.read_csv(
    'review/reviewed_sample.csv',
    sep=';'
)

df_completo = pd.read_csv(
    'data/processed/occam_hdbscan_full.csv'
)


def primeira_decisao(coluna):
    decisoes = coluna.dropna().astype(str).str.strip()
    decisoes = decisoes[decisoes != '']
    return decisoes.iloc[0] if len(decisoes) else None


decisoes = (
    revisao.groupby('cluster')['decisao']
    .apply(primeira_decisao)
)

df_completo['decisao'] = df_completo['cluster'].map(decisoes)


# ------------------------------------------------------------
# 3. Associar as decisões a cada linha
# ------------------------------------------------------------

df_linhas = recuperar_linhas('data/raw/reports_redacted.jsonl')

decisoes_linhas = (
    df_completo[['original_message_id', 'summary_line_number', 'decisao', 'texto_limpo']]
    .drop_duplicates(['original_message_id', 'summary_line_number'])
)

df_linhas = df_linhas.merge(
    decisoes_linhas,
    on=['original_message_id', 'summary_line_number'],
    how='left'
)


# ------------------------------------------------------------
# 4. Manter só as linhas úteis e reagrupar por parágrafo
# ------------------------------------------------------------

df_util = df_linhas[df_linhas['decisao'].astype(str).str.strip().str.lower().eq('util')].copy()
df_util['texto_limpo'] = df_util['texto_limpo'].apply(limpar_padroes)
df_util = df_util[df_util['texto_limpo'].str.len() > 0]

df_final = (
    df_util
    .sort_values(['original_message_id', 'paragraph_number', 'summary_line_number'])
    .groupby(['original_message_id', 'paragraph_number'], as_index=False)
    .agg(
        texto_paragrafo=('texto_limpo', '\n'.join),
        linhas_uteis=('texto_limpo', 'size'),
    )
)


# ------------------------------------------------------------
# 5. Recuperar os metadados originais
# ------------------------------------------------------------

colunas_meta = [
    coluna for coluna in
    ['subject', 'source_tag', 'specialist', 'created_at']
    if coluna in df_completo.columns
]

metadados = df_completo.drop_duplicates('original_message_id')[
    ['original_message_id'] + colunas_meta
]

df_final = df_final.merge(
    metadados,
    on='original_message_id',
    how='left'
)

df_final = df_final[
    ['original_message_id', 'paragraph_number',
     'texto_paragrafo', 'linhas_uteis'] + colunas_meta
]

if 'created_at' in df_final.columns:
    df_final['created_at'] = (
        pd.to_datetime(df_final['created_at'], format='ISO8601', utc=True)
        .dt.tz_convert('America/Sao_Paulo')
        .dt.tz_localize(None)
        .dt.normalize()
    )


# ------------------------------------------------------------
# 6. Exportar
# ------------------------------------------------------------

df_final.to_excel('paragrafos_util.xlsx', index=False)

print(f'Parágrafos com ao menos uma linha útil: {len(df_final)}')
print(f'Linhas úteis mantidas: {int(df_final["linhas_uteis"].sum())}')
print(f'Relatórios de origem: {df_final["original_message_id"].nunique()}')
print('Arquivo exportado: paragrafos_util.xlsx')