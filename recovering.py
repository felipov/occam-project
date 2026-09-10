import pandas as pd
import re

def limpar_padroes(texto):
    termos = [
        'landscape', 'drivers', 'unpacked', 'risks', 
        'the case', 'implications', 'context', 'outlook', 
        'general comment', 'definitive impacts'
    ]
    padrao_cabecalhos = rf"(?i)^({'|'.join(termos)})\.\s*(?:-\s*)?"
    
    texto = re.sub(padrao_cabecalhos, '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\s*\[REDACTED\]\s*', ' ', texto)

    return re.sub(r'\s{2,}', ' ', texto).strip()

# ============================================================
# AMOSTRA REVISADA
# ============================================================

df_amostra = pd.read_csv('amostra_revisada.csv', sep=';')

decisao_cluster = df_amostra[['cluster', 'decisao']].drop_duplicates(subset=['cluster'])

# ============================================================
# DATASET COMPLETO
# ============================================================

dataset_completo = pd.read_csv('dataset_occam_hdbscan_completo.csv')
dataset_completo = dataset_completo.merge(decisao_cluster, on='cluster', how='left')

linhas_inuteis = (dataset_completo['decisao'] == 'lixo').sum()
print(f'Linhas inúteis descartadas: {linhas_inuteis}')

dataset_util = dataset_completo[dataset_completo['decisao'] == 'util'].copy()
dataset_util['texto_limpo'] = dataset_util['texto_limpo'].apply(limpar_padroes)
dataset_util['num_linha'] = dataset_util['message_id'].apply(
    lambda x: int(str(x).split('__line_')[1])
)
dataset_util = dataset_util.sort_values(by=['original_message_id', 'num_linha'])

# ============================================================
# RECONSTRÓI OS TEXTOS DA AMOSTRA REVISADA
# ============================================================

df_reconstruido = (
    dataset_util
    .groupby('original_message_id', as_index=False)
    .agg(
        sender=('sender', 'first'),
        subject=('subject', 'first'),
        specialist=('specialist', 'first'),
        source_type=('source_type', 'first'),
        source_tag=('source_tag', 'first'),
        received_at=('received_at', 'first'),
        created_at=('created_at', 'first'),
        
        summary=('texto_limpo', lambda x: '\n'.join(x.astype(str)))
    )
)

df_reconstruido = df_reconstruido.rename(columns={'original_message_id': 'message_id'})

ordem_colunas = [
    'message_id', 'sender', 'subject', 'specialist', 
    'source_type', 'source_tag', 'received_at', 'created_at', 'summary'
]
df_reconstruido = df_reconstruido[ordem_colunas]

pd.set_option('display.max_colwidth', None)
print("--- VISÃO DE TABELA ---")
print(df_reconstruido[['message_id', 'summary']].head(3))

# ============================================================

print("\n\n--- (DOCUMENTO 0) ---")
print(df_reconstruido['message_id'].iloc[0])
print("-" * 50)
print(df_reconstruido['summary'].iloc[0])

print("\n\n--- (DOCUMENTO 1) ---")
print(df_reconstruido['message_id'].iloc[1])
print("-" * 50)
print(df_reconstruido['summary'].iloc[1])

df_reconstruido.to_excel('resultado/textos_limpos.xlsx', index=False)