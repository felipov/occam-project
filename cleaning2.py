"""
Pipeline documentado para limpeza, representação semântica e clusterização
 de relatórios financeiros.

Fluxo geral:

    JSONL
      -> limpeza estrutural
      -> divisão dos documentos em blocos compatíveis com o tokenizer
      -> embeddings semânticos com SentenceTransformer
      -> média dos embeddings dos blocos por documento
      -> redução de dimensionalidade com UMAP
      -> agrupamento com HDBSCAN
      -> exportação da base completa
      -> geração de amostra para revisão manual
      -> exportação dos parâmetros da execução

O programa não decide automaticamente o que é "lixo". O cluster -1 do
HDBSCAN representa baixa densidade e deve ser revisado manualmente, assim
como os demais clusters.
"""

import json
# Biblioteca padrão para limpeza e identificação de padrões textuais por regex.
import re
from pathlib import Path
# HDBSCAN identifica regiões de maior densidade e atribui -1 a documentos
# que não pertencem a uma região suficientemente densa.
import hdbscan
import numpy as np
import pandas as pd
# SentenceTransformer gera embeddings semânticos dos textos.
from sentence_transformers import SentenceTransformer
# UMAP reduz a dimensionalidade dos embeddings antes do HDBSCAN.
from umap import UMAP


# ---------------------------------------------------------------------------
# Configurações globais da execução
# ---------------------------------------------------------------------------

# Modelo multilíngue usado para representar textos em português e inglês.
MODEL_NAME = 'BAAI/bge-m3'

# Seed do UMAP. Ajuda a obter resultados reproduzíveis entre execuções
# feitas com as mesmas versões das bibliotecas e os mesmos dados.
RANDOM_STATE = 10

# Seed usada para selecionar sempre a mesma amostra manual por cluster.
SAMPLE_RANDOM_STATE = 42

# Quantidade padrão de documentos selecionados para cada cluster na amostra.
DEFAULT_SAMPLE_SIZE = 3

# Quantidade de textos processados simultaneamente pelo SentenceTransformer.
# O valor deve ser reduzido se houver pouca memória disponível.
DEFAULT_BATCH_SIZE = 32

# Tamanho mínimo padrão para que o HDBSCAN forme um cluster válido.
DEFAULT_MIN_CLUSTER_SIZE = 15


# ---------------------------------------------------------------------------
# Limpeza textual
# ---------------------------------------------------------------------------


def limpar_texto_ia(texto):
    """
    Remove ruído estrutural e preserva o contexto semântico do documento.

    A função não remove stopwords gerais, pois o texto será enviado a um
    modelo Transformer. Palavras funcionais, negações e relações temporais
    podem contribuir para o significado do documento.

    Parâmetros
    ----------
    texto : object
        Valor original do campo ``summary``.

    Retorno
    -------
    str
        Texto limpo. Valores que não sejam strings retornam uma string vazia.
    """

    # Se o campo não for textual, não há conteúdo confiável para limpar.
    if not isinstance(texto, str):
        return ''

    # Remove a marca de anonimização sem remover o restante do documento.
    texto = re.sub(r'\[REDACTED\]', ' ', texto)

    # Esses rótulos são metadados estruturais em Markdown. O conteúdo que
    # aparece depois do rótulo é preservado.
    padroes_rotulos = [
        r'\*\*Source:\*\*',
        r'\*\*Focus:\*\*',
        r'\*\*General comment:\*\*',
        r'\*\*Theme:\*\*',
        r'\*\*Markets:\*\*',
    ]

    # A flag IGNORECASE permite reconhecer rótulos em diferentes combinações
    # de maiúsculas e minúsculas.
    for padrao in padroes_rotulos:
        texto = re.sub(padrao, ' ', texto, flags=re.IGNORECASE)

    # Remove somente os marcadores de cabeçalho (#, ##, ### etc.).
    # O conteúdo do título é preservado; por exemplo, "# Fed outlook"
    # torna-se "Fed outlook", e não apenas "outlook".
    texto = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', ' ', texto)

    # URLs e endereços de e-mail são removidos por serem ruído estrutural
    # para o objetivo de agrupar relatórios por tema.
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'\S+@\S+', ' ', texto)

    # Remove somente os marcadores financeiros especificados. A operação é
    # feita antes de qualquer outra normalização que possa quebrar as barras.
    # Caso esses marcadores sejam informativos para a sua análise, remova
    # esta linha ou substitua-os por tokens normalizados.
    texto = re.sub(r'\b(?:y/y|m/m|q/q)\b', ' ', texto, flags=re.IGNORECASE)

    # Converte o texto para minúsculas. Pontuação e estrutura textual são
    # preservadas para que o Transformer receba contexto suficiente.
    texto = texto.lower()

    # Converte sequências de espaços, quebras de linha e tabulações em um
    # único espaço e remove espaços no início e no fim.
    return re.sub(r'\s+', ' ', texto).strip()


# ---------------------------------------------------------------------------
# Divisão dos documentos por tokens
# ---------------------------------------------------------------------------


def _blocos_por_tokenizer(
    modelo,
    texto,
    max_tokens_por_bloco=512,
    margem_tokens=2,
):
    """
    Divide um documento em blocos com base no tokenizer do próprio modelo.

    A função não divide o texto por quantidade aproximada de palavras. Em vez
    disso, converte o texto em IDs de tokens usando o tokenizer associado ao
    SentenceTransformer e depois reconstrói blocos que respeitam o limite de
    entrada do modelo.

    Isso é importante porque uma palavra pode gerar vários tokens, sobretudo
    em textos com números, siglas, abreviações, palavras acentuadas e jargão
    financeiro. Portanto, uma divisão fixa por palavras não garante que o
    limite de tokens será respeitado.

    Parâmetros
    ----------
    modelo : SentenceTransformer
        Modelo SentenceTransformer já carregado. A função utiliza:

        - ``modelo.tokenizer`` para converter texto em tokens;
        - ``modelo.max_seq_length`` para conhecer o limite da entrada.

    texto : str
        Documento textual que será dividido.

    max_tokens_por_bloco : int, default=512
        Limite desejado de tokens por bloco. Esse valor pode ser menor que o
        limite máximo do modelo. Para clusterização temática, blocos menores
        ajudam a evitar que vários subtemas de um relatório sejam misturados
        em um único embedding.

    margem_tokens : int, default=2
        Quantidade de tokens reservada para tokens especiais adicionados pelo
        modelo, como tokens de início e fim de sequência.

    Retorno
    -------
    list[str]
        Lista de blocos textuais. Cada bloco está abaixo do limite calculado
        e pode ser enviado ao método ``modelo.encode``.

    Raises
    ------
    ValueError
        Se os parâmetros forem inválidos ou se o modelo não informar um limite
        de sequência utilizável.
    """

    # Valida o limite solicitado pelo usuário.
    if max_tokens_por_bloco <= 0:
        raise ValueError(
            'max_tokens_por_bloco deve ser maior que zero.'
        )

    # A margem não pode ser negativa. Uma margem negativa aumentaria o limite
    # do bloco e poderia causar truncamento durante o encoding.
    if margem_tokens < 0:
        raise ValueError(
            'margem_tokens não pode ser negativo.'
        )

    # Obtém o tokenizer diretamente do modelo. Assim, a divisão utiliza a
    # mesma tokenização que será utilizada posteriormente pelo encode.
    tokenizer = getattr(modelo, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError(
            'O modelo não possui um tokenizer acessível.'
        )

    # Obtém o limite máximo de sequência informado pelo SentenceTransformer.
    max_seq_length = getattr(modelo, 'max_seq_length', None)
    if max_seq_length is None:
        raise ValueError(
            'O modelo não informa max_seq_length.'
        )

    max_seq_length = int(max_seq_length)

    if max_seq_length <= 0:
        raise ValueError(
            f'max_seq_length inválido: {max_seq_length}.'
        )

    # O limite seguro é o menor valor entre:
    #   1. o limite desejado para o bloco; e
    #   2. o limite máximo do modelo menos a margem de segurança.
    limite_seguro_modelo = max_seq_length - margem_tokens
    tamanho_bloco = min(
        max_tokens_por_bloco,
        limite_seguro_modelo,
    )

    # É necessário deixar pelo menos um token útil no bloco.
    if tamanho_bloco < 1:
        raise ValueError(
            'A margem de tokens é maior ou igual ao limite do modelo.'
        )

    # Converte o texto em IDs sem adicionar tokens especiais. Os tokens
    # especiais, se necessários, serão tratados pelo próprio modelo durante
    # a chamada a encode.
    token_ids = tokenizer.encode(
        texto,
        add_special_tokens=False,
    )

    # Textos vazios não geram blocos.
    if not token_ids:
        return []

    blocos = []

    # Percorre os tokens em fatias sem ultrapassar tamanho_bloco.
    for inicio in range(0, len(token_ids), tamanho_bloco):
        fim = inicio + tamanho_bloco
        ids_do_bloco = token_ids[inicio:fim]

        # Converte novamente os IDs para texto. O SentenceTransformer receberá
        # esse texto e fará a tokenização final durante modelo.encode().
        bloco = tokenizer.decode(
            ids_do_bloco,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        # Evita adicionar blocos que ficaram vazios após a decodificação.
        if bloco:
            blocos.append(bloco)

    return blocos




# ---------------------------------------------------------------------------
# Geração e agregação dos embeddings
# ---------------------------------------------------------------------------


def embedding_documentos_longos(
    modelo,
    documentos,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Gera um embedding por documento a partir de blocos limitados por tokens.

    Cada documento pode originar vários blocos. O modelo gera um embedding
    normalizado para cada bloco; em seguida, os embeddings dos blocos de um
    mesmo documento são promediados e normalizados novamente.

    Parâmetros
    ----------
    modelo : SentenceTransformer
        Modelo de embeddings já carregado.
    documentos : list[str]
        Textos limpos que serão representados.
    batch_size : int, default=32
        Número de blocos enviados simultaneamente ao modelo.

    Retorno
    -------
    numpy.ndarray
        Matriz com uma linha por documento e uma coluna por dimensão do
        embedding.
    """

    # Um batch size zero ou negativo não é válido para o encode.
    if batch_size <= 0:
        raise ValueError('batch_size deve ser maior que zero.')

    # Lista global de todos os blocos de todos os documentos.
    todos_os_blocos = []

    # Para cada bloco, registra o índice do documento de origem. Isso permite
    # reconstruir um vetor único por documento depois do encode.
    documento_dos_blocos = []

    # Obtém a dimensão do embedding para criar vetores nulos em casos vazios.
    dimensao = modelo.get_sentence_embedding_dimension()

    # Cria os blocos de cada documento usando o tokenizer do modelo.
    for indice_documento, texto in enumerate(documentos):
        blocos = _blocos_por_tokenizer(modelo, texto, max_tokens_por_bloco=512, margem_tokens=2)

        # Textos vazios são ignorados nesta etapa. Em condições normais,
        # eles já foram removidos por carregar_e_limpar_dados.
        if not blocos:
            continue

        # Adiciona os blocos à lista global.
        todos_os_blocos.extend(blocos)

        # Registra o documento proprietário de cada bloco.
        documento_dos_blocos.extend(
            [indice_documento] * len(blocos)
        )

    # Se nenhum bloco foi criado, não é possível gerar embeddings.
    if not todos_os_blocos:
        raise ValueError('Nenhum bloco válido foi criado para os embeddings.')

    # Gera embeddings em lote. A normalização facilita comparações baseadas
    # em cosseno e reduz o efeito de diferenças de magnitude entre vetores.
    embeddings_blocos = modelo.encode(
        todos_os_blocos,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings_documentos = []

    # Reconstrói um embedding para cada documento original, na mesma ordem.
    for indice_documento in range(len(documentos)):
        # Localiza os blocos pertencentes ao documento atual.
        indices = [
            i
            for i, dono in enumerate(documento_dos_blocos)
            if dono == indice_documento
        ]

        # Se o documento não gerou blocos, usa vetor nulo. Esse caso é
        # defensivo; a limpeza normalmente impede que ele ocorra.
        if not indices:
            embedding = np.zeros(dimensao, dtype=np.float32)
        else:
            # Calcula a média dos embeddings dos blocos do documento.
            embedding = np.mean(embeddings_blocos[indices], axis=0)

            # Normaliza novamente o vetor médio.
            norma = np.linalg.norm(embedding)
            if norma > 0:
                embedding = embedding / norma

        embeddings_documentos.append(embedding)

    # Empilha os vetores em uma matriz: uma linha por documento.
    return np.vstack(embeddings_documentos)


# ---------------------------------------------------------------------------
# Leitura e limpeza do dataset
# ---------------------------------------------------------------------------


def carregar_e_limpar_dados(caminho_arquivo):
    """
    Carrega o JSONL, valida rastreabilidade e aplica a limpeza textual.

    O arquivo precisa conter um objeto JSON por linha e possuir, no mínimo,
    as colunas ``message_id`` e ``summary``.
    """

    dados = []

    # Abre o JSONL explicitamente em UTF-8 para preservar acentos e símbolos.
    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        # enumerate permite informar a linha exata quando houver JSON inválido.
        for numero_linha, linha in enumerate(arquivo, start=1):
            # Linhas vazias são ignoradas.
            if not linha.strip():
                continue

            try:
                # Cada linha deve conter um objeto JSON independente.
                dados.append(json.loads(linha))
            except json.JSONDecodeError as erro:
                # Repassa um erro mais informativo para facilitar a correção
                # do arquivo de entrada.
                raise ValueError(
                    f'JSON inválido na linha {numero_linha}: {erro}'
                ) from erro

    # Converte os registros para um DataFrame.
    df = pd.DataFrame(dados)

    # message_id é necessário para rastrear decisões manuais; summary é o
    # campo que contém o texto principal a ser limpo e embutido.
    colunas_essenciais = ['message_id', 'summary']
    faltantes = [col for col in colunas_essenciais if col not in df.columns]
    if faltantes:
        raise ValueError(
            f'Colunas obrigatórias ausentes no dataset: {faltantes}'
        )

    # Impede identificadores ausentes, que comprometeriam a auditoria.
    if df['message_id'].isna().any():
        raise ValueError('Há message_id ausentes no dataset.')

    # Impede IDs duplicados, que poderiam fazer uma decisão manual atingir
    # mais de um documento.
    if df['message_id'].duplicated().any():
        raise ValueError('Há message_id duplicados no dataset.')

    # Copia o DataFrame antes de adicionar ou filtrar colunas.
    df = df.copy()

    # Aplica a limpeza à coluna summary e cria a coluna texto_limpo.
    df['texto_limpo'] = df['summary'].apply(limpar_texto_ia)

    # Remove documentos que ficaram sem conteúdo após a limpeza.
    df = df[df['texto_limpo'].str.strip() != ''].copy()

    # Interrompe com mensagem clara se não houver documentos aproveitáveis.
    if df.empty:
        raise ValueError('Nenhum documento válido após a limpeza.')

    return df


# ---------------------------------------------------------------------------
# Embeddings, UMAP e HDBSCAN
# ---------------------------------------------------------------------------


def aplicar_umap_hdbscan(
    df,
    modelo_nome=MODEL_NAME,
    batch_size=DEFAULT_BATCH_SIZE,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
):
    """
    Aplica embeddings semânticos, UMAP adaptativo e HDBSCAN.

    O HDBSCAN recebe os cinco ou menos componentes produzidos pelo UMAP,
    e não o texto bruto nem diretamente a matriz original de embeddings.
    """

    # UMAP precisa de uma quantidade mínima de documentos para formar uma
    # representação de vizinhança útil.
    if len(df) < 3:
        raise ValueError('São necessários pelo menos três documentos válidos.')

    # min_cluster_size deve ser um inteiro positivo.
    if min_cluster_size <= 0:
        raise ValueError('min_cluster_size deve ser maior que zero.')

    # Converte o texto da coluna para uma lista na ordem do DataFrame.
    documentos = df['texto_limpo'].tolist()

    print('   -> Carregando modelo e gerando embeddings por tokens...')

    # O modelo pode ser baixado na primeira execução se não estiver no cache.
    modelo = SentenceTransformer(modelo_nome)

    # Gera um embedding agregado para cada documento.
    embeddings = embedding_documentos_longos(
        modelo,
        documentos,
        batch_size=batch_size,
    )

    # Para bases pequenas, reduz n_neighbors para não exceder o número de
    # documentos disponíveis. No dataset principal, o valor permanece 15.
    n_neighbors = min(15, len(df) - 1)

    # Reduz n_components em bases muito pequenas; para bases grandes, usa 5.
    n_components = min(5, len(df) - 1)

    print(
        f'   -> UMAP com n_neighbors={n_neighbors}, '
        f'n_components={n_components}...'
    )

    # Reduz a dimensionalidade usando distância cosseno, apropriada para
    # embeddings semânticos normalizados.
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=RANDOM_STATE,
    )
    embeddings_reduzidos = umap_model.fit_transform(embeddings)

    # Evita solicitar min_cluster_size maior que a própria base em datasets
    # pequenos. No dataset principal, o parâmetro permanece 15.
    tamanho_cluster = min(min_cluster_size, len(df))

    print(
        f'   -> HDBSCAN com min_cluster_size={tamanho_cluster}...'
    )

    # HDBSCAN agrupa os pontos pela densidade. Documentos sem uma atribuição
    # suficientemente confiável podem receber o rótulo -1.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=tamanho_cluster,
        metric='euclidean',
        cluster_selection_method='eom',
    )

    # Copia o DataFrame para não modificar silenciosamente o objeto recebido.
    resultado = df.copy()

    # Atribui o rótulo do cluster a cada documento.
    resultado['cluster'] = clusterer.fit_predict(embeddings_reduzidos)

    return resultado


# ---------------------------------------------------------------------------
# Amostragem para auditoria manual
# ---------------------------------------------------------------------------


def gerar_amostra_para_auditoria(
    df,
    n_amostras=DEFAULT_SAMPLE_SIZE,
):
    """
    Seleciona uma amostra reproduzível de cada cluster.

    A seleção é feita sem ``groupby.apply`` para reduzir diferenças de
    comportamento entre versões do pandas. O cluster -1 também é incluído.
    """

    # A quantidade de amostras precisa ser positiva.
    if n_amostras <= 0:
        raise ValueError('n_amostras deve ser maior que zero.')

    # Sem a coluna cluster, não há como separar os grupos.
    if 'cluster' not in df.columns:
        raise ValueError('A coluna cluster é obrigatória para a amostragem.')

    partes = []

    # Obtém os índices de cada grupo. sort=True mantém uma ordem estável
    # dos rótulos na saída, incluindo o cluster -1 antes dos demais.
    for cluster, indices in df.groupby('cluster', sort=True).groups.items():
        # Seleciona os registros pertencentes ao cluster atual.
        grupo = df.loc[indices]

        # Para clusters menores que n_amostras, seleciona todos os registros.
        quantidade = min(len(grupo), n_amostras)

        # random_state fixo torna a seleção reproduzível.
        partes.append(
            grupo.sample(
                n=quantidade,
                random_state=SAMPLE_RANDOM_STATE,
            )
        )

    # Uma base sem clusters não pode gerar amostra.
    if not partes:
        raise ValueError('Nenhum cluster disponível para amostragem.')

    # Concatena as amostras e renumera o índice do CSV final.
    amostras = pd.concat(partes, axis=0).reset_index(drop=True)

    # Essas colunas são essenciais para a revisão e posterior aplicação das
    # decisões. A função falha explicitamente se alguma estiver ausente.
    colunas_obrigatorias = [
        'message_id',
        'cluster',
        'subject',
        'summary',
        'texto_limpo',
    ]
    faltantes = [col for col in colunas_obrigatorias if col not in amostras.columns]
    if faltantes:
        raise ValueError(
            f'Colunas obrigatórias ausentes na amostra: {faltantes}'
        )

    # Define a ordem das colunas úteis para leitura no Excel ou em outro
    # programa de revisão manual. Metadados opcionais são incluídos quando
    # existem na base.
    colunas_desejadas = [
        'message_id',
        'cluster',
        'subject',
        'source_tag',
        'specialist',
        'summary',
        'texto_limpo',
    ]
    colunas_foco = [col for col in colunas_desejadas if col in amostras.columns]

    # Cria a saída final da amostra.
    amostra_final = amostras[colunas_foco].copy()

    # Campos preenchidos manualmente pelo revisor. Sugestão de valores para
    # decisao: manter, remover ou duvidoso.
    amostra_final['decisao'] = ''
    amostra_final['justificativa'] = ''

    # Campo destinado a avaliar a coerência do cluster, separadamente da
    # decisão sobre cada documento.
    amostra_final['observacao_cluster'] = ''

    return amostra_final


# ---------------------------------------------------------------------------
# Registro dos parâmetros da execução
# ---------------------------------------------------------------------------


def salvar_parametros(
    caminho,
    modelo_nome,
    batch_size,
    min_cluster_size,
    n_amostras,
):
    """
    Salva os principais parâmetros usados na execução em JSON.

    Esse arquivo não substitui um ambiente congelado, mas facilita a
    comparação entre diferentes rodadas do pipeline.
    """

    parametros = {
        'modelo_embeddings': modelo_nome,
        'batch_size': batch_size,
        'min_cluster_size': min_cluster_size,
        'n_amostras_por_cluster': n_amostras,
        'random_state_umap': RANDOM_STATE,
        'random_state_amostragem': SAMPLE_RANDOM_STATE,
        'pipeline': 'limpeza -> embeddings por tokens -> UMAP -> HDBSCAN',
    }

    # ensure_ascii=False preserva caracteres acentuados no JSON.
    Path(caminho).write_text(
        json.dumps(parametros, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    # Arquivo de entrada: um objeto JSON por linha.
    caminho_do_arquivo = 'reports_redacted.jsonl'

    # Arquivo com todos os documentos e seus clusters.
    arquivo_completo = 'dataset_occam_hdbscan_completo.csv'

    # Arquivo com a amostra selecionada para revisão manual.
    arquivo_amostra = 'amostra_revisao_manual.csv'

    # Arquivo que registra modelo e parâmetros da rodada.
    arquivo_parametros = 'parametros_execucao.json'

    print('Iniciando leitura e limpeza...')

    # Carrega os relatórios, valida IDs e cria texto_limpo.
    df_limpo = carregar_e_limpar_dados(caminho_do_arquivo)
    print(f'Documentos válidos: {len(df_limpo)}')

    print('\nIniciando pipeline (embeddings -> UMAP -> HDBSCAN)...')

    # Gera embeddings, reduz a dimensionalidade e atribui clusters.
    df_final = aplicar_umap_hdbscan(df_limpo)

    print('\n=== Resumo dos clusters ===')

    # Mostra quantos documentos foram atribuídos a cada rótulo.
    contagem = df_final['cluster'].value_counts().sort_index()
    print(contagem.to_string())

    # Exporta a base completa, preservando todos os campos originais, o texto
    # limpo e o cluster atribuído.
    df_final.to_csv(
        arquivo_completo,
        index=False,
        encoding='utf-8',
    )

    # Gera três documentos por cluster por padrão, incluindo o cluster -1.
    df_amostra = gerar_amostra_para_auditoria(
        df_final,
        n_amostras=DEFAULT_SAMPLE_SIZE,
    )

    # Exporta a amostra com os campos de decisão manual vazios.
    df_amostra.to_csv(
        arquivo_amostra,
        index=False,
        encoding='utf-8',
    )

    # Salva os parâmetros principais para facilitar a auditoria da rodada.
    salvar_parametros(
        arquivo_parametros,
        modelo_nome=MODEL_NAME,
        batch_size=DEFAULT_BATCH_SIZE,
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        n_amostras=DEFAULT_SAMPLE_SIZE,
    )

    # Informa os arquivos gerados.
    print('\nArquivos gerados:')
    print(f'- {arquivo_completo}')
    print(f'- {arquivo_amostra}')
    print(f'- {arquivo_parametros}')
