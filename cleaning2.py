"""
Pipeline documentado para limpeza, representação semântica e clusterização
 de relatórios financeiros.

Fluxo geral:

    JSONL
      -> limpeza estrutural
      -> divisão dos documentos em blocos compatíveis com o tokenizer
      -> embeddings semânticos com SentenceTransformer
      -> média dos embeddings dos blocos por documento
      -> redução de dimensionalidade com UMAP reprodutível por seed fixa
      -> agrupamento com HDBSCAN
      -> exportação da base completa
      -> geração de amostra para revisão manual
      -> exportação dos parâmetros da execução

O programa não decide automaticamente o que é "lixo". O cluster -1 do
HDBSCAN representa baixa densidade e deve ser revisado manualmente, assim
como os demais clusters.
"""

import json
import os

# Solicita comportamento determinístico nas operações CUDA quando possível.
# Deve ser definido antes do carregamento do PyTorch.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import re
from pathlib import Path
import hdbscan
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from sentence_transformers import SentenceTransformer

# Solicita algoritmos determinísticos no PyTorch. warn_only=True evita que uma
# operação específica sem implementação determinística interrompa o programa.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Configurações globais da execução
# ---------------------------------------------------------------------------

# Modelo somente em inglês para gerar embeddings semânticos.
# O Nomic exige trust_remote_code=True nas versões antigas do
# SentenceTransformers/Transformers.
MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'

# Prefixo recomendado pelo model card do Nomic para agrupar textos por tema.
NOMIC_TASK_PREFIX = 'clustering: '

# Seed usada somente para selecionar sempre a mesma amostra manual por cluster.
# Ela não participa da geração dos clusters.
SAMPLE_RANDOM_STATE = 42

# Seed fixa usada pelo UMAP para tornar a projeção reproduzível no mesmo
# ambiente, com as mesmas versões, hardware e ordem dos dados.
RANDOM_STATE = 42

# Parâmetros do UMAP usados antes do HDBSCAN.
UMAP_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'
DETERMINISTIC_DIMENSIONALITY_REDUCTION = 'UMAP(random_state=42)'

# Quantidade padrão de documentos selecionados para cada cluster na amostra.
DEFAULT_SAMPLE_SIZE = 3

# Quantidade de textos processados simultaneamente pelo SentenceTransformer.
# Começa em 16 para equilibrar velocidade e memória. Reduza para 8 ou 4
# se houver pouca VRAM/RAM disponível.
DEFAULT_BATCH_SIZE = 16

# Limite desejado de tokens por bloco. O Nomic aceita sequências longas,
# mas blocos de 512 tokens evitam misturar subtemas durante o clustering.
MAX_TOKENS_POR_BLOCO = 512

# Sensibilidade do HDBSCAN à densidade local. Valores menores permitem que
# mais pontos entrem em clusters, mas podem reduzir a conservadoriedade.
HDBSCAN_MIN_SAMPLES = 5

# Tamanho mínimo padrão para que o HDBSCAN forme um cluster válido.
DEFAULT_MIN_CLUSTER_SIZE = 40


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
    batch_size : int, default=16
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

    # Guarda diretamente os índices dos blocos de cada documento. Isso evita
    # percorrer todos os blocos novamente para cada documento na agregação.
    indices_por_documento = [[] for _ in documentos]

    # Obtém a dimensão do embedding para criar vetores nulos em casos vazios.
    dimensao = modelo.get_sentence_embedding_dimension()

    # Cria os blocos de cada documento usando o tokenizer do modelo.
    for indice_documento, texto in enumerate(documentos):
        blocos = _blocos_por_tokenizer(
            modelo,
            texto,
            max_tokens_por_bloco=MAX_TOKENS_POR_BLOCO,
            margem_tokens=2,
        )

        # Textos vazios são ignorados nesta etapa. Em condições normais,
        # eles já foram removidos por carregar_e_limpar_dados.
        if not blocos:
            continue

        # Registra os índices que esses blocos ocuparão na lista global.
        inicio_blocos = len(todos_os_blocos)
        indices_por_documento[indice_documento] = list(
            range(inicio_blocos, inicio_blocos + len(blocos))
        )

        # Adiciona os blocos à lista global.
        todos_os_blocos.extend(blocos)

    # Se nenhum bloco foi criado, não é possível gerar embeddings.
    if not todos_os_blocos:
        raise ValueError('Nenhum bloco válido foi criado para os embeddings.')

    # O Nomic exige um prefixo de tarefa. Para este pipeline, usamos
    # `clustering:` porque os embeddings serão agrupados por similaridade
    # temática, e não usados para busca de consultas.
    blocos_com_prefixo = [
        f'{NOMIC_TASK_PREFIX}{bloco}'
        for bloco in todos_os_blocos
    ]

    # Gera embeddings em lote. A normalização facilita comparações baseadas
    # em cosseno e reduz o efeito de diferenças de magnitude entre vetores.
    embeddings_blocos = modelo.encode(
        blocos_com_prefixo,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings_documentos = []

    # Reconstrói um embedding para cada documento original, na mesma ordem.
    for indice_documento in range(len(documentos)):
        # Recupera diretamente os índices dos blocos do documento atual.
        indices = indices_por_documento[indice_documento]

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


def expandir_registros_por_linha(dados):
    """
    Cria registros a partir das linhas materiais de ``summary``.

    O título Markdown inicial e os metadados de cabeçalho não viram documentos
    independentes. Headers de seção, como ``### Context``, são preservados
    apenas como contexto e anexados ao próximo trecho material.

    Todos os novos registros herdam os metadados do relatório original.
    ``original_message_id`` preserva a origem, enquanto ``message_id`` recebe
    um sufixo único para permitir decisões manuais por trecho.
    """
    registros_expandidos = []

    # Linhas que fazem parte do cabeçalho estrutural inicial do relatório.
    padrao_metadado = re.compile(
        r'^\s*\*{0,2}(source|focus|general comment|theme|markets|author|date)\s*:',
        flags=re.IGNORECASE,
    )
    padrao_header = re.compile(r'^\s*#{1,6}\s+')

    for registro in dados:
        registro_original = dict(registro)
        id_original = registro_original.get('message_id')
        summary = registro_original.get('summary')

        if not isinstance(summary, str):
            linhas = []
        else:
            linhas = [linha.strip() for linha in summary.splitlines()]

        partes = []
        header_pendente = ''
        inicio = True

        for linha in linhas:
            if not linha:
                continue

            # O primeiro header de nível 1 é o título do relatório. Como o
            # subject já é preservado e também enviado ao embedding, o título
            # não é criado como um documento separado. Mantemos `inicio=True`
            # para que os metadados seguintes também sejam ignorados.
            if inicio and re.match(r'^\s*#\s+', linha):
                continue

            # Metadados consecutivos ao título formam o cabeçalho inicial e
            # também não devem ser documentos independentes.
            if inicio and padrao_metadado.match(linha):
                continue

            inicio = False

            # Headers internos não são documentos: ficam pendentes e são
            # anexados ao próximo conteúdo material.
            if padrao_header.match(linha):
                header_pendente = re.sub(r'^\s*#{1,6}\s+', '', linha).strip()
                continue

            parte = f'{header_pendente}. {linha}' if header_pendente else linha
            partes.append(parte.strip())
            header_pendente = ''

        # Se o relatório só tinha cabeçalho, ele não gera registro material.
        for numero_linha, parte in enumerate(partes, start=1):
            novo_registro = dict(registro_original)
            novo_registro['original_message_id'] = id_original
            novo_registro['message_id'] = f'{id_original}__line_{numero_linha:04d}'
            novo_registro['summary_line_number'] = numero_linha
            novo_registro['summary'] = parte
            registros_expandidos.append(novo_registro)

    return registros_expandidos


def carregar_e_limpar_dados(caminho_arquivo):
    """
    Carrega o JSONL, divide cada ``summary`` por quebra de linha e limpa os
    registros resultantes.

    A quebra ``\\n`` dentro do JSON é convertida pelo ``json.loads`` em uma
    quebra real. Cada trecho vira um registro separado, com todos os campos do
    relatório original preservados e com um ID derivado único.
    """
    dados = []

    with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue
            try:
                dados.append(json.loads(linha))
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f'JSON inválido na linha {numero_linha}: {erro}'
                ) from erro

    dados_expandidos = expandir_registros_por_linha(dados)
    df = pd.DataFrame(dados_expandidos)

    colunas_essenciais = ['message_id', 'original_message_id', 'summary']
    faltantes = [col for col in colunas_essenciais if col not in df.columns]
    if faltantes:
        raise ValueError(f'Colunas obrigatórias ausentes no dataset: {faltantes}')

    if df['message_id'].isna().any():
        raise ValueError('Há message_id ausentes no dataset expandido.')
    if df['message_id'].duplicated().any():
        raise ValueError('Há message_id duplicados no dataset expandido.')

    df = df.copy()
    df['texto_limpo'] = df['summary'].apply(limpar_texto_ia)
    df = df[df['texto_limpo'].str.strip() != ''].copy()

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
    Aplica embeddings semânticos, UMAP reprodutível e HDBSCAN.

    O HDBSCAN recebe os componentes produzidos pelo UMAP, e não o texto bruto
    nem diretamente a matriz original de embeddings.
    """

    # O UMAP precisa de pelo menos três documentos para produzir uma redução
    # útil antes do agrupamento.
    if len(df) < 3:
        raise ValueError('São necessários pelo menos três documentos válidos.')

    # min_cluster_size deve ser um inteiro positivo.
    if min_cluster_size <= 0:
        raise ValueError('min_cluster_size deve ser maior que zero.')

    # Combina o subject com o trecho limpo para dar contexto aos fragmentos
    # curtos. O subject original continua preservado separadamente no CSV.
    subjects = (
        df['subject'].fillna('').astype(str).tolist()
        if 'subject' in df.columns
        else [''] * len(df)
    )
    textos_limpos = df['texto_limpo'].tolist()
    documentos = [
        (
            f'subject: {subject}. content: {texto}'
            if subject.strip()
            else f'content: {texto}'
        )
        for subject, texto in zip(subjects, textos_limpos)
    ]

    print('   -> Carregando modelo e gerando embeddings por tokens...')

    # O modelo pode ser baixado na primeira execução se não estiver no cache.
    # trust_remote_code=True é exigido pelo Nomic em versões antigas das
    # bibliotecas SentenceTransformers e Transformers.
    modelo = SentenceTransformer(
        modelo_nome,
        trust_remote_code=True,
    )

    # Gera um embedding agregado para cada documento.
    embeddings = embedding_documentos_longos(
        modelo,
        documentos,
        batch_size=batch_size,
    )

    # O UMAP preserva melhor as vizinhanças locais e a estrutura de densidade
    # dos embeddings, que será usada pelo HDBSCAN.
    n_neighbors = min(UMAP_N_NEIGHBORS, len(df) - 1)
    n_components = min(UMAP_COMPONENTS, len(df) - 1)

    print(
        f'   -> UMAP com n_neighbors={n_neighbors}, '
        f'n_components={n_components}, random_state={RANDOM_STATE}...'
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    embeddings_reduzidos = umap_model.fit_transform(embeddings)

    # Evita solicitar min_cluster_size maior que a própria base em datasets
    # pequenos. No dataset principal, o parâmetro permanece 15.
    tamanho_cluster = min(min_cluster_size, len(df))

    min_samples = min(HDBSCAN_MIN_SAMPLES, tamanho_cluster)

    print(
        f'   -> HDBSCAN com min_cluster_size={tamanho_cluster}, '
        f'min_samples={min_samples}...'
    )

    # HDBSCAN agrupa os pontos pela densidade. Documentos sem uma atribuição
    # suficientemente confiável podem receber o rótulo -1.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=tamanho_cluster,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=1,
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
        grupo = df.loc[indices].copy()

        # Um relatório original pode ter vários trechos no mesmo cluster.
        # Embaralhamos os registros e mantemos apenas o primeiro trecho de
        # cada original_message_id. Assim, a amostra representa relatórios
        # originais diferentes, e não várias linhas do mesmo relatório.
        grupo = grupo.sample(
            frac=1,
            random_state=SAMPLE_RANDOM_STATE,
        )
        grupo = grupo.drop_duplicates(
            subset=['original_message_id'],
            keep='first',
        )

        # Se houver menos relatórios originais que n_amostras, seleciona todos
        # os IDs disponíveis; caso contrário, seleciona exatamente n_amostras.
        quantidade = min(len(grupo), n_amostras)
        partes.append(grupo.head(quantidade))

    # Uma base sem clusters não pode gerar amostra.
    if not partes:
        raise ValueError('Nenhum cluster disponível para amostragem.')

    # Concatena as amostras e renumera o índice do CSV final.
    amostras = pd.concat(partes, axis=0).reset_index(drop=True)

    # Essas colunas são essenciais para a revisão e posterior aplicação das
    # decisões. A função falha explicitamente se alguma estiver ausente.
    colunas_obrigatorias = [
        'message_id',
        'original_message_id',
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
        'original_message_id',
        'summary_line_number',
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
        'prefixo_embedding': NOMIC_TASK_PREFIX,
        'batch_size': batch_size,
        'min_cluster_size': min_cluster_size,
        'min_samples_hdbscan': HDBSCAN_MIN_SAMPLES,
        'n_amostras_por_cluster': n_amostras,
        'reducao_dimensionalidade': DETERMINISTIC_DIMENSIONALITY_REDUCTION,
        'random_state_amostragem': SAMPLE_RANDOM_STATE,
        'determinismo_clusterizacao': True,
        'pipeline': 'limpeza -> embeddings por tokens -> UMAP -> HDBSCAN',
        'random_state_umap': RANDOM_STATE,
        'umap_n_neighbors': UMAP_N_NEIGHBORS,
        'umap_n_components': UMAP_COMPONENTS,
        'umap_min_dist': UMAP_MIN_DIST,
        'umap_metric': UMAP_METRIC,
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
    caminho_do_arquivo = 'dataset/reports_redacted.jsonl'

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

    # Gera embeddings, reduz a dimensionalidade com UMAP e atribui clusters.
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
