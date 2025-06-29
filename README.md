# Sistema de Ensemble Morfológico com EfficientNet para Classificação de Tecidos Tumorais Gástricos

## Visão Geral do Projeto

Este projeto implementa um sistema de classificação baseado em deep learning para análise do microambiente tumoral em lâminas histopatológicas com coloração H\&E. O sistema é baseado em um ensemble morfológico de modelos EfficientNet-B0, B3 e B4, com suporte a empilhamento (stacked generalization) e comparação estatística(Demšar2006 e Wolpert1992).

---

## Justificativas Teóricas e Arquiteturais

### 1. Uso de Ensembles na Classificação Histopatológica

**Fundamentação:** Ensembles são conhecidos por melhorar a acurácia e robustez de sistemas de classificação ao combinar a diversidade de modelos base (Çataltepe et al., 2003; Dietterich, 2000). Wolpert (1992) introduz o conceito de *stacked generalization* como alternativa superior ao simples voto, ao aprender uma regra de combinação baseada em um meta-aprendizado.

**Decisão Arquitetural:** Implementa-se um ensemble com duas formas de agregação:

* Soft-voting ponderado por desempenho das folds
* Stacking com um meta-modelo (regressão logística ou MLP leve)

---

### 2. Seleção dos Modelos EfficientNet-B0, B3 e B4

**Fundamentação:** A arquitetura EfficientNet (Tan & Le, 2019) permite escala balanceada de profundidade, largura e resolução. Modelos com diferentes escalas são sensíveis a diferentes padrões morfológicos (Tellez et al., 2019; Liang et al., 2023).

**Decisão Arquitetural:**

* B0: tecidos simples (ADI, DEB, MUS)
* B3: tecidos intermediários (LYM, MUC, NOR)
* B4: tecidos complexos (STR, TUM)

---

### 3. Validação com Cross-Validation em L Folds

**Fundamentação:** Demšar (2006) recomenda o uso de validação cruzada (e não hold-out) para comparações estatísticas confiáveis entre múltiplos classificadores. Isso evita viés de particionamento e permite aplicação dos testes de Friedman, Nemenyi e Wilcoxon.

**Decisão Arquitetural:** Cada modelo é treinado com validação cruzada estratificada (k=5). As predições de todas as folds são salvas para composição do meta-modelo e para estatísticas.

---

## Estrutura Modular do Sistema

### `inference.py`

Módulo de inferência finalque carrega os modelos treinados por fold (ensemble interno), aplica predições sobre uma imagem ou lote de imagens, combina as saídas dos modelos com soft-voting ponderado (com base nos weights do config.yaml) e retorna a classe predita e a probabilidade agregada, com possibilidade de salvar o resultado em CSV/JSON.

🔧 2. Estratégias e Decisões Técnicas
🧠 Modelos treinados por fold

    Para cada modelo (b0, b3, b4), teremos n_folds modelos (ex: 5).

    Para o ensemble final, vamos carregar todos os folds de cada modelo.

🔍 Justificativa:
Demsar (2006) e Wolpert (1992) indicam que o uso de ensembles reduz erro generalizado — especialmente ao explorar variação entre folds. Isso garante robustez.


🧮 3. Predição por modelo

Cada modelo será usado da seguinte forma:

output = model(image_tensor.unsqueeze(0).to("cuda"))
prob = torch.softmax(output, dim=1)

    Esse prob será coletado para cada modelo de cada fold.

    Será acumulado com pesos definidos no config.yaml.

🔗 4. Estratégia de Combinação - Weighted Soft Voting
Fórmula:
Pfinal(c)=1Z∑i=1Mwi⋅(1K∑k=1KPik(c))
Pfinal​(c)=Z1​i=1∑M​wi​⋅(K1​k=1∑K​Pik​(c))

    wiwi​: peso do modelo (ex: b0: 0.25, b3: 0.35, b4: 0.40)

    KK: número de folds

    Pik(c)Pik​(c): probabilidade da classe cc predita pelo modelo ii no fold kk

    Z=∑wiZ=∑wi​: normalização

📘 Referência: Ensemble learning literature (e.g., Opitz & Maclin, 1999; Kuncheva, 2004)

📊 5. Saída esperada

Para cada imagem:

    filename

    label_predito

    probabilidade_predita

    distribuição completa (array com todas as probabilidades)

Opcionalmente:

    Salvar como .json, .csv, .xlsx ou exibir no console.

🧱 6. Componentes do inference.py
Componente	Função
load_all_models()	Carrega todos os modelos treinados por fold, para cada EfficientNet
prepare_image()	Aplica preprocessamento (mesmo que no treino)
predict_image()	Faz forward em todos os modelos e aplica softmax
combine_predictions()	Aplica a fórmula do weighted soft voting
main()	Faz inferência sobre imagem ou diretório de imagens
⚙️ 7. Requisitos Técnicos

    torch, torchvision, albumentations, yaml, cv2, pandas

    Config compatível com config.yaml

    Reutilização da pipeline de core.model, core.config_loader, core.augmentations



### Diretório: `config/`

* `config.yaml`: define hiperparâmetros globais, estruturas dos modelos e pesos do ensemble (Peng, 2011).

- model_name: efficientnet_b0, b3, b4
- dropout: entre 0.2–0.4 (Tellez et al., 2019)
- input_size: 224, 300, 380
- folds: 5 (Demsar, 2006 recomenda L-fold)
- weights: para o soft-voting (baseado na média da acurácia por modelo)

### Diretório: `core/`

#### `model.py`

- Fábrica de modelos (EfficientNetClassifier) com parâmetros configuráveis via config.yaml
- Define os modelos base que alimentam os ensembles
- Inclui o meta-modelo de stacking (Wolpert1992)
- Adapta a última camada para 8 classes fixas (definidas no dataset HMU-GC-HE-30K)
- Aplica dropout para melhorar a regularização e evitar overfitting em poucos dados (Srivastava et al., 2014)
- Usa biblioteca `timm` para EfficientNet (Tan & Le, 2019)
- Define a classe `StackedEnsemble` com suporte a empilhamento

🧩 Componentes e Responsabilidades
Classe / Função	Responsabilidade
ModelFactory	Carregar arquitetura EfficientNet correta com base no config.yaml
SingleModel	Wrapper que encapsula o modelo + nome + metainformação útil
load_pretrained_model()	Carrega EfficientNet via torchvision ou timm e aplica head final
get_model()	Função principal: retorna modelo ajustado com número correto de classes

✅ Justificativas Técnicas (Literatura)
Decisão	Justificativa
Uso de EfficientNet-B0/B3/B4	Modelos leves e eficientes para histologia com bom desempenho em múltiplos benchmarks (Tan & Le, 2019)
Treinamento individual por arquitetura	Segue estratégia definida por Wolpert (1992) de treinar especialistas independentes antes do ensemble
Modularização por ModelFactory	Facilita experimentação, generalização e plugabilidade (Demšar, 2006)
Uso de timm (opcional)	Biblioteca amplamente usada e otimizada para modelos SOTA (Wightman, 2021)

📚 Referências:

- Tan & Le (2019): estrutura EfficientNet + trade-offs entre B0–B7
- Wolpert (1992): define a necessidade de combinar saídas via metamodelo
- Liang et al. (2023), Echle et al. (2021): uso de EfficientNet para classificação morfológica
- Demsar (2006): define que todos os modelos devem ser comparáveis de forma justa → output padronizado

#### `dataset.py`

- Abstração do conjunto de amostras e deve operar sob índices definidos externamente
- Implementa o dataset customizado que carrega os patches `TMEDataset`
- Usa `albumentations` para aplicar as transformacoes (Echle et al., 2021). As augmentations são delegadas ao code/augmentations.py
- Constroi train_loader, val_loader via StratifiedKFold
- Cada instância participa de treino e validação em algum ponto (Demsar, 2006)
- StratifiedKFold preserva proporção de classes em cada fold (melhora confiabilidade)
-“A separação clara entre dados e lógica de treino é crítica para reprodutibilidade e extensibilidade em experimentos com múltiplos folds.” — Demšar, 2006; Bouthillier et al., NeurIPS Reproducibility Checklist

- Usamos data augmentations específicas por classe histológica, respaldada por três pilares:

  1- Preserva morfologia crítica
    - Evita distorções irreais em estruturas como núcleos linfocitários ou fibras musculares
  2- Simula artefatos reais
    - Mimetiza variações típicas de coloração, corte e escaneamento (como sugerido por Tellez et al., 2019; Echle et al., 2021)
  3- Aprimora generalização
    - Treina o modelo para reconhecer variações intra-classe fisiologicamente possíveis, reduzindo overfitting (dos Santos et al., 2024)

 🔍 Essa estratégia é inovadora, pois tratamos o microambiente tumoral como um espaço morfologicamente heterogêneo, considerando que cada tipo de tecido exige sensibilidades distintas de detecção (como regras de negócio específicas).


*decisões tecnicas*

dataset.py deve receber:
  -image_paths, labels
  - indices (ou is_train + fold_idx se quiser lógica interna opcional)
  - augmentation_strategy: AugmentationStrategy
  - config, para reuso de parâmetros (ex: tamanho da imagem)
E deve aplicar a estratégia no __getitem__:
  - img = self.augmentation.get_transform(class_name)(image=np.array(img))["image"]

🧠 Objetivos e responsabilidades da TumorDataset

A classe TumorDataset será o único ponto do sistema com as seguintes responsabilidades:
Responsabilidade	Justificativa
Carregar imagens e rótulos a partir de uma lista	Reuso simples de imagens pré-particionadas
Aplicar a estratégia de augmentation associada à classe	Suporte à lógica do ensemble morfológico
Operar com índices de treino/validação passados por fora	Suporte a k-fold, mantendo controle externo no training.py
Permitir extensões futuras (ex: WSI, ROI dinâmicos)	Sustentabilidade arquitetural
🛠️ Requisitos arquiteturais

    📁 Caminho absoluto da imagem deve ser resolvido a partir de uma raiz (root_dir) + nome do arquivo (armazenado em CSV, XLSX etc.)

    🎯 A estratégia de augmentation vem de augmentations.py e será injetada via __init__, suportando Strategy Pattern

    🧪 O índice de fold (opcional) e o split (train, val) é controlado externamente

    🔒 Todas as transformações devem usar albumentations com ToTensorV2()

    🧼 Dataset deve ser robusto a erros e suportar debug (ex: imagem corrompida)

| Critério                    | Atendido? | Justificativa                                                  |
| --------------------------- | --------- | -------------------------------------------------------------- |
| Separation of Concerns      | ✅         | Augmentation, split e carregamento separados                   |
| Configurável via YAML       | ✅         | Estratégia passada via objeto externo                          |
| Plugabilidade de Estratégia | ✅         | Suporte completo ao Strategy Pattern                           |
| Suporte a Folds             | ✅         | Usa índices externos do StratifiedKFold                        |
| Extensibilidade futura      | ✅         | Suporta outras fontes (ex: WSI), outros canais, outras tarefas |


#### `training.py`

* Núcleo operacional que define o pipeline de treinamento supervisionado dos modelos base com validação cruzada estratificada
* Usa loop de epochs com treino -> validação -> early stopping
* Salva o melhor modelo por fold
* Armazena métricas em CSV
* Early stopping com paciência para evitar overfitting precoce (Echle et al., 2021)

🧱 OBJETIVO DO training.py

O módulo training.py é responsável por:
| Responsabilidade                                                 | Justificativa                                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1. Criar os folds e iterar por eles                              | Conforme validação cruzada k-fold de Demšar (2006)                   |
| 2. Treinar modelos individualmente (um por fold por arquitetura) | Estratégia de especialistas independentes de Wolpert (1992)          |
| 3. Salvar o melhor modelo por fold com nomeação padronizada      | Reprodutibilidade e comparabilidade estatística                      |
| 4. Avaliar desempenho em cada fold e registrar logs              | Necessário para análises pós-treino (e.g., ensemble ou estatísticas) |
| 5. Integrar: dataset.py, model.py, augmentations.py, config.yaml | Conexão de todas as partes do pipeline                               |


📐 Visão Geral da Arquitetura

training.py
├── lê config.yaml
├── para cada modelo (b0, b3, b4):
│   ├── para cada fold (1 a k):
│   │   ├── divide dataset
│   │   ├── aplica augmentations com TumorDataset
│   │   ├── instancia modelo via ModelFactory
│   │   ├── treina por N epochs com early stopping
│   │   ├── avalia métrica (ex: accuracy ou F1)
│   │   ├── salva melhor modelo em /weights/
│   │   └── loga resultados por fold
└── salva csv com estatísticas de treino


🧩 Componentes Integrados
| Componente             | Módulo de Origem             | Função                                               |
| ---------------------- | ---------------------------- | ---------------------------------------------------- |
| `TumorDataset`         | `core/dataset.py`            | Aplicar augmentations específicas por classe         |
| `ModelFactory`         | `core/model.py`              | Criar EfficientNet já configurado                    |
| `PerClassAugmentation` | `core/augmentations.py`      | Estratégia de transformação morfológica              |
| `config`               | `config.yaml`                | Parâmetros como `batch_size`, `epochs`, `loss`, etc. |
| `torch.optim`          | PyTorch                      | Otimizador (Adam, SGD, etc.)                         |
| `torchmetrics`         | PyTorch Lightning (opcional) | Métricas de desempenho (accuracy, F1-score, etc.)    |
| `tqdm`                 | barra de progresso           | Progresso de treino                                  |

🔍 Estratégias Arquiteturais Fundamentadas

📘 Wolpert (1992): Diversidade e Independência
    Cada fold é um treinamento totalmente separado
    Sem mistura de pesos ou reuso de inferência entre modelos base
    O ensemble final é feito após o treinamento completo

📘 Demšar (2006): Estatística baseada em rank
    K-fold estratificado com mesma seed
    Mesma divisão para todos os modelos
    Precisão, F1 ou outra métrica devem ser logadas por fold

📘 Fundamentação:

    Demšar (2006): não trata diretamente de early stopping, mas ressalta que comparações justas requerem modelos otimizados com o mesmo critério de avaliação. O early stopping entra aqui como controle padronizado de "término de otimização".

    Wolpert (1992): defende diversidade de especialistas. Modelos superajustados ao treino convergem para especializações enviesadas → early stopping preserva diversidade útil para ensemble.

    Na prática em histopatologia (CHD e artigos correlatos): early_stopping com patience=5–10 baseado na val_loss ou macro-F1 é padrão (referências: CHD, Kather et al., 2019).

✅ Recomendação:

early_stopping:
  monitor: "val_macro_f1"
  mode: "max"
  patience: 7

Essa escolha é:

    Alinhada com o objetivo de classificar tecidos variados com desempenho robusto (macro-F1)

    Compatível com a diversidade esperada entre modelos no ensemble

    Replicável nos logs por fold

✅ 2. Função de Custo
🔍 Revisão das opções:
Função de Custo	Vantagem	Quando Usar
CrossEntropyLoss	Simples e eficaz para classes balanceadas	Base padrão — usada na CHD e EfficientNet originais
Focal Loss	Penaliza mais os erros em classes raras	Quando há desequilíbrio severo de classes
Label Smoothing	Suaviza targets → evita overfitting local	Quando as classes são semelhantes morfologicamente
📘 Fundamentação:

    Wolpert (1992): favorece arquiteturas que podem se especializar (ex: focal loss pode enviesar demais).

    Demšar (2006): requer comparação com controle de variabilidade — cross-entropy mantém baseline simples e replicável.

    CHD e Kather et al. usam CrossEntropyLoss com sucesso.

✅ Recomendação:

loss_function: "CrossEntropyLoss"

E opcionalmente:

label_smoothing: 0.0  # pode ser 0.1 em experimentos controlados


✅ 3. Métrica Principal
🎯 Alvo: refletir a performance em todas as classes de tecidos, não só nas mais fáceis (como TUM).
📘 Fundamentação:

    Demšar (2006): defende o uso de métricas robustas para comparação inter-modelos. Média simples ou ponderada é recomendada.

    Wolpert (1992): reforça o valor da diversidade e consistência dos especialistas.

    Macro-F1 é a escolha predominante em estudos histopatológicos (CHD, PanNuke, BACH), pois trata todas as classes com igual importância.

✅ Recomendação:

metric: "macro_f1"


✅ 5. Balanceamento de Classes
📘 Fundamentação:

    O dataset CHD possui equilíbrio em número de imagens por classe, mas não em complexidade morfológica, conforme consta no [s41597-025-04489-9.pdf].

    PanNuke (2020) e Kather et al. discutem o problema de complexidade de classe como um fator de viés morfológico, e sugerem:

        Data augmentation morfo-específica (✅ já aplicada)

        Uso de WeightedRandomSampler baseado em dificuldade ou confusão — MAS isso viola a premissa de diversidade dos especialistas em ensembles.

✅ Recomendação:

Não aplicar balanceamento artificial adicional no loader.
O ensemble + augmentations já são estratégias consistentes e respeitam a independência dos modelos (Wolpert).

🧩 Arquitetura Modular do training.py

training.py
├── parse_args() → model_name, fold_id
├── carregar config.yaml
├── preparar fold (train_idx, val_idx)
├── criar datasets com augmentations
├── carregar modelo com ModelFactory
├── configurar otimizador, scheduler, etc
├── executar epochs com early stopping
│   ├── treinar 1 epoch
│   ├── validar
│   ├── logar métrica (macro-F1, loss)
│   ├── salvar melhor modelo (por val_macro_f1)
├── salvar:
│   ├── checkpoint.pth
│   ├── log.csv (loss, acc, f1 por epoch)
│   ├── val_predictions.csv
│   └── metrics.json (macro-F1 final, best_epoch, etc)
└── atualiza summary global (opcional)

📋 Módulos e Responsabilidades Conectadas
Módulo	Função
dataset.py	Gera o TumorDataset com os índices corretos para cada fold
model.py	Constrói o modelo de acordo com o nome e configurações
augmentations.py	Garante augmentations específicas por classe
stats.py (futuro)	Pode calcular métricas pós-treino (e.g., confusion matrix, F1 detalhado)
🛠️ Logging no Console

Vamos usar dois níveis de logging:

    logging padrão para salvar em arquivo (log por epoch e fold)

    tqdm para progresso visual no terminal (tempo estimado, perda atual etc.)

Configuração de log no início do script:

import logging

def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

✅ Resumo das Garantias do training.py
Garantia	Como será feito
Execução modular e paralelizável	Recebe argumentos CLI: --model e --fold
Compatibilidade com config.yaml	Carregamento unificado e dinâmico
Outputs organizados e independentes	outputs/{model_name}/fold_{k}/...
Registro científico completo	log.csv, metrics.json, val_predictions.csv por fold
Reprodutibilidade	Todos os elementos controlados por config, inclusive seed e augmentation
Conexão com ensemble posterior	Modelos salvos organizadamente para uso posterior no ensemble

#### `evaluation.py`

* Avalia cada fold com: Acurácia, Balanced Accuracy, Kappa, F1 (macro)
* Usa `sklearn.metrics` para facilitar integração
* Balanced Accuracy para lidar com dataset desbalanceado (Haixiang et al., 2017)
* Usa Kappa para ter mais cuidado com acerto por acaso (Viera & Garrett, 2005)

#### `ensemble.py`

* Implementa dois tipos de ensemble: Soft voting ponderado / Stacked generalization (Wolpert, 1992)

* Soft voting: média ponderada das saídas de softmax
* Stacked generalization:
    - input: concatenação das saídas softmax dos modelos base
    - modelo meta: regressão logística ou MLP simples

* Wolpert (1992) mostra que empilhar modelos com um meta-aprendizado é mais robusto que votos


#### `stats.py`

* Executa testes Friedman, Nemenyi e Wilcoxon (Demsar, 2006)
* Testa se o ensemble é estatisticamente melhor que modelos individuais
* Usa `Orange.evaluation` ou `scipy.stats`
* Gera gráfico de ranking para o artigo usando Orange
* Demsar (2006) proíbe uso de ANOVA quando dados são não normais ou dependentes
* Nemenyi controla erro tipo I em múltiplas comparações (controle de FWER

#### `augmentations.py`

- Responsável por centralizar e encapsular toda a lógica relacionada a transformações visuais (data augmentation)
- Implementa a função get_augmentation_by_class() com base em análise morfológica
- Permite alternar entre transformações padronizadas ou específicas por classe
- Parametrizável via config.yaml

*Decisões de implementação*

- Padrão Strategy Pattern para vialbilizar estratégias de transformações plugávies e com interface comum (por classe, global, nenhuma, etc.) (Gamma et al. 1994)

- A etapa de transformação morfológica foi implementada com base no padrão Strategy de design de software, permitindo que diferentes estratégias de data augmentation sejam facilmente intercambiáveis e parametrizáveis por classe histológica. Isso garante clareza, reprodutibilidade e possibilidade de generalização do sistema para outros domínios clínicos.
---

#### `metrics.py`

| Item                | Detalhes                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------- |
| 📁 Localização      | `ensemble_tme/core/metrics.py`                                                                |
| 🎯 Responsabilidade | Calcular e registrar as métricas quantitativas do modelo por **fold**, **classe**, **modelo** |
| 👥 Chamado por      | Diretamente de `training.py` após cada `fold`                                                 |
| 📊 Exporta          | `.csv` e `.json` contendo:                                                                    |
| Métrica               | Justificativa                                                                  |
| --------------------- | ------------------------------------------------------------------------------ |
| Accuracy              | Baseline para análise estatística (Demšar, 2006)                               |
| F1-score (macro)      | Ideal para avaliar desempenho multiclasse com desequilíbrio                    |
| F1-score (weighted)   | Compensa diferença de frequência entre classes                                 |
| F1-score (por classe) | Permite análise de especialização dos modelos (Wolpert, 1992)                  |
| Matriz de confusão    | Fundamental para identificar padrões de erro e redundância entre especialistas |

#### `visualization.py`

✅ Objetivo das visualizações

Mostrar com clareza, profundidade estatística e apelo visual:

    O comportamento e aprendizado dos modelos ao longo das épocas.

    A performance por classe e por modelo.

    O comportamento entre folds e entre modelos, respeitando comparações justas (Demšar).

    Evidenciar onde o ensemble ganha, e o que cada modelo individual erra ou acerta.

    Facilitar a compreensão prática para o público-alvo clínico (ex: patologistas, médicos computacionais).

🧭 Visões obrigatórias com base na literatura
1. 📉 Curvas de aprendizado (loss, F1, acc por época)

    📌 Já geradas no training.py, são obrigatórias.

    👨‍🔬 Servem para mostrar se houve overfitting, underfitting ou convergência.

    ✅ Demšar (2006) reforça que análise temporal é essencial em pipelines comparativos.

    ✅ Mostrar até onde o modelo foi capaz de aprender, inclusive o efeito do early stopping.

2. 📊 Matriz de confusão (por modelo)

    Ideal: uma matriz por modelo, agregada entre folds (normalizada e com valores absolutos).

    Serve para entender quais classes são confundidas entre si.

    Muito útil para evidenciar falhas como:

        confusão entre TUM e DEB

        ou STR com MUS.

    📌 Importante para justificar decisões clínicas futuras baseadas no modelo.

3. 🎯 Curvas ROC + AUC (por classe e média macro/micro)

    Um gráfico por modelo, contendo 8 curvas (uma para cada classe), além da curva macro e micro.

    Indica quão bem o modelo separa as classes, mesmo em desequilíbrio de decisão binária.

    Referenciado amplamente em:

        Litjens et al., 2017 (MedIA)

        Wang et al., 2020 (EfficientNet for colorectal cancer classification)

4. 📦 Boxplot (ou Violin Plot) de macro F1 por fold e por modelo

    Permite comparar a distribuição do desempenho ao longo dos folds.

    📌 Demšar (2006) recomenda fortemente boxplots e ranks para comparar modelos em múltiplos datasets/folds.

    Pode ser estendido para rank plot se estivermos comparando mais de três variantes.

5. 🧠 Gráfico de barras com desempenho por classe e por modelo

    Para evidenciar que alguns modelos funcionam melhor para certas classes (ex: B3 para LYM).

    Usar F1-score por classe como base.

📍 Visões complementares (opcional, mas recomendadas)
6. 🧩 Gráfico de contribuição dos modelos no ensemble

    Um radar chart ou stacked bar para mostrar como os modelos individuais contribuem por classe.

    Relaciona com Wolpert (1992): como as "biases" individuais do modelo formam um sistema mais robusto.

7. 📈 Heatmaps de saliência / Grad-CAM (qualitativos)

    Uma imagem por classe mostrando onde o modelo foca para decidir.

    Útil em artigos voltados para aplicação médica e justificativa de decisões.

    Pode vir como apêndice ou suplemento visual.

✅ Resumo consolidado: o que devemos gerar
Visualização	Relevância	Para quem serve?	Fundamentação
Curvas de treino (loss/F1/acc)	Alta	Pesquisadores de IA	Demšar 2006, prática comum
Matriz de confusão (por modelo)	Alta	Patologistas, IA	Explica erros e confusões
Curvas ROC + AUC (por classe)	Alta	IA, revisão por pares	Padrão clínico e comparativo
Boxplot F1 macro (por fold/modelo)	Alta	Pesquisadores, editores	Demšar 2006, comparação estatística
Barras por classe e modelo	Alta	Comparação intra-classe	Diagnóstico diferencial
Contribuição dos modelos	Média	Para justificar o ensemble	Wolpert 1992
Grad-CAM	Média	Patologistas, visual clínico	Explicabilidade

### Diretorio: `scripts/`

#### `run_all_folds.py`

🧠 O que esse script faz?

| Etapa                     | Descrição                                                           |
| ------------------------- | ------------------------------------------------------------------- |
| 🔄 `load_config()`        | Lê o `config.yaml` para saber quais modelos e quantos folds existem |
| 🔁 `itertools.product()`  | Gera todas as combinações `(modelo, fold)`                          |
| 🎯 `CUDA_VISIBLE_DEVICES` | Atribui cada job a uma GPU disponível                               |
| ⚙️ `subprocess.Popen`     | Executa `training.py` como subprocesso com redirecionamento de log  |
| 📁 `logs/*.out`           | Cada job salva seu log em um arquivo independente                   |
| 🛑 `--dry-run`            | Permite simular sem executar (útil para debug)                      |

✅ Exemplo de execução:

python scripts/run_all_folds.py --gpus 2 --config config/config.yaml

- Executará os folds usando 2 GPUs de forma rotativa e criará logs em logs/efficientnet_b3_fold4.out

#### `evaluate_ensemble.py`

Executa o ensemble morfológico a partir dos modelos treinados, aplica a inferência sobre os dados de validação ou teste, e gera todas as métricas e visualizações automaticamente.

🧪 Como executar
Com os modelos já treinados e salvos em outputs/efficientnet_b*/fold_*/checkpoint.pth, execute:

python scripts/evaluate_ensemble.py --config config/config.yaml --csv data/test.csv

📂 Estrutura de saída

O script criará a pasta:

outputs/
 └── ensemble/
      ├── ensemble_summary.json          # F1 e ACC do ensemble
      ├── confusion_matrix.png
      ├── roc_curve.png
      └── classification_report.csv


#### `run_statistical_analysis.py`

- Garante execução isolada e padronizada;
- Pode ser facilmente reaproveitado em qualquer experimento;
- Gera automaticamente tabelas e gráficos de análise estatística com base na literatura (Demšar, Wolpert etc.).

📌 Como usar

No terminal:

python scripts/run_statistical_analysis.py \
    --csv outputs/fold_metrics.csv \
    --baseline efficientnet_b0 \
    --metric f1 \
    --output-dir outputs/stats

#### `run_inference.py`

🧠 O que o script faz:

    Lê o config.yaml

    Permite inferir uma única imagem (--image) ou todas as de uma pasta (--folder)

    Usa a função predict_image() definida em inference.py

    Imprime os resultados no console (classe final + score)

    Salva opcionalmente os resultados num CSV (--save)
    
💡 Exemplo de uso:

# Inferência de uma única imagem
python scripts/run_inference.py --image test_images/ADI_1.png

# Inferência em uma pasta + salvar CSV
python scripts/run_inference.py --folder test_images --save outputs/inference_results.csv


### Diretório: `experiments/`

#### `run_experiment.py`

* Roda o treinamento de todos os modelos base
* Salva as predições por fold
* Treina o meta-modelo com as predições
* Avalia e executa os testes estatísticos

---

## Tecnologias Selecionadas

| Tecnologia          | Justificativa                                              |
| ------------------- | ---------------------------------------------------------- |
| PyTorch + TIMM      | Suporte às arquiteturas EfficientNet com pretrained models |
| scikit-learn        | Métricas, modelos leves (LogReg) para stacking             |
| albumentations      | Augmentations sensíveis à histologia (Tellez et al., 2019) |
| pandas              | Manipulação de dados e tabelas de resultados               |
| matplotlib / Orange | Visualização de rankings com Nemenyi                       |

---

## Saídas Esperadas

* `outputs/scores.csv`: acurácia e F1 por modelo/fold
* `outputs/ranks.png`: ranking estatístico com teste de Nemenyi
* `outputs/checkpoints/`: pesos salvos por modelo/fold
* `README.md`: documentação arquitetural completa para escrita de artigo

---

🔄 Fluxograma de Execução Geral

```
INÍCIO
  │
  ├──> 1. Carregar configuração (config.yaml)
  │        └── model_name, input_size, lr, dropout, folds
  │
  ├──> 2. Construir modelo base (EfficientNet-B0/B3/B4)
  │        └── via model.py -> EfficientNetClassifier
  │
  ├──> 3. Carregar dataset com aug. e labels (dataset.py)
  │        └── Construção por fold (StratifiedKFold)
  │
  ├──> 4. Treinamento por fold (training.py)
  │        ├── Loop: treino/validação por fold
  │        └── Checkpoint do melhor modelo (early stopping)
  │
  ├──> 5. Avaliação por fold (evaluation.py)
  │        └── Métricas padronizadas (acc, F1, kappa)
  │
  ├──> 6. Registro dos resultados (scores.csv)
  │
  ├──> 7. Comparação estatística (stats.py)
  │        ├── Teste de Friedman + post-hoc Nemenyi
  │        └── Geração de gráfico de ranks
  │
  └──> 8. Geração de ensemble (ensemble.py)
            ├── Soft-voting
            └── Stacked generalization (Wolpert)
FIM

```

🏗️ Fluxo arquitetural

```
               +------------------+
               | config.yaml      |
               +------------------+
                        |
                Load config (main.py)
                        ↓
               +------------------+
               | training.py      | ← controla StratifiedKFold (sklearn)
               +------------------+
                        |
             ┌──────────┴────────────┐
             ↓                       ↓
     Fold indices (train/test)   Fold loop
             ↓                       ↓
     +-----------------+       +-------------------+
     | dataset.py      |  →    | model.py          |
     | recebe índices  |       | instancia modelo  |
     | aplica aug por  |       +-------------------+
     | classe          |
     +-----------------+

```

---

🧩 Arquivos .py e suas Responsabilidades

| Arquivo                       | Responsabilidade Principal                                                              | Chamado por                             |
| ----------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------- |
| `training.py`                 | Executar o ciclo de treinamento de um modelo por fold                                   | (n/a - ponto de entrada)                |
| `config_loader.py`            | Carregar e gerenciar configurações do `config.yaml`                                     | `training.py`, `model.py`, `dataset.py` |
| `dataset.py`                  | Gerar datasets para treino e validação com augmentations por classe e fold              | `training.py`                           |
| `augmentations.py`            | Fornecer estratégia de transformação morfológica por classe                             | `dataset.py`                            |
| `model.py`                    | Construir o modelo (`EfficientNet`) com as configurações de input e dropout específicas | `training.py`                           |
| `metrics.py` (a ser criado)   | Calcular métricas: accuracy, macro-F1, matriz de confusão                               | `training.py`                           |
| `logger.py` (opcional)        | Criar logs de arquivo + console (se não for feito inline)                               | `training.py`                           |
| `visualization.py` (opcional) | Gerar gráficos para uso no artigo (curvas de loss, F1, acc)                             | `training.py`                           |
| `stats.py` (opcional)         | Agregador de métricas globais após treino de todos os folds                             | Fase posterior ao ensemble              |

🔗 Fluxo de Relação entre Módulos

```
training.py
├── parse_args() ← CLI
├── load_config() ← config_loader.py
├── select_model_config() ← config["models"]
├── StratifiedKFold ← scikit-learn
│   └── Dataset(indexes) ← dataset.py
│       └── AugmentationStrategy ← augmentations.py
├── build_model() ← model.py
├── optimizer/scheduler/criterion ← torch + config
├── train loop:
│   ├── forward, backward, step
│   ├── validation loop:
│       ├── metrics ← metrics.py
│       └── logs ← CSV + console
│   ├── early stopping
│   └── save_best_model()
├── save_metrics(), predictions.csv
└── plot_curves() ← visualization.py

```

## Referências Bibliográficas

* Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs. ICML.
* Tellez, D. et al. (2019). Whole-slide mitosis detection in H\&E breast histology using PHH. Medical Image Analysis.
* Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.
* Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR.
* Echle, A. et al. (2021). Deep learning in cancer pathology: a new generation of clinical biomarkers. The Lancet Oncology.
* Liang, H. et al. (2023). Deep learning supported discovery of biomarkers for improved cancer immunotherapy. Nat Mach Intell.
