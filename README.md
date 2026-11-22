# energAIze Inference

Serviço HTTP que executa um algoritmo para uma instalação. Recebe observações e previsões do Percepta, prepara as entradas esperadas pelo algoritmo e devolve uma decisão. No modo gerido, a resposta identifica a instalação e a versão em execução e explica as correções aplicadas aos valores propostos. Quem recebe a decisão é responsável por a guardar ou a encaminhar para o equipamento.

## Escolher o modo de execução

| Preciso de… | Modo | Onde começar |
| --- | --- | --- |
| Preparar um bundle, testar uma proposta, trocar de algoritmo e recuperar a versão após reinício | Gerido: contexto da casa, configuração e ativação explícitos; pedidos em `/inference/v2` | [Operar uma instalação](docs/operacao.md) |
| Executar um bundle existente pela interface `/admin/load` e `/inference`, incluindo estratégias especializadas iCharging/RH1 | Clássico: carregamento por manifesto e seleção de agente por pedido | [Referência da API clássica](docs/api_contract.md) |

Um **bundle** contém o manifesto, o modelo ONNX ou política JSON e os ficheiros referenciados. O **contexto da instalação** descreve os equipamentos, canais, unidades e limites daquela casa. O **pedido de inferência** traz os valores de um ciclo. São três entradas distintas: acrescentar uma chave à mensagem não instala um carregador nem muda o algoritmo. O Runtime deve comunicar a alteração de composição. O serviço não descobre equipamentos, não treina modelos e não faz deployments.

## Executar no portátil

Num checkout novo, criar o ambiente na raiz do repositório com Python 3.10, a versão usada pela integração contínua. Se o ambiente já estiver instalado, basta executar a última linha:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Sem configuração nem manifesto de arranque, o processo responde em [health](http://127.0.0.1:8000/health) e aguarda um modelo. A [API interativa](http://127.0.0.1:8000/docs) mostra os corpos aceites. Para experimentar o modo gerido desde o arranque, seguir o guia em vez de iniciar uma segunda instância na mesma porta.

Para Docker, [compose.preparation.yml](compose.preparation.yml) descreve o modo gerido com uma pasta de bundles só de leitura e outra de estado persistente. O [Compose clássico](docker-compose.yml) usa os bundles de exemplo, publica a porta configurada e monta uma pasta de logs. O [perfil GPU](docker-compose.gpu.yml) requer um runtime NVIDIA compatível. Estes ficheiros descrevem como lançar o serviço; não identificam o que está instalado numa casa.

## Imagens e releases

`calof/energaize_inference:v0.2.0` é a release CPU para `linux/amd64` e
`linux/arm64`, publicada como um único manifesto. Selecionar
`CPUExecutionProvider` e runtime Docker `runc`; não requer CUDA ou JetPack.
A imagem usa as mesmas rotas e contratos nas duas arquiteturas. A compatibilidade
do host e os recursos de uma instalação continuam a exigir ensaio no destino.

O `Dockerfile` e as tags `latest`/`pilot-<commit>` mantêm o perfil CUDA da cloud.
O `Dockerfile.cpu` usa ONNX Runtime CPU e as restantes dependências partilhadas.
Para reproduzir a instalação CPU fora de Docker, usar `requirements-cpu.txt`.

Uma tag Git `vX.Y.Z` cujo commit pertença a `main` inicia o workflow
`CPU multi-architecture release`: testes, publicação de um candidato por digest,
validação HTTP das duas arquiteturas e promoção da tag versionada. AMD64 é testado
nativamente; ARM64 usa QEMU num runner AMD64. O ensaio gera um pequeno modelo ONNX,
ativa-o, envia uma mensagem nativa Percepta com unidades e verifica a decisão,
a correção de safety e a recuperação após reinício. O resultado fica no resumo
do workflow; a tag final só é publicada se ambos os ensaios passarem.
Este workflow não faz deployment nem acede às casas.

## GPU na Jetson Xavier — ensaio de 22/11/2026

A release CPU `v0.2.0` não utiliza a GPU. Na Xavier da Régua (L4T 35.3.1,
CUDA 11.4, TensorRT 8.5.2.2), três ONNX sintéticos foram comparados nativamente:
CPU na imagem publicada, com dois threads; GPU no TensorRT instalado, sem
fallback CPU. Cem ciclos após aquecimento, batch 1, transferências GPU incluídas;
sem HTTP/rede, sem alterar clocks ou modo de energia.

| Rede (larguras) | CPU mediana | GPU FP32 | GPU com FP16 permitido |
| --- | --- | --- | --- |
| 64–32–2 | 0,049 ms | 0,325 ms | 0,322 ms |
| 64–1024–1024–1024–2 | 0,428 ms | 0,473 ms | 0,563 ms |
| 64–2048–2048–2048–2 | 1,253 ms | 1,183 ms | 0,734 ms |

O erro absoluto máximo foi 2,99e-7 em FP32 e 0,00108 com FP16 permitido,
nos oito vetores de referência. A construção FP16 avisou sobre pesos subnormais:
este ensaio não certifica a precisão de um algoritmo real. A rede maior ganhou
aproximadamente 1,7×; as pequenas não justificaram a GPU. É uma comparação
exploratória de dois motores e precisões, não uma garantia de desempenho geral.

Também passou a execução ONNX/TensorRT dentro do container NVIDIA já existente
`nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`, usando `--runtime=nvidia`,
com paridade numérica. A imagem da aplicação não foi substituída. Falta preparar
um perfil de inferência GPU específico para JetPack 5, compatível com Python e
CUDA/TensorRT, e validar nele o percurso HTTP/Percepta e os modelos reais.
Não selecionar a imagem CUDA AMD64 da cloud para a Xavier.

O ensaio reproduzível está em [benchmark_jetson_gpu.py](scripts/benchmark_jetson_gpu.py).
Gerar os ficheiros com `--prepare DIR`; executar `--cpu DIR` dentro da imagem
CPU na Jetson e `--gpu DIR` no ambiente TensorRT 8 (opcional `--fp16`).
A pasta deve ser temporária e removida depois. Os modelos, referências e
containers deste ensaio foram removidos; VPN, drivers, worker e piloto preservados.

## Alterar e verificar o código

| Alteração | Começar em |
| --- | --- |
| Corpo dos pedidos e respostas | [Rotas HTTP](app/routers) e [contratos versionados](app/contracts) |
| Como uma mensagem passa a entradas do algoritmo | [Adaptação e execução](app/adapters/channel_message.py) |
| Como aceitar um bundle ou configuração | [Preparação](app/preparation) e [schemas publicados](contracts) |
| Regras físicas e auditoria | [Safety](app/adapters/safety_policy.py) |
| Troca de algoritmo, estado e recuperação | [Ciclo de vida](app/lifecycle) |
| Carregamento e execução clássicos | [Serviço de aplicação](app/application), [adaptadores](app/adapters) e [motores ONNX/RBC](app/services/runtimes) |

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/export_contract_schemas.py --check
.venv/bin/python scripts/export_preparation_schema.py --check
.venv/bin/python scripts/export_safety_schemas.py --check
```

Os [exemplos](examples) e os [pedidos Postman](postman) acompanham o código. O histórico das iterações e das suas verificações está nas [evidências da entrega](https://linear.app/softcps/document/inferencia-evidencias-das-iteracoes-1-6-e-integracao-pendente-236cd094e956).
