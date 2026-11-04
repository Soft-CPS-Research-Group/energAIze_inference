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
