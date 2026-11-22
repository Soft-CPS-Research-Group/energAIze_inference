# Operar a inferência de uma instalação

O processo mantém um algoritmo ativo por instalação. É possível preparar e avaliar um candidato enquanto a versão atual continua a servir pedidos. A troca só acontece depois de uma ativação explícita. A inferência devolve propostas e ações finais; o consumidor decide se ainda pertencem à instalação e à execução certas antes de as usar.

## Arranque e armazenamento

Usar um processo Uvicorn por pasta de estado. A identidade da instalação fica associada a essa pasta na primeira utilização. Sem a pasta, o serviço arranca em modo clássico.

| Definição | Valor a fornecer |
| --- | --- |
| `INSTALLATION_SITE_ID` | Identificador da casa; deve coincidir com o da configuração |
| `ALLOWED_BUNDLE_ROOT` | Diretório de entrada dos bundles; montar só de leitura |
| `CONFIGURATION_STORE_DIR` | Diretório privado, persistente e gravável para SQLite, configurações e cópias validadas dos bundles |
| `ADMIN_API_TOKEN` | Segredo local de pelo menos 32 caracteres; usado nas rotas administrativas e em `/inference/v2` |
| `DEPLOYMENT_MODEL_ID` | Identificador estável do modelo escolhido em Deployment para esta instalação |
| `PERCEPTA_CYCLE_TOKEN` | Credencial dedicada de pelo menos 32 caracteres para a ligação do Percepta. O Runtime gera-a ao preparar a instância. |
| `ONNX_EXECUTION_PROVIDERS` | Provider disponível no destino, por exemplo `CPUExecutionProvider` |
| `CONFIGURATION_FILE`, `PREPARATION_REQUEST_FILE` | Opcionais: guardar configuração e preparar um candidato no arranque. Não o ativam. |

Em Docker, o [Compose gerido](../compose.preparation.yml) recebe as pastas do host em `BUNDLES_HOST_PATH` e `STATE_HOST_PATH` e monta-as em `/srv/inference/bundles` e `/var/lib/inference`. A pasta de estado tem de suportar bloqueio POSIX, SQLite e sincronização durável dos ficheiros. Não misturar este modo com `MODEL_MANIFEST_PATH`.

## Primeiro ciclo: exemplo com dois carregadores

Usar o [exemplo sintético](../examples/safety_demo) a partir da raiz do repositório, com o ambiente Python instalado. Os comandos abaixo usam Bash, curl e jq e destinam-se a uma instância local vazia. A pasta temporária serve apenas para esta demonstração. Para operação continuada, escolher uma pasta permanente.

```bash
unset MODEL_MANIFEST_PATH CONFIGURATION_FILE PREPARATION_REQUEST_FILE
export INSTALLATION_SITE_ID=example-house
export ALLOWED_BUNDLE_ROOT="$PWD/examples/safety_demo"
export CONFIGURATION_STORE_DIR="$(mktemp -d)"
export ADMIN_API_TOKEN="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
base=http://127.0.0.1:8000
demo=examples/safety_demo
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
service_pid=$!
```

No mesmo terminal, aguardar que o processo indique que arrancou. Esta função envia o token; a segunda acompanha uma operação assíncrona até obter o identificador do recurso ou mostrar a falha.

```bash
api() {
  curl -fsS -H "Authorization: Bearer $ADMIN_API_TOKEN" \
    -H 'Content-Type: application/json' "$@"
}
resource() {
  local job="$1" receipt
  for ((i=0; i<120; i++)); do
    receipt=$(api "$base/admin/v1/operations/$job") || return 1
    case "$(jq -r .status <<<"$receipt")" in
      succeeded) jq -er .resource_id <<<"$receipt"; return ;;
      pending|running) sleep 1 ;;
      *) echo "$receipt" >&2; return 1 ;;
    esac
  done
  echo "Consultar novamente a operação $job antes de repetir o pedido." >&2
  return 1
}
```

**1. Guardar a configuração e preparar o bundle.** O serviço confere a revisão, os caminhos, os ficheiros, os hashes declarados e a compatibilidade entre canais e algoritmo. Uma revisão existente não pode receber conteúdo diferente. Um recibo HTTP 202 significa apenas que o pedido foi aceite. Antes de continuar, confirmar que o candidato ficou em `ready`.

```bash
api -X PUT "$base/admin/v1/configurations/configuration-safety-1" --data-binary "@$demo/configuration.json"
job=$(api "$base/admin/v1/candidates" -H 'Idempotency-Key: demo-prepare' \
  --data-binary "@$demo/prepare.json" | jq -er .operation_id)
candidate=$(resource "$job")
api "$base/admin/v1/candidates/$candidate" | jq -e '.state == "ready"'
```

**2. Experimentar sem ativar.** Os dois carregadores pedem 6 kW cada. Com 3 kW de consumo de base e um limite de 11 kW, a prioridade configurada reduz o segundo carregador para 2 kW. Comparar a proposta, as ações finais e a auditoria:

```bash
api "$base/admin/v1/candidates/$candidate/evaluations" --data-binary "@$demo/request.json" | jq .
```

**3. Declarar a instalação, ativar e executar um ciclo.** Os valores esperados a `null` nos exemplos só se aplicam a um estado inicial vazio. No fim, o pedido transporta as medições originais dentro de `features` e identifica o contexto e a ativação esperados.

```bash
context=$(api -X PUT "$base/admin/v1/installation" -H 'Idempotency-Key: demo-context' \
  --data-binary "@$demo/installation-update.json" | jq -er .context_id)
job=$(jq --arg c "$candidate" --arg x "$context" '.candidate_id=$c | .expected_context_id=$x' \
  "$demo/activation.template.json" | api "$base/admin/v1/activations" \
  -H 'Idempotency-Key: demo-activate' --data-binary @- | jq -er .operation_id)
activation=$(resource "$job")
curl -fsS "$base/ready"
jq --arg a "$activation" --arg x "$context" '.expected_activation_id=$a | .expected_context_id=$x' \
  "$demo/inference-request.template.json" | api "$base/inference/v2" --data-binary @- | jq .
```

A resposta inclui IDs de contexto, ativação e ciclo, proposta original, ações finais e auditoria. Não confirma entrega a um equipamento. Terminar o processo de demonstração com `kill "$service_pid"`; reiniciar com a mesma pasta e as mesmas definições recupera a ativação confirmada.

## Ligação direta do Percepta

O Runtime configura o destino, a credencial de ciclo e o cabeçalho `X-Installation-Context`. A rota `/inference/sites/{site_id}/models/{model_id}/observe` aceita GET para verificar a instalação e POST para receber a mensagem nativa do ciclo. O início e o fim do período têm de incluir o fuso horário. A resposta contém `decision` com as ações, a ativação, o contexto e o identificador do ciclo. O Percepta usa esta ligação em observação.

Uma nova versão ativada no mesmo contexto passa a responder na mesma rota. Se mudar o contexto da instalação, usar **Connect active model** em Deployment e aplicar a revisão do Percepta. Uma instância sem versão ativa responde 503. Um pedido com um contexto desatualizado responde 409.

## Mensagem, algoritmos e limites físicos

O adaptador `channels-1` lê apenas os caminhos associados a entradas do algoritmo, abaixo de `features`. Chaves com pontos são literais; em arrays, o índice é um componente do caminho. Equipamentos adicionais ou o histórico de decisões presentes na mensagem não passam a entradas. As unidades das observações sem metadados vêm do contexto declarado. Nas previsões, são verificadas unidade, frequência e horizonte quando presentes. Há conversão de escala entre W/kW/MW ou Wh/kWh/MWh, mas não entre energia e potência. Campos obrigatórios ausentes e valores inválidos dão erro; não há imputação nem garantia de frescura.

O modo gerido suporta ONNX com uma entrada e uma saída float32 de largura fixa, e políticas JSON simples de igualdade com ações escalares. As estratégias especializadas RH1/iCharging e as interfaces dinâmicas não devem ser ativadas por esta via. A versão do contrato de configuração é independente da versão do formato de exportação de Algorithms; a compatibilidade com os bundles v2 desse produtor requer validação própria.

| Modo de safety | Ações devolvidas |
| --- | --- |
| `off` | Proposta original, sem verificação física |
| `observe` | Proposta original; a auditoria indica as alterações sugeridas |
| `intervene` | Valores corrigidos; a auditoria identifica cada alteração aplicada |

O perfil suporta limites individuais, precisão decimal e **um limite superior agregado** com prioridades explícitas. Se a carga não controlada impedir o cumprimento do limite, devolve os mínimos configurados e indica o excesso residual. No exemplo, usar `request-residual.json` na avaliação: com 12 kW de carga de base, o resultado é 1 + 1 kW, com 3 kW de excesso. Uma violação física gera um resultado auditável; uma falha técnica de execução não gera ações fictícias.

Os [schemas v2](../contracts/v2) definem os documentos do modo gerido. Configurações v1 continuam disponíveis para preparação e avaliação, sem migração implícita: consultar os [schemas v1](../contracts/v1/schemas) e [exemplos](../contracts/v1/examples). Para ativar, usar configuração v2 e o adaptador `channels-1`.

## Troca, reinício e diagnóstico

- **Mesmo contexto, novo algoritmo:** preparar e avaliar o candidato. Ativar com uma nova chave de idempotência e os IDs atuais. A execução anterior continua até à publicação integral da nova versão. Repetir o mesmo pedido com a mesma chave devolve o recibo original, sem reativar.
- **Mudou a casa ou o formato da mensagem:** coordenar a mudança com o produtor e o consumidor. Declarar a instalação atual com os IDs anteriores, ativar o candidato compatível e distribuir os novos IDs depois de confirmar `/ready`. Ciclos já aceites podem terminar com os IDs antigos; o consumidor deve descartá-los. O ID de ciclo não define expiração de comandos.
- **Reinício ou retorno à versão anterior:** o serviço recupera a geração confirmada na pasta de estado. Se os ficheiros ou o provider já não forem válidos, fica indisponível. Para voltar a um bundle anterior, pedir nova ativação e revalidar contra a instalação atual. Não apagar linhas de SQLite nem trocar ficheiros de um snapshot em uso.
- **Consultar o estado:** `GET /health` verifica o processo; `GET /ready` indica se pode servir. Usar `GET /admin/v1/status` para estado local, `GET /admin/v1/installation` para o contexto e `GET /admin/v1/activations/{id}` para uma ativação. Após timeout, consultar a operação: o prazo impede publicação tardia, mas não interrompe à força código nativo.
- **Interpretar erros:** 400 indica entrada consumida inválida; 401, autenticação; 409, conflito de revisão/estado; 422, contrato ou perfil incompatível; 429, capacidade concorrente esgotada; 503, indisponibilidade; 507, limite de armazenamento/histórico. Uma falha nativa pode devolver 500. Consultar também o resultado das operações assíncronas.
- **Dimensionar:** por omissão, quatro preparações pendentes/em curso, 64 registos de candidatos, 512 MiB por cópia, 4 GiB para snapshots e quatro inferências concorrentes. As definições correspondentes são `MAX_PENDING_PREPARATIONS`, `MAX_STORED_CANDIDATES`, `MAX_CANDIDATE_BYTES`, `MAX_SNAPSHOT_STORAGE_BYTES` e `MAX_CONCURRENT_INFERENCES`. Estes valores não limitam toda a memória consumida por um modelo.
