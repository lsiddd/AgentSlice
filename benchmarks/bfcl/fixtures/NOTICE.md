# Origem dos dados

`multi_turn_base_gfs.jsonl` e `gorilla_file_system_tools.json` são um
subconjunto do dataset da Berkeley Function Calling Leaderboard (BFCL) v4,
mantido em https://github.com/ShishirPatil/gorilla (licença Apache-2.0).

- `multi_turn_base_gfs.jsonl`: as 13 tasks da categoria `multi_turn_base`
  cujo `involved_classes` é só `["GorillaFileSystem"]`, extraídas de
  `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json`
  e mescladas com o `ground_truth` correspondente de
  `.../data/possible_answer/BFCL_v4_multi_turn_base.json`.
- `gorilla_file_system_tools.json`: convertido de
  `.../data/multi_turn_func_doc/gorilla_file_system.json`, com `type: dict`
  normalizado para `type: object` (JSON Schema padrão) e um campo `effects`
  próprio do AgentSlice adicionado por método (`pure` para os que só leem
  estado, `effectful` para os que mutam o sistema de arquivos simulado).

Outras categorias multi-turn da BFCL (`multi_turn_long_context`,
`multi_turn_miss_func`, `multi_turn_miss_param`) e as demais classes de API
simuladas (`TwitterAPI`, `TicketAPI`, `TradingBot` etc.) não estão cobertas
nesta primeira fatia — ver `benchmarks/README.md`.
