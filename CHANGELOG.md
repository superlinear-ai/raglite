## v0.7.0 (2025-03-17)

### Feat

- replace post-processing with declarative optimization (#112)
- compute optimal sentence boundaries (#110)
- migrate from poetry-cookiecutter to substrate (#98)
- make llama-cpp-python an optional dependency (#97)
- add ability to directly insert Markdown content into the database (#96)
- make importing faster (#86)

### Fix

- fix CLI entrypoint regression (#111)
- lazily raise module not found for optional deps (#109)
- revert pandoc extra name (#106)
- avoid conflicting chunk ids (#93)

## v0.6.2 (2025-01-06)

### Fix

- remove unnecessary stop sequence (#84)

## v0.6.1 (2025-01-06)

### Fix

- fix Markdown heading boundary probas (#81)
- improve (re)insertion speed (#80)
- **deps**: exclude litellm versions that break get_model_info (#78)
- conditionally enable `LlamaRAMCache` (#83)

## v0.6.0 (2025-01-05)

### Feat

- add support for Python 3.12 (#69)
- upgrade from xx_sent_ud_sm to SaT (#74)
- add streaming tool use to llama-cpp-python (#71)
- improve sentence splitting (#72)

## v0.5.1 (2024-12-18)

### Fix

- improve output for empty databases (#68)

## v0.5.0 (2024-12-17)

### Feat

- add MCP server (#67)
- let LLM choose whether to retrieve context (#62)

### Fix

- support pgvector v0.7.0+ (#63)

## v0.4.1 (2024-12-05)

### Fix

- add and enable OpenAI strict mode (#55)
- support embedding with LiteLLM for Ragas (#56)

## v0.4.0 (2024-12-04)

### Feat

- improve late chunking and optimize pgvector settings (#51)

## v0.3.0 (2024-12-03)

### Feat

- support prompt caching and apply Anthropic's long-context prompt format (#52)

## v0.2.1 (2024-11-22)

### Fix

- improve structured output extraction and query adapter updates (#34)
- upgrade rerankers and remove flashrank patch (#47)
- improve unpacking of keyword search results (#46)
- add fallbacks for model info (#44)

## v0.2.0 (2024-10-21)

### Feat

- add Chainlit frontend (#33)

## v0.1.4 (2024-10-15)

### Fix

- fix optimal chunking edge cases (#32)

## v0.1.3 (2024-10-13)

### Fix

- upgrade pdftext (#30)
- improve chunk and segment ordering (#29)

## v0.1.2 (2024-10-08)

### Fix

- avoid pdftext v0.3.11 (#27)

## v0.1.1 (2024-10-07)

### Fix

- patch rerankers flashrank issue (#22)

## v0.1.0 (2024-10-07)

### Feat

- add reranking (#20)
- add LiteLLM and late chunking (#19)
- add PostgreSQL support (#18)
- make query adapter minimally invasive (#16)
- upgrade default CPU model to Phi-3.5-mini (#15)
- add evaluation (#14)
- infer missing font sizes (#12)
- automatically adjust number of RAG contexts (#10)
- improve exception feedback for extraction (#9)
- optimize config for CPU and GPU (#7)
- simplify document insertion (#6)
- implement basic features (#2)
- initial commit

### Fix

- lazily import optional dependencies (#11)
- improve indexing of multiple documents (#8)
