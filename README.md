# free-model-aggregator-2

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Sazharr911/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Sazharr911/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server behind two client wires. A client speaks either OpenAI Chat Completions (`POST /v1/chat/completions`) or Anthropic Messages (`POST /v1/messages`), in JSON or streaming mode, and every answer is a real completion from one of the trial's free models, dialled through the grader's gateway — the server never answers from itself. `GET /v1/models` answers in the wire the client's auth header names.

Routing is the score: candidates are the free models that meet the prompt's requirements (tools, vision, JSON mode, context length), ranked by the house's published rating, best first, and the reported `model` is always the id actually dialled. A refused gateway call is retried once when it looks transient, then the next candidate is tried; a 2xx whose text does not carry the prompt is failed over too. Refusals recovered from are reported on the response's `x-makefun-upstream-failures` header. Gateway answers are reassembled and re-emitted in the client's wire with the dialled model on every body and every streamed chunk, usage whole, and stop reasons mapped into the client's vocabulary. Every choice derives from the brief, the request and the gateway's answers — no wall clock, no randomness — so a replay walks the same path. One dependency-free `main.ts`, Node 22.

## Run it

```sh
node main.ts --list   # print the catalog document
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL, OPENAI_API_KEY)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=szkznt2e&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
