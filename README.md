# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/KaivanETH/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/KaivanETH/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

An OpenAI- and Anthropic-compatible aggregation server that routes each request to the best free model that can serve it, with failover across ten free-tier providers.

## Run it

```sh
# a local practice trial: the kit's env, this entry, then the kit's scorer
/root/entries/bin/run_trial_fma.sh trial-0 <this repo>

# inside a trial the sandbox sets MAKEFUN_PROMPT_FILE (the brief) and MAKEFUN_ENV_URL (the env)
MAKEFUN_PROMPT_FILE=/input/prompt.txt MAKEFUN_ENV_URL=http://store:8080 node main.ts

# print the catalog document it registers, then exit 0
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=73rfmaq9&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
