# Results

## biokg

|dataset|model|hidden_dim|mrr|hit@1|hit@3|hit@10|
|---|---|---|---|---|---|---|
|*reported*|TransE|2000|0.7452|-|-|-|
|*reported*|PairRE|200|0.8164|-|-|-
|default|TransE|2000|
|default (running)|PairRE|1000|0.7426|0.6476|0.8085|0.9152
|freeze relation (running)|TransE|2000|0.6037|0.4685|0.6954|0.8515
|no relation type (running)|TransE|2000|0.6082|0.4763|6966|0.8494

## wikikg2

|dataset|model|hidden_dim|mrr|hit@1|hit@3|hit@10|
|---|---|---|---|---|---|---|
|*reported*|TransE|600|0.4536|-|-|-|
|*reported*|PairRE|200|0.5289|-|-|-|
|default|TransE|100
|new split|TransE|

## fb15k

|dataset|model|hidden_dim|mrr|hit@1|hit@3|hit@10|
|---|---|---|---|---|---|---|
|*reported*|TransE|-|0.463|0.297|0.578|0.749|
|*reported*|PairRE|-|0.811|0.895|0.843|0.762|
|default|TransE|200|0.6675|0.5539|0.7499|0.8582

## fb15k237

|dataset|model|hidden_dim|mrr|hit@1|hit@3|hit@10|
|---|---|---|---|---|---|---|
|*reported*|TransE|-|0.294|-|-|0.465|
|*reported*|PairRE|-|0.351|0.256|0.387|0.544|
|default|TransE|200|0.4568|0.3355|0.5207|0.6809