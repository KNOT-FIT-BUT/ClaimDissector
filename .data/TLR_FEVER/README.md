# TLR-FEVER dataset
## processed dataset
is available in file `.data/TLR_FEVER/TLRFEVER.jsonl`. The format is the following
```python
{
   "id":13630, # id of the sample, compatible with fever shared_task_dev.jsonl set
   "data":[  # relevant sentences from FEVER dataset
      "Melancholia is a 2011 science-fiction drama-psychological thriller film written and directed by Lars von Trier and starring Kirsten Dunst, Charlotte Gainsbourg, Alexander Skarsgård, Cameron Spurr, and Kiefer Sutherland.",
      "Lars von Trier ( Lars Trier, 30 April 1956 ) is a Danish film director and screenwriter."
   ],
   "label":[  # relevant tokens selected by up to 4 annotators
      "Melancholia directed by Lars von Trier Lars von Trier Trier Danish screenwriter .",
      "Lars von Trier Danish film screenwriter .",
      "Melancholia directed by Lars von Trier Danish screenwriter .",
      "Melancholia directed by Lars von Trier Lars von Trier is a Danish film director and screenwriter ."
   ], 
   "titles":"Melancholia (2011 film)|Lars von Trier", # titles of the Wikipedia articles, from which the relevant sentences come from
   "claim":"Melancholia's director was a Danish screenwriter.", # original claim from FEVER dataset
   "claim_label":"SUPPORTS", # original label from FEVER dataset
   "enumkey":35 # you can ignore this, serves for reverse mapping to raw dataset, should it every be necessary
}
```

## raw dataset
is available in files `.data/TLR_FEVER/raw_annotations_[1-4].jsonl`.
These contain raw annotations from 4 annotators. The format is the following
```python
{
   "id":13630, 
    # relevant sentences, concatenated with '|'
   "data":"Melancholia is a 2011 science-fiction drama-psychological thriller film written and directed by Lars von Trier and starring Kirsten Dunst, Charlotte Gainsbourg, Alexander Skarsgård, Cameron Spurr, and Kiefer Sutherland.|Lars von Trier ( Lars Trier, 30 April 1956 ) is a Danish film director and screenwriter.",
   "label":[ # character spans from data, annotated by annotators as relevant
      [
         0,
         11,
         "RELEVANT"
      ],
      [
         84,
         109,
         "RELEVANT"
      ],
      [
         270,
         276,
         "RELEVANT"
      ],
      [
         295,
         307,
         "RELEVANT"
      ]
   ],
   "titles":"Melancholia (2011 film)|Lars von Trier",
   "claim":"Melancholia's director was a Danish screenwriter.",
   "claim_label":"SUPPORTS"
}
```
Uncommented fields have same semantics as for processed dataset.

## Evaluation
Inspect method [eval_wlann_for_threshold](https://github.com/KNOT-FIT-BUT/ClaimDissector/blob/d511cf48987be3a050f6f2d8ccdd30c4825f9283/src/fact_checking/verifier/lrm_verifier_trainer.py#L3745) to see how evaluation is implemented.
