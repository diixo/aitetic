# aitetic

Action as intention ("tense": "present"):
```json
{
  "example": "The company is going to close down its factory next year.",
  "annotation": [
    {
      "action_text": "close down",
      "action": "close down",
      "tense": "present",
      "tense_aspect": "simple",
      "time_relation": "future",
      "modal_txt": "is going to",
      "subject": "The company",
      "object": "its factory",
      "prep": null,
      "prep_object": null,
      "time_expr": "next year"
    }
  ]
}
```

Action in future ("tense": "future"):
```json
{
  "example": "The company will close down its factory next year.",
  "annotation": [
    {
      "action_text": "close down",
      "action": "close down",
      "tense": "future",
      "tense_aspect": "simple",
      "time_relation": "future",
      "modal_txt": "will",
      "subject": "The company",
      "object": "its factory",
      "prep": null,
      "prep_object": null,
      "time_expr": "next year"
    }
  ]
}
```

```json
{
  "example": "He fell off the bicycle and hurt his leg.",
  "annotation": [
    {
      "action_text": "fell off",
      "action": "fall off",
      "tense": "past",
      "tense_aspect": "simple",
      "time_relation": "past",
      "modal_txt": null,
      "subject": "He",
      "object": null,
      "prep": "off",
      "prep_object": "the bicycle"
    },
    {
      "action_text": "hurt",
      "action": "hurt",
      "tense": "past",
      "tense_aspect": "simple",
      "time_relation": "past",
      "modal_txt": null,
      "subject": "He",
      "object": "his leg",
      "prep": null,
      "prep_object": null
    }
  ]
}
```
