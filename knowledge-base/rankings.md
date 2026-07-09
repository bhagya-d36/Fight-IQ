# UFC Rankings

## How current is this ranking information?

This information is current as of July 2026. Rankings update frequently —
re-check current sources and re-run `python ingest.py` after editing this
file to keep rankings fresh.

## How do UFC rankings work?

Each UFC weight division has a top-15 ranking of contenders below the
champion, used to determine title eligibility and matchmaking. Fighters move
up or down based on their results, and a win over a higher-ranked opponent
generally moves a fighter up more than a win over a lower-ranked one.

## What changed with UFC rankings in 2026?

On June 20, 2026, the UFC replaced its long-standing media-panel voting
system with the "Meta UFC Rankings," a new model built in partnership with
Meta (the technology company). It is an Elo-based mathematical rating system,
not an AI system — it does not watch fights, but takes structured inputs
(who won, against whom, and by what method) and calculates a rating.

## How does the Meta UFC Rankings system calculate a fighter's position?

The system weighs wins and losses against the quality of the opponent faced,
applies recency weighting so recent results matter more than old ones, and
applies inactivity penalties for fighters who go long periods without
fighting. A dominant finish over a highly ranked opponent moves a fighter up
more than a close decision over an unranked one. Rankings update
automatically every Monday following each UFC event, with minimal human
intervention beyond basic eligibility rules.

## What happened to the pound-for-pound rankings?

The traditional pound-for-pound (P4P) rankings, which compared the best
fighters across all weight classes regardless of division, were discontinued
as part of the June 2026 transition to the Meta UFC Rankings system. Before
the change, fighters like Islam Makhachev, Justin Gaethje, and Merab
Dvalishvili were frequently cited at or near the top of most pound-for-pound
lists.

## Where can I check the latest official UFC rankings?

The UFC publishes its official, current rankings at ufc.com/rankings. Because
rankings update weekly, this knowledge base only tracks broad context about
how the system works, not exact current rankings positions.
