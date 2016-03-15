# motivation

## what is the goal
We seek to develop a computational method to automatically construct signaling pathways using both a background interactome and incomplete information of the targeted pathway through manual curation.  Ideally this method should outperform similar techniques which utilize only the background interactome without the manually curated information.  Performance being measured by ability to reconstruct a gold standard signalling pathway while minimizing the number of false positives.

# background
Signaling pathways are a fundamental aspect of systems biology, they control how a cell responds to external stimuli and ultimately what genes are expressed.  Understanding these complex systems guide what future biological experiments will be most illuminating, while also suggesting potential pharmaceutical targets.  There exist multiple databases which store information on cellular interactions, but curating these interactions into meaningful pathways has traditionally been a costly manual process.  There exist methods to computationally predict these pathways, though informative, they are far from perfect.

## why are you doing it
By leveraging computational methods on readily available information from interaction databases, we are able to reduce the need for painstaking manual curation.  Computationally constructed signalling pathways can accelarate the manual curation process, while also revealing potentially novel connections that human curators may have overlooked.  These constructed signalling pathways can then guide experimentalists on what interactions may be most interesting to study.

# related and previous research
Pathlinker
Response Net
Page Rank
PCSF
IPA
BowTie Builder

## what makes your angle different from similar approaches
While predicting pathways, we've previously only considered the confidence that an interaction takes place through experimental means (Bayesian approach that computes interaction probabilities based on sources of evidence).  Edge weights become more informative when we not only consider the confidence of the interaction, but whether it has already been currated for our pathway of interest manually.  

# approach

## how are you going about it

We use a preprocessing step of the interactome originally used in Pathlinker, altering edge weights based of if that interaction is passed as known or not.  Edges which are considered known have their edge weight (probability of interaction) increased, while edges which are unknown have their edge weight decreased.  As per the original Pathlinker, these weights are then negative log transformed, then a k-shortest-paths algorithm is then run on this resulting interactome utilizing a super-source and super-target.
We intend to use multiple weighting schemes to see which has the best performance on the wnt pathway.  It should be noted that the precision recall curve calculation must be altered from the original Pathlinker's formulation.  Recall of information we already passed the algorithm is not informative, thus we should only use the portion of the known wnt pathway we witheld from Pathlinker 2.0 as the gold standard.

# data

## what progress have you made so far

`python PathLinker.py`

# any preliminary results

```python
print("hello world")
```
