# Prediction Tendency
This repository contains code to calculate prediction tendencies. As described in [here](https://academic.oup.com/cercor/article/33/11/6608/6975346?login=true.).

### In brief 
#### experimental setup
We manipulated the entropy of a sequence of events (4 different pure tones; f1: 440 Hz, f2: 587 Hz, f3: 782 Hz, and f4: 1,043 Hz) using markov chains. In one condition events were "ordered" i.e. the transitional probability from f1 → f2, f2 → f3, f3 → f4 and f4 → f1 was 75% (self repetitions had a probability of 25%; e.g. f1 → f1). In another condition events were "random" i.e. the probability of forward transitions and self repetitions was 25%.
#### estimating prediction tendencies
We trained a classifier on the forward transitions of the "ordered" condition in a time-resolved manner and tested on the self-repetitions of the "random" and "ordered" conditions This resulted in classifier decision values (dvals) for every possible sound frequency (d1, d2, d3, and d4) of all test trials (t) which were then transformed into corresponding transitions with respect to the preceding sound (t − 1) (e.g. d1(t) | f1(t − 1) “dval for f1 at trial t given that f1 was presented at trial t − 1” → repetition, d2(t) | f1(t − 1) → forward,...). We quantify prediction tendency as the classifiers pre-stimulus decision in favor of a forward transition in an ordered compared with a random context. For more info [see](https://academic.oup.com/cercor/article/33/11/6608/6975346?login=true.)

### Related scientific findings
We have succesfully linked this measure to speech processing ([see](https://academic.oup.com/cercor/article/33/11/6608/6975346?login=true.)) (recently [replicated](https://www.biorxiv.org/content/biorxiv/early/2023/06/29/2023.06.27.546746.full.pdf))

## Reference
The mastermind behind the algorithm is [Juliane Schubert](https://twitter.com/julianeschuber2).

The Python code was written by me.

If you use the code in your work please cite:

Schubert, J., Schmidt, F., Gehmacher, Q., Bresgen, A., & Weisz, N. (2023). Cortical speech tracking is related to individual prediction tendencies. Cerebral Cortex, 33(11), 6608-6619. ([link](https://academic.oup.com/cercor/article/33/11/6608/6975346))
