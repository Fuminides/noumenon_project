# Project Noumenon

This repo contains the necessary code to work with the idea of Noumenon applied to Social Network Analysis: the Semantic value. We have included both the code necessary to replicate our results and precomputed some of them in csv format. Those csv contains additional results not reported in the paper. (Mostly semantic values and similarities of actors not analyzed in the paper)

## Code

Four different .py files are included.

1. affinity.py contains the implementation of different affinity functions. We also included a GPU version of them, but it requires more testing.
2. complexity.py contains the implemention of the semantic value.
3. pipes.py contains the Pipe algorithm to compute the Semantic affinity.
4. speech_to_adjacency.py contains many functions to work with text. It contains functions to replicate the main experiments of the paper. (Which is probably what you want) Yo do not need to download the texts. If the program does not find them, it will download them automatically from project gutenberg.

## Precomputed results

In a separated folder I have added the results for the semantic value for all actors, and different semantic similarities. The files that contain "ss" contain semantic similarities. The "all" files contain the frequency of each word. Those who contain nothing are the semantic values.

## Dependencies

1. Numpy
2. Pandas
3. Scipy
4. Networkx

## Reference

Cite the preprint as:


@article{Fumanal-Idocin2021Sep,
	author = {Fumanal-Idocin, Javier and Cordón, Oscar and Dimuro, Graçaliz and Minárová, María and Bustince, Humberto},
	title = {{The Concept of Semantic Value in Social Network Analysis: an Application to Comparative Mythology}},
	journal = {arXiv},
	year = {2021},
	month = {Sep},
	eprint = {2109.08023},
	url = {https://arxiv.org/abs/2109.08023v1}
}

