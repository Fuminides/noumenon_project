# Project Noumenon

This repo contains the necessary code to work with the idea of Noumenon applied to Social Network Analysis: the Semantic value. We have included both the code necessary to replicate our results and precomputed some of them in csv format. Those csv contains additional results not reported in the paper. (Mostly semantic values and similarities of actors not analyzed in the paper)

We hope to publish it sometime this year, and while a preprint is coming soon, you can check this work meanwhile:
@article{fumanal2020community,
  title={Community detection and social network analysis based on the Italian wars of the 15th century},
  author={Fumanal-Idocin, Javier and Alonso-Betanzos, A and Cord{\'o}n, O and Bustince, H and Min{\'a}rov{\'a}, M},
  journal={Future Generation Computer Systems},
  volume={113},
  pages={25--40},
  year={2020},
  publisher={North-Holland}
}

We will update accordingly when something else is released.

## Code

Four different .py files are included.

1. affinity.py contains the implementation of different affinity functions. We also included a GPU version of them, but it requires more testing.
2. complexity.py contains the implemention of the semantic value.
3. pipes.py contains the Pipe algorithm to compute the Semantic affinity.
4. speech_to_adjacency.py contains many functions to work with text. It contains functions to replicate the main experiments of the paper. (Which is probably what you want) Yo do not need to download the texts. If the program does not find them, it will download them automatically from project gutenberg.

## Precomputed results

In a separated folder I have added the results for the semantic value for all actors, and different semantic similarities. The files that contain "ss" contain semantic similarities. The "all" files contain the frequency of each word. Those who contain nothing are the semantic values.

## Reference

We hope to publish at least the preprint soon.
