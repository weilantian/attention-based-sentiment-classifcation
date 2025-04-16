# Attention-based Sentiment Classification

This project explores the application of attention mechanisms in sentiment classification using a RNN model. Our experiments demonstrate that attention mechanisms enhance the model's ability to identify and focus on sentiment-relevant parts of sentences. Below are two visualization examples showing attention heatmaps for positive and negative sentiment analysis.

![Attention heatmap for positive sentiment](./assets/exmaple_positive.png)
_Attention heatmap visualization for a sentence containing positive sentiment_

![Attention heatmap for negative sentiment](./assets/example_negative.png)
_Attention heatmap visualization for a sentence containing negative sentiment_

## Dataset

This project utilizes the SST-2 (Stanford Sentiment Treebank) dataset, which consists of movie reviews annotated with binary sentiment labels. The dataset is sourced from the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html).

## To Run the visualization

This following command can be used to run the visualization with the trained weights supplied. The sentences used for visualization can be configured under the config.py or specified under the `--sentences` argument.

```bash
python main.py visualize --checkpoint weights/2025-04-15_11-13-14__last.pth
```
