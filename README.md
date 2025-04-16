# Attention-based Sentiment Classification

This project explores the application of attention mechanisms in sentiment classification using a RNN model. Our experiments demonstrate that attention mechanisms enhance the model's ability to identify and focus on sentiment-relevant parts of sentences. Below are two visualization examples showing attention heatmaps for positive and negative sentiment analysis.

![Attention heatmap for positive sentiment](./assets/exmaple_positive.png)
_Attention heatmap visualization for a sentence containing positive sentiment_

![Attention heatmap for negative sentiment](./assets/example_negative.png)
_Attention heatmap visualization for a sentence containing negative sentiment_

## Dataset

This project utilizes the SST-2 (Stanford Sentiment Treebank) dataset, which consists of movie reviews annotated with binary sentiment labels. The dataset is sourced from the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html).

## Getting Started

### Clone the repository + Install dependencies

```bash
git clone https://github.com/weilantian/attention-based-sentiment-classifcation.git
cd attention-based-sentiment-classifcation
pip install -r requirements.txt
```

Experiment with the `sentiment_analysis_playground.ipynb` notebook to understand the model's performance and the attention mechanism.

## To Run the visualization

This following command can be used to run the visualization with the trained weights supplied. The sentences used for visualization can be configured under the config.py or specified under the `--sentences` argument.

```bash
python main.py visualize --checkpoint weights/2025-04-15_11-13-14__last.pth
```

## Project Structure

```
.
├── README.md                               # Project documentation
├── app/                                    # Application code (future expansion)
├── assets/                                 # Image assets for documentation
│   ├── attention_visualization.png
│   ├── example_negative.png
│   └── exmaple_positive.png
├── config.py                               # Configuration parameters
├── data/                                   # Data loading and processing
│   ├── __init__.py
│   └── dataset.py                          # Dataset loading and preparation
├── main.py                                 # Entry point for CLI commands
├── models/                                 # Model architecture definitions
│   ├── __init__.py
│   ├── attn_model.py                       # Attention mechanism implementation
│   └── model.py                            # Main sentiment classification model
├── requirements.txt                        # Project dependencies
├── sentiment_analysis_playground.ipynb     # Interactive notebook for exploration
├── training/                               # Training logic
│   ├── __init__.py
│   └── trainer.py                          # Model training implementation
├── utils/                                  # Utility functions
│   ├── __init__.py
│   ├── cli.py                              # Command-line interface
│   ├── tokenizer.py                        # Text tokenization
│   └── utils.py                            # Miscellaneous helpers
├── visualization/                          # Visualization tools
│   └── attention_viz.py                    # Attention visualization
└── weights/                                # Pre-trained model weights
    └── 2025-04-15_11-13-14__last.pth      # Trained model checkpoint
```

### Key Components

- **models/attn_model.py**: Implementation of the attention mechanism
- **models/model.py**: Full sentiment classification model with bidirectional GRU
- **visualization/attention_viz.py**: Tools for visualizing attention weights
- **sentiment_analysis_playground.ipynb**: Interactive notebook for experimenting with the model
- **data/dataset.py**: Loads and processes the SST-2 dataset
- **training/trainer.py**: Handles model training and evaluation

## Citation

If you use this code or model in your research, please cite:

```bibtex
@misc{attention-sentiment-classifier,
  author = {Lantian},
  title = {Attention-based Sentiment Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/weilantian/attention-based-sentiment-classification}}
}
```

For the SST-2 dataset:

```bibtex
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE.txt file for details.
