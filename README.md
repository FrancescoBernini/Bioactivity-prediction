# Plant-Derived Molecule Bioactivity Prediction

Final project for the AI Laboratory course. The project combines molecular mass-spectrometry embeddings, protein sequence representations, dimensionality reduction, and supervised classification to study molecule–target activity.

**[Explore the analysis notebook](main.ipynb)** | [Browse experiment results](results/)

## Data sources and availability

The datasets and prepared feature files are **not included** in this repository. The notebook expects project-prepared CSVs at these paths:

```text
dataset_bioactivities/
	bioactivities/dataset_bioactivities_negative.csv
	mass_spectra_embeddings/ALL_GNPS_negative_ionization_DATASET.csv
	protein_embeddings/protein_embeddings_UniRep64.csv
	protein_embeddings/protein_embeddings_UniRep256.csv
	protein_embeddings/protein_embeddings_UniRep1900.csv
```

The notebook loads these CSVs directly; it does not generate the Spec2Vec or UniRep embeddings. It then joins the tables, removes unused columns, derives active/inactive labels, scales the features, and evaluates models. The repository does not document the complete procedure used to produce the prepared CSVs or generate the embeddings, so the source links alone may not be sufficient to recreate those exact files and results.

| Source | Used for | License or citation |
| --- | --- | --- |
| [Papyrus 05.5 dataset](https://doi.org/10.5281/zenodo.7019874) | Bioactivity data and UniRep protein representations | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [GNPS fragmentation spectra, accessed 2020-05-11](https://doi.org/10.5281/zenodo.3979010) | Source spectra for the molecular mass-spectrometry features | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [Spec2Vec publication](https://doi.org/10.1371/journal.pcbi.1008724) | Method used to create molecular spectral embeddings | Method citation; not a data license |
| [UniRep publication](https://doi.org/10.1038/s41592-019-0598-1) | Protein sequence representation method | Method citation; not a data license |

## Run the analysis

1. Obtain the source data from the links above and recreate the prepared CSVs at the expected paths.
2. Install the Python packages imported by the notebook. Exact dependency versions are not pinned in this repository.
3. Open [`main.ipynb`](main.ipynb) in Jupyter or VS Code and run its cells in order.

The notebook contains the project rationale, data exploration, preprocessing decisions, model evaluation, and discussion of results. Full cross-validation can be computationally and memory intensive; the original experiments used University of Parma HPC resources.

## Results

Saved experiment summaries are available under [`results/`](results/), including `best_models.csv` and `selection.csv`. The notebook provides the context for interpreting these comparisons; high recall alone does not imply strong overall classification, so MCC and the other reported metrics should be considered as well.
