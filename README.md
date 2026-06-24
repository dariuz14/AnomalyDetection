# Anomaly Detection
Questa repository contiene il codice che implementa il task di Out Of Distribution Detection che sfrutta l'errore di ricostruzione delle mappe di attenzione estratte da un ViT da parte di un modello generativo come criterio per discriminare tra In Distribution e Out Of Distribution. 

Il lavoro si ispira al paper:
> Leveraging Visual Attention for out-of-distribution Detection.

## Panoramica del Progetto
Il ViT utilizzato è [DeepFaune](https://www.deepfaune.cnrs.fr/en/) che è un modello pre-trained sulla fauna selvatica europea. Questo modello è necessario per l'estrazione delle mappe di attenzione su input provenienti da due dataset distinti:
- **WildCapture**: dataset ID.
- **Snapshot Kruger**: dataset OOD (https://lila.science/datasets/snapshot-kruger).

A partire dalle mappe di attenzione è stato addestrato un VAE (*Variational AutoEncoder*) convoluzionale allo scopo della sola ricostruzione.
L'errore di ricostruzione, dunque, serve da feature discriminatoria ID/ OOD.

## Struttura del progetto
```
AnomalyDetection/
├── AutoEncoder/
│   ├── vae.py                      # Variational Autoencoder
│   ├── train.py                    # Training dell'autoencoder
│   ├── test.py                     # Test dell'autoencoder
│   └── attention_maps_dataset.py   # Dataset per le attention maps
├── model.py                        # Vision Transformer (ViT)
├── dataset.py                      # Gestione del dataset
├── preprocess.py                   # Preprocessing dei dati
├── forward_wrapper.py              # Wrapper per il forward pass
├── config.py                       # Configurazione e iperparametri
├── utility.py                      # Funzioni di utilità
├── data/                           # Dataset (non incluso)
└── attention_maps/                 # Mappe di attenzione (non incluse)
```

## Utilizzo del codice
1. Caricamento del modello, estrazione e memorizzazione delle mappe di attenzione: `python utility.py` (il dataset delle immagini è gestito da `dataset.py`)
2. Addestramento del VAE sulle mappe di attenzione ID: `python AutoEncoder/train.py` (il dataset delle mappe è gestito da `AutoEncoder/attention_maps_dataset.py`)
3. Test del modello mediante l'errore di ricostruzione: `python AutoEncoder/test.py`
