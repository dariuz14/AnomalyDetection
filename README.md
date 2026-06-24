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
