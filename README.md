# PDR Detection Drone - Projet de Recherche

Projet de detection de drones sur le dataset FRED (Florence RGB-Event Drone Dataset), avec plusieurs implementations pour comparaison.

## Implementations

- `snn_yolov1_custom_fred.ipynb`
	- Implementation SNN-YOLO custom (base snnTorch, architecture type YOLOv1 adaptee).
- `spikeyolo_biclab_fred.ipynb`
	- Pipeline SpikeYOLO (BICLab) avec export FRED au format YOLO, entrainement, evaluation et visualisations.
- `spikeyolo_training_pipeline_fred.ipynb`
	- Variante etape-par-etape de l'entrainement SpikeYOLO (preparation, debug, training, metriques finales).
    - Lien du modèle pushed sur HF : https://huggingface.co/Hipo17/spikeyolo-fred/tree/main
- `yolov8n_finetuned_fred.ipynb`
	- Baseline YOLOv8n fine-tune pour comparer ANN vs SNN.

## Ce qu'on a fait

1. Preparation des donnees FRED en format YOLO (`images/` + `labels/`).
2. Entrainement de plusieurs modeles (SpikeYOLO, SNN-YOLO custom, YOLOv8n).
3. Evaluation sur validation/test avec metriques de detection.
4. Visualisation des predictions et des courbes d'entrainement.

## Resultats (rapide)

Les notebooks produisent principalement :

- `mAP@0.5`
- `mAP@0.5:0.95`
- Precision / Recall / F1
- Courbes de loss et de mAP
- Visualisations des bounding boxes (prediction vs ground truth)

Les artefacts sont generes pendant l'execution des notebooks (modeles `.pt`, figures `.png`, tableaux CSV de metriques).

## Lancer le projet

1. Ouvrir le notebook de l'implementation voulue.
2. Executer les cellules dans l'ordre.
3. Recuperer les metriques et artefacts en fin de notebook.

