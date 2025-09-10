Im PyTorch Dataset wird entschieden wie der Output ist:
- erm_dataset.py Output: X_cont, X_cat, y, domain_id
- episodic_dataset.py Output: query, ...

Im PyTorch Lightning DataModule wird entschieden wie der Split ist:
- base.py Split: 70/20/10 (Interpolation)
- loco.py Split: Entweder KMeans oder custom split nach konfiguration

Im DataModule wird mitgegeben welcher Dataset-Typ benötigt wird und dann wird dadurch entschieden welche Dataset Klasse benutzt wird.

Im model.py muss folgendes implementiert werden:
- Normale loss-Funktion für Interpolation
- REx loss-Funktion als Addon wenn ausgewählt (benötigt domain_id)
- Episodic training logik