from PyQt6 import QtWidgets

import torch
import torch.nn as nn

from data_collector import DataCollectionUI

# ---------- Modèle (même archi que l'entraînement) ----------
class ControllerMLP(nn.Module):
    def __init__(self, in_dim=16, hidden=128, out_dim=4, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim)   # logits (pas de sigmoid)
        )
    def forward(self, x): return self.net(x)


class ExampleNNMsgProcessor:
    def __init__(self, ckpt_path="controller_multilabel.pt",
                 thresholds=(0.5, 0.5, 0.5, 0.5),
                 command_names=("forward", "back", "left", "right")):

        # Device: CPU par défaut (évite surcoût transfert GPU si petit modèle)
        self.device = torch.device("cpu")

        # Charge checkpoint (+ normalisation)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = ControllerMLP(16, 128, 4, p_drop=0.0).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.mean = ckpt["mean"].to(self.device)  # shape [1,16]
        self.std  = ckpt["std"].to(self.device)   # shape [1,16]

        # Seuils par sortie (tu peux ajuster)
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)

        # Noms de commandes (adapter si ton DataCollectionUI attend autre chose)
        self.cmd_names = list(command_names)

        # État courant (pour n'envoyer que les transitions)
        self.state = {name: False for name in self.cmd_names}

    def _features_from_message(self, message):
        speed = float(message.car_speed)
        dists = list(message.raycast_distances)
        if len(dists) != 15:
            # garde-fou: tronque ou pad si besoin
            if len(dists) > 15:
                dists = dists[:15]
            else:
                dists = dists + [dists[-1]] * (15 - len(dists))
        x = torch.tensor([[speed] + list(map(float, dists))], dtype=torch.float32, device=self.device)
        # normalisation identique à l'entraînement
        x = (x - self.mean) / self.std
        return x

    def _logits_to_commands(self, logits):
        # logits -> probabilités -> décisions 0/1 selon seuils
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [4]
        decisions = [bool(p > thr) for p, thr in zip(probs, self.thresholds)]

        # règle anti-conflits: si left & right -> garde le plus probable
        if decisions[2] and decisions[3]:
            decisions[2] = probs[2] > probs[3]
            decisions[3] = probs[3] > probs[2]

        # règle anti-conflits: forward & back -> garde le plus probable
        if decisions[0] and decisions[1]:
            decisions[0] = probs[0] > probs[1]
            decisions[1] = probs[1] > probs[0]

        return probs, decisions

    def nn_infer(self, message):
        # Prépare features
        x = self._features_from_message(message)
        # Inférence
        with torch.no_grad():
            logits = self.model(x)
        probs, decisions = self._logits_to_commands(logits)

        # Construit la liste d'ordres à envoyer (uniquement transitions)
        to_send = []
        for name, new_flag in zip(self.cmd_names, decisions):
            if self.state[name] != new_flag:
                self.state[name] = new_flag
                to_send.append((name, new_flag))

        # (Optionnel) debug
        # print(f"probs={probs.round(3)} decisions={decisions} to_send={to_send}")

        return to_send

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)
        for command, start in commands:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    # Instancie le “cerveau” NN
    nn_brain = ExampleNNMsgProcessor(
        ckpt_path="controller_multilabel.pt",           
        thresholds=(0.5, 0.5, 0.5, 0.5),                # ex: frein un peu plus strict
        command_names=("forward", "back", "left", "right")
    )

    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()
    app.exec()
