
# Cliff Walking: Q-Learning & SARSA

This project demonstrates **Q-Learning** and **SARSA** reinforcement learning algorithms on the classic `CliffWalking-v0` environment from OpenAI Gym. It also includes a simple OpenCV-based visualization of agent movements.

---

## 📂 Files Overview

| File         | Description                                  |
|--------------|----------------------------------------------|
| `qlearning.py` | Trains an agent using Q-Learning and saves Q-table |
| `sarsa.py`     | Trains an agent using SARSA and saves Q-table      |
| `index.py`     | Visualizes agent movement using saved Q-table or random policy (OpenCV based) |

---

## 📊 Algorithms Used

- **Q-Learning**: Off-policy TD control algorithm
- **SARSA**: On-policy TD control algorithm
- Both use an ε-greedy policy

---

## ⚙️ Training Parameters

- `EPSILON = 0.1` (exploration)
- `ALPHA = 0.1` (learning rate)
- `GAMMA = 0.9` (discount)
- `NUM_EPISODES = 500`

---

## 🧠 How to Run

1. **Train with Q-Learning**
   ```bash
   python qlearning.py
   ```

2. **Train with SARSA**
   ```bash
   python sarsa.py
   ```

3. **Visualize Agent Movement**
   - Modify `index.py` to load either `qlearning_qtable.pkl` or `sarsaqtable.pkl`
   - Then run:
     ```bash
     python index.py
     ```

---

## 🖼️ Visualization

- Uses OpenCV to show a 4x12 grid
- Agent (`A`) moves step-by-step
- Cliff region is marked in **purple**
- Goal is marked as `G`

---

## 📦 Requirements

- `gym`
- `opencv-python`
- `numpy`
- Python ≥ 3.8

Install via:
```bash
pip install gym opencv-python numpy
```

---

## 📘 Reference

Environment:
- [CliffWalking-v0 Docs](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)

---

## ✨ Author

Made by Seika

```
