# Snake RL Project

## Usage (Start Here)

### 1) Activate environment (Windows PowerShell)
```powershell
cd C:\Users\Mahdi\Desktop\RL_project
.\pygame_env\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Train agent (fast/headless)
```powershell
python scripts\train.py --games 2000 --eval-every 100
```

### 4) Train + render + plot + save terminal log
```powershell
python scripts\train.py --games 3000 --render --plot --eval-every 100 --eval-episodes 10 | Tee-Object train.log
```

### 5) Play manually (human mode)
```powershell
python scripts\play_human.py
```

## Overview
Reinforcement Learning Snake built with:
- `PyTorch` (DQN/Double DQN style training)
- `Pygame` (game environment and rendering)
- `Matplotlib` (optional live training curve)

## Project Structure
```text
RL_project/
  assets/
    fonts/
      arial.ttf
  notebooks/
    aa.ipynb
  scripts/
    train.py
    play_human.py
  src/
    snake_rl/
      __init__.py
      agent.py
      game.py
      helper.py
      model.py
      snake_game_human.py
  .gitignore
  README.md
  requirements.txt
```

## Notes
- Trained models are saved under `model/` during training.
- If your machine has Torch DLL issues, reinstall CPU wheels:
```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
