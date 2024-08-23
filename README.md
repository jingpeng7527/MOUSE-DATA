# MOUSE-DATA
practice code for mice

```sh
pyenv install -s 3.9.5
pyenv virtualenv 3.9.5 ppo_imp
pyenv local ppo_imp

poetry init
poetry add gym torch stable_baselines3 tensorboard wandb // gym = 0.25.2

python ppo.py --track // upload to wandb

## pyenv ppo_imp doesn't work:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
