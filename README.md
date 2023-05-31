# DQNAtari

This project implements deep q networks to solve atari games in openai gym.

Recommended to use GPU if possible else it is extremely slow.

To run program use the command: python Atari_dqn.py -e "environment_name"

environment_name is the atari game to be solved. Eg "Breakout-v0" Full list of environments can be found [here](https://gym.openai.com/envs/#atari). 
  
Be sure to use non Ram versions as the program uses pixels as inputs instead.
  
The full list of arguments available can be accessed with python Atari_dqn.py -h 
  
