#%%
import gym

#%%
env = gym.make('SpaceInvadersNoFrameskip-v4')

#%%
ACTIONS = env.action_space.n
def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

#%%
env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

#%%
obser = env.reset()

#%%
obser

#%%
obser, r, done, info = env.step(1)

#%%
env.render()

#%%
done