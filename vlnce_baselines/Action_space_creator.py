from gym.spaces import Discrete

class ActionSpace(Discrete):
    def __init__(self,actions):
        self.actions = actions
        super(ActionSpace, self).__init__(len(actions))

    def __getitem__(self,key):
        return self.actions[key]

actions = {
    0:'STOP',
    1:'MOVE_FORWARD',
    2:'TURN_LEFT',
    3:'TURN_RIGHT'
}

action_space = ActionSpace(actions)

