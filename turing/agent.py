from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import enum

class CommandEnvAction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SET_VALUE_BENEATH = "set_value_beneath"

class OperatorsBin(Enum):
    COPY = "copy"
    # ADD_TO_STACK = "add_to_stack"


class DefaultRegistry(Enum) :
    TOP = "top"
    TOP_RIGHT = "top_right"
    RIGHT = "right"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM = "bottom"
    BOTTOM_LEFT = "bottom_left"
    LEFT = "left"
    TOP_LEFT = "top_left"
    BENEATH = "beneath"
    NB_STEPS = "nb_steps"
    X = "x"
    Y = "y"

@dataclass
class ProgramLine :
    is_env_action: bool
    command: CommandEnvAction | OperatorsBin
    arg1 : Optional[int | str]
    arg2 : Optional[int | str]

class Agent :
    def __init__(
        self,
        program : List[ProgramLine],
        initial_memory : Optional[Dict[str, int]] = None,
    ) -> None:
        self.program = program
        self.line = 0
        self.step = -1
        self.memory = initial_memory if initial_memory is not None else {}
        registries = {k: 0 for k in DefaultRegistry.__members__.keys()}
        self.memory.update(registries)

    def act(self, observation: np.ndarray) -> Tuple[CommandEnvAction, Optional[int], Optional[int]]:
        self.step += 1
        if self.line >= len(self.program):
            raise IndexError("Program counter out of bounds")

        # write observation to memory, obeservation os 3x3 grid of what is seen round agent rught now
        self.memory[DefaultRegistry.TOP.name] = int(observation[0, 1])
        self.memory[DefaultRegistry.TOP_LEFT.name] = int(observation[0, 0])
        self.memory[DefaultRegistry.TOP_RIGHT.name] = int(observation[0, 2])
        self.memory[DefaultRegistry.RIGHT.name] = int(observation[1, 2])
        self.memory[DefaultRegistry.BOTTOM.name] = int(observation[2, 1])
        self.memory[DefaultRegistry.BOTTOM_LEFT.name] = int(observation[2, 0])
        self.memory[DefaultRegistry.BOTTOM_RIGHT.name] = int(observation[2, 2])
        self.memory[DefaultRegistry.LEFT.name] = int(observation[1, 0])
        self.memory[DefaultRegistry.BENEATH.name] = int(observation[1, 1])
        self.memory[DefaultRegistry.NB_STEPS.name] = self.step

        program_line : ProgramLine = self.program[self.line]
        if program_line.is_env_action:            
            match program_line.command:
                case CommandEnvAction.UP:
                    return program_line.command, None, None
                case CommandEnvAction.DOWN:
                    return program_line.command, None, None
                case CommandEnvAction.LEFT:
                    return program_line.command, None, None
                case CommandEnvAction.RIGHT:
                    return program_line.command, None, None
                case CommandEnvAction.SET_VALUE_BENEATH:
                    assert isinstance(program_line.arg1, int)
                    return program_line.command, program_line.arg1, None
                case _:
                    raise NotImplementedError(f"Unknown command: {program_line.command}")
        else : 
            match program_line.command:
                case OperatorsBin.COPY:
                    assert isinstance(program_line.arg1, str)
                    assert isinstance(program_line.arg2, str)
                    self.memory[program_line.arg2] = self.memory[program_line.arg1]
                case _:
                    raise NotImplementedError(f"Unknown command: {program_line.command}")

        self.line += 1


class Env :
    def __init__(
        self,
        field : np.ndarray,
        agent : Agent
    ) -> None:
        self.field = field
        self.agent_pos = (0, 0)
        self.agent = agent

    def get_agent_observation(self) -> np.ndarray:
        field_padded = np.pad(self.field, pad_width=1, mode='constant', constant_values=-1)
        x, y = self.agent_pos
        return field_padded[x:x+3, y:y+3]

    def step(self) -> None:
        action, arg1, arg2 = self.agent.act(self.get_agent_observation())
        self.act(action, arg1, arg2)

    def act(self, action: CommandEnvAction, arg1: Optional[int], arg2: Optional[int]) -> None:
        if action == CommandEnvAction.UP:
            self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == CommandEnvAction.DOWN:
            self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == CommandEnvAction.LEFT:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == CommandEnvAction.RIGHT:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == CommandEnvAction.SET_VALUE_BENEATH:
            assert arg1 is not None
            self.field[self.agent_pos[0], self.agent_pos[1]] = arg1