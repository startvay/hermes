"""
军师旅制多Agent系统
5个Agent：司令部、参谋部、侦察连、工兵连、通信连
"""

from .commander.commander_agent import CommanderAgent
from .staff.staff_agent import StaffAgent
from .scout.scout_agent import ScoutAgent
from .builder.builder_agent import BuilderAgent
from .comms.comms_agent import CommsAgent

__all__ = [
    "CommanderAgent",
    "StaffAgent",
    "ScoutAgent",
    "BuilderAgent",
    "CommsAgent",
]
