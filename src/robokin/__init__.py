"""robokin: kinematics, trajectory planning and collision checking for serial robots."""

from .collision import CollisionModel, LinkCapsule, PathCheck, Sphere, check_motion, check_samples
from .dh import DHLink, IKSolution, SerialRobot
from .ik import IKResult, ik_dls, ik_numeric
from .program import Program
from .robots import AnalyticRobot, Planar3R, Puma560, Scara
from .trajectory import (
    CartesianTrajectory,
    JointTrajectory,
    QuinticProfile,
    SampledJointTrajectory,
    TrapezoidProfile,
    plan_lin,
    plan_ptp,
    track_lin,
)
from .workspace import (
    WorkspaceSamples,
    hit_or_miss_volume,
    occupied_area,
    occupied_volume,
    sample_workspace,
    two_link_area,
)

__version__ = "0.1.0"

__all__ = [
    "AnalyticRobot",
    "CartesianTrajectory",
    "CollisionModel",
    "DHLink",
    "IKResult",
    "IKSolution",
    "JointTrajectory",
    "LinkCapsule",
    "PathCheck",
    "Planar3R",
    "Program",
    "Puma560",
    "QuinticProfile",
    "SampledJointTrajectory",
    "Scara",
    "SerialRobot",
    "Sphere",
    "TrapezoidProfile",
    "WorkspaceSamples",
    "check_motion",
    "check_samples",
    "hit_or_miss_volume",
    "ik_dls",
    "ik_numeric",
    "occupied_area",
    "occupied_volume",
    "plan_lin",
    "plan_ptp",
    "sample_workspace",
    "track_lin",
    "two_link_area",
]
