"""Generate the Franka gripper collision spheres used by cuTAMP for grasp filtering.

The spheres are cuRobo's own sphere model (franka_mesh.yml, plus its collision_sphere_buffer)
for the gripper links, re-expressed in the tool frame. Edit the finger spheres in franka_mesh.yml
and rerun this to keep the two in sync.

--tool-z must match the tool_from_ee offset in load_panda_container, otherwise the spheres sit
in a different frame to the grasps they are checked against.

Usage: python -m cutamp.robots.gen_franka_gripper_spheres --tool-z 0.1884
"""

import argparse
from pathlib import Path

import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.transform import quaternion_to_matrix
from curobo.types.robot import RobotConfig
from cutamp.robots.franka import franka_curobo_cfg

GRIPPER_LINKS = ("panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger")
DEFAULT_OUT = Path(__file__).parent / "assets" / "franka_gripper_spheres.pt"


def gripper_spheres_in_tool_frame(tool_z: float) -> Float[torch.Tensor, "num_spheres 4"]:
    """Spheres of the gripper links, expressed in the tool frame (z-up, origin at the TCP)."""
    robot_cfg = RobotConfig.from_dict(franka_curobo_cfg()["robot_cfg"])
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    kin_cfg = kin_model.kinematics_config

    # Gripper-link spheres are rigid relative to the hand, so any valid configuration works
    q = kin_cfg.joint_limits.position.mean(dim=0)[None]
    state = kin_model.get_state(q)
    spheres = state.get_link_spheres()[0]

    idxs = [
        torch.nonzero(kin_cfg.link_sphere_idx_map == kin_cfg.link_name_to_idx_map[link]).view(-1)
        for link in GRIPPER_LINKS
    ]
    spheres = spheres[torch.cat(idxs)]

    # Base -> ee (panda_hand), then ee -> tool. Must match load_panda_container's tool_from_ee.
    ee_from_world = torch.linalg.inv(state.ee_pose.get_matrix()[0])
    gripper_down_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=spheres.device, dtype=spheres.dtype)
    tool_from_ee = torch.eye(4, device=spheres.device, dtype=spheres.dtype)
    tool_from_ee[:3, :3] = quaternion_to_matrix(gripper_down_quat)[0]
    tool_from_ee[2, 3] = tool_z

    centers = torch.cat([spheres[:, :3], torch.ones_like(spheres[:, :1])], dim=1)
    centers = centers @ (tool_from_ee @ ee_from_world).T
    return torch.cat([centers[:, :3], spheres[:, 3:4]], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-z", type=float, required=True, help="TCP offset from panda_hand in metres")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    spheres = gripper_spheres_in_tool_frame(args.tool_z)
    torch.set_printoptions(precision=4, sci_mode=False)
    print(spheres)
    torch.save(spheres.cpu(), args.out)
    print(f"Wrote {spheres.shape[0]} spheres to {args.out}")


if __name__ == "__main__":
    main()
