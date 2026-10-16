"""
THE LEG AND ITS VISUALS -- owned by Person 1.

Rendering machinery and camera work. Person 1 owns how the leg looks and how
the motion is filmed; the leg's own geometry, materials, lighting and cameras
live in model/leg.xml, which is also Person 1's.

The ground's appearance is NOT here -- GroundModel.update_visual owns that, so
Person 2 can change how deformation is drawn without touching this file.

    rec = Recorder(env.model, track="body_mass")
    rec.capture(env.data)
    rec.save("videos/rollout.mp4")
"""
import os

# Headless machines (lab servers, WSL, CI) have no X display and MuJoCo's
# default GLFW backend fails with "no OpenGL platform library". EGL works
# without a display. MUST be set before mujoco is imported; harmless on a
# laptop with a desktop session.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np          # noqa: E402
import mujoco               # noqa: E402


class Recorder:
    """Captures frames and writes a video.

    track : body name for the camera to follow, or None for the fixed
            `side` camera defined in leg.xml. The leg translates forward
            during stance, so a fixed camera loses it on longer episodes.
    """

    def __init__(self, model, width=640, height=480, track=None, distance=1.8):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.frames = []

        if track is None:
            self.cam = "side"
        else:
            self.track_bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, track)
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.distance = distance
            self.cam.azimuth = 90.0        # look along +x: a true side view
            self.cam.elevation = -8.0

    def capture(self, data):
        """Grab one frame. Call after every env.step()."""
        if not isinstance(self.cam, str):
            # Follow the body in x and z, keeping the leg centred as it vaults
            # forward over the foot.
            self.cam.lookat[:] = data.xpos[self.track_bid]
            self.cam.lookat[2] *= 0.6      # bias down so the foot stays in shot

        self.renderer.update_scene(data, camera=self.cam)
        frame = self.renderer.render()
        self.frames.append(frame)
        return frame

    def save(self, path, fps=20):
        """Write the video.

        100 frames at 20 fps = 5 s of slow motion. Real time would be 0.5 s --
        far too fast to see an impact.
        """
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)
        n = len(self.frames)
        self.frames = []
        return n

    def close(self):
        self.renderer.close()


def side_by_side(frames_a, frames_b):
    """Stack two rollouts horizontally for the learned-vs-baseline comparison.

    Run the SAME seed through both controllers. On average ground they look
    near-identical; on held-out soft ground one stays up and the other folds.
    That single clip is worth more in a presentation than any plot.
    """
    n = min(len(frames_a), len(frames_b))
    return [np.hstack([frames_a[i], frames_b[i]]) for i in range(n)]
