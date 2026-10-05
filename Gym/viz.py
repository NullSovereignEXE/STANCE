"""
LEG VISUALIZATION -- Person 1

Handles rendering and video recording for the prosthetic-leg model.

Physics is NOT modified here.

Usage:
    rec = Recorder(env.model, track="body_mass")

    frame = rec.capture(env.data)

    rec.save("videos/rollout.mp4")

    rec.close()
"""

import os

# ------------------------------------------------------------
# HEADLESS RENDERING
# ------------------------------------------------------------
# Must be set BEFORE importing mujoco.
#
# EGL allows rendering on machines without a normal display,
# such as lab servers / CI.
#
# On your Ubuntu desktop this can still be used for off-screen
# video rendering.

os.environ.setdefault("MUJOCO_GL", "egl")


import numpy as np
import mujoco


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    """
    MuJoCo video recorder.

    Parameters
    ----------
    model:
        MuJoCo MjModel.

    width, height:
        Output image resolution.

    track:
        Name of body for camera to follow.

        track=None
            -> use fixed camera "side" from leg.xml

        track="body_mass"
            -> camera follows the body mass.

    distance:
        Tracking-camera distance from the model.
    """

    def __init__(
        self,
        model,
        width=640,
        height=480,
        track=None,
        distance=1.8,
    ):

        self.model = model

        # ----------------------------------------------------
        # Create off-screen MuJoCo renderer
        # ----------------------------------------------------

        self.renderer = mujoco.Renderer(
            model,
            height=height,
            width=width,
        )

        # Store captured video frames
        self.frames = []


        # ----------------------------------------------------
        # FIXED CAMERA
        # ----------------------------------------------------

        if track is None:

            # This camera must exist in leg.xml
            self.cam = "side"

            self.track_bid = None


        # ----------------------------------------------------
        # TRACKING CAMERA
        # ----------------------------------------------------

        else:

            self.track_bid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                track,
            )

            if self.track_bid == -1:

                raise ValueError(
                    f"Cannot track body '{track}': "
                    "body not found in MuJoCo model."
                )


            self.cam = mujoco.MjvCamera()

            mujoco.mjv_defaultCamera(
                self.cam
            )

            self.cam.type = (
                mujoco.mjtCamera.mjCAMERA_FREE
            )

            self.cam.distance = distance

            # Side view of X-Z motion.
            #
            # If this appears mirrored on your machine,
            # change 90 -> -90.
            self.cam.azimuth = 90.0

            self.cam.elevation = -8.0


    # ========================================================
    # CAPTURE ONE FRAME
    # ========================================================

    def capture(self, data):

        """
        Capture one RGB frame from the current MuJoCo state.

        Call this AFTER env.step() or mj_step().
        """


        # ----------------------------------------------------
        # Update tracking camera
        # ----------------------------------------------------

        if self.track_bid is not None:

            body_position = data.xpos[
                self.track_bid
            ]

            self.cam.lookat[:] = body_position

            # Bias the camera slightly downward so both
            # the body and foot remain visible.
            self.cam.lookat[2] *= 0.60


        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        self.renderer.update_scene(
            data,
            camera=self.cam,
        )

        frame = self.renderer.render()


        # Make a copy because renderer memory may be reused
        frame = frame.copy()

        self.frames.append(frame)

        return frame


    # ========================================================
    # SAVE VIDEO
    # ========================================================

    def save(
        self,
        path,
        fps=20,
    ):

        """
        Save all captured frames as an MP4 video.

        Example:

            rec.save(
                "videos/rollout.mp4",
                fps=20
            )
        """

        if len(self.frames) == 0:

            raise RuntimeError(
                "No frames have been captured."
            )


        import imageio.v2 as imageio


        # Create output directory if necessary
        directory = os.path.dirname(path)

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )


        imageio.mimsave(
            path,
            self.frames,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
        )


        number_of_frames = len(
            self.frames
        )


        # Clear frames after saving
        self.frames = []


        return number_of_frames


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        """
        Release MuJoCo rendering resources.
        """

        if self.renderer is not None:

            self.renderer.close()

            self.renderer = None


# ============================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================

def side_by_side(
    frames_a,
    frames_b,
):

    """
    Combine two rollouts horizontally.

    Useful for:

        RL controller
              VS
        fixed-impedance baseline

    Both simulations should ideally use the same random seed.
    """

    n = min(
        len(frames_a),
        len(frames_b),
    )

    combined = []

    for i in range(n):

        frame = np.hstack(
            [
                frames_a[i],
                frames_b[i],
            ]
        )

        combined.append(frame)

    return combined