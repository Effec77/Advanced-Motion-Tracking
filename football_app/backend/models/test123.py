from pathlib import Path
import sys

# Ensure project root is on sys.path so `football_app` can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from football_app.backend.models.pose_estimation_2d import PoseEstimator2D
from football_app.backend.models.pose_2d_temporal_analyzer import Pose2DTemporalAnalyzer


def main() -> None:
    # TODO: replace with your actual video path
    video_path = "path_to_your_video.mp4"

    # ---- Stage 2 (2D pose estimation) ----
    #
    # NOTE: PoseEstimator2D in this project does NOT expose a `process_video`
    # method. Instead, it expects a PlayerTrackSet and CameraView produced by
    # Stage 1 (DetectionTrackingOrchestrator). The line below will therefore
    # raise AttributeError if called as-is.
    #
    pose_estimator = PoseEstimator2D()

    # This will fail with AttributeError in the current implementation:
    # pose_track_set = pose_estimator.process_video(video_path)
    #
    # Proper usage (high level) is:
    #   1) Build an ActionInstance from a folder with your video(s)
    #   2) Run DetectionTrackingOrchestrator to get a PlayerTrackSet
    #   3) Call pose_estimator.process_player_track_set(player_track_set, camera_view)
    #
    # For now, we just demonstrate that imports work and the analyzer can be
    # constructed; wiring to Stage 1 is a separate step.

    # Placeholder so the script runs without crashing:
    print("PoseEstimator2D imported successfully.")

    # ---- Analyzer construction (Stage-2 diagnostics) ----
    analyzer = Pose2DTemporalAnalyzer()
    print("Pose2DTemporalAnalyzer imported successfully.")


if __name__ == "__main__":
    main()

