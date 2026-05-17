# 🔍 Stage 2 → Stage 3 Integration Audit Report

**Date:** February 17, 2026  
**Objective:** Audit 2D pose estimation (Stage 2) and 3D lifting (Stage 3) integration for temporal consistency optimization  
**Mode:** Single-Camera Production (Multi-view components archived)

---

## 📋 Executive Summary

**Status:** ✅ **AUDIT COMPLETE**

The pipeline stores 2D poses per frame with full temporal access, but **no temporal smoothing or consistency enforcement** exists between Stage 2 and Stage 3. Raw per-frame estimates flow directly into MotionFormer with only spatial normalization (root joint centering).

**Key Finding:** Temporal jitter reduction can be implemented **between Stage 2 output and Stage 3 input** without modifying either stage's architecture.

---

## 🏗️ Architecture Overview

### Data Flow Path

```
Stage 2: PoseEstimator2D
    ↓
PoseTrackSet (collection of PoseTrack objects)
    ↓
PoseTrack.poses: List[Pose2D]  ← Temporal sequence stored here
    ↓
Stage 3: PoseLifter3D.process_pose_track_set()
    ↓
pose_track.to_array() → (T, 17, 3) numpy array
    ↓
Non-overlapping windows: (243, 17, 3) per window
    ↓
_normalize_2d_poses() → Root joint centering (spatial only)
    ↓
MotionAGFormer input: (1, 243, 17, 3) tensor
```

---

## 📊 Task 1: Codebase Audit Results

### ✅ 1.1 Are 2D keypoints stored per frame?

**Answer:** YES

**Location:** `football_app/backend/models/pose_estimation_2d.py`

**Data Structure:**
```python
@dataclass
class PoseTrack:
    track_id: int
    poses: List[Pose2D] = field(default_factory=list)  # ← Temporal sequence here
    player_track: Optional[PlayerTrack] = None
```

**Details:**
- Each `Pose2D` contains 17 `Keypoint2D` objects
- Each `Keypoint2D` has: `x: float`, `y: float`, `confidence: float`
- Poses are stored in frame order (sequential list)
- Frame IDs preserved: `Pose2D.frame_id` maps to original video frame

**Access Pattern:**
```python
# Sequential access during Stage 2 processing
pose_tracks_dict[track_id].add_pose(pose)  # Line 428, 544

# Random access available
pose_track.get_pose_at_frame(frame_id)  # Line 118-123
```

---

### ✅ 1.2 Are they buffered temporally before Stage 3?

**Answer:** YES - Full track buffered

**Location:** `football_app/backend/models/pose_lifting_3d.py:350`

**Process:**
1. Stage 2 completes entire `PoseTrackSet` before Stage 3 begins
2. Stage 3 receives complete `PoseTrackSet` object
3. Each `PoseTrack` contains full temporal sequence in `poses` list
4. Conversion to numpy array happens on-demand: `poses_2d_array = pose_track.to_array()`

**Evidence:**
```python
# Stage 3 processes complete tracks
def process_pose_track_set(self, pose_track_set: PoseTrackSet) -> Pose3DTrackSet:
    # ...
    for pose_track in pose_track_set.pose_tracks:  # Full track available
        pose_3d_track = self._process_single_pose_track(pose_track, model)

# Inside _process_single_pose_track:
poses_2d_array = pose_track.to_array()  # Converts entire list to array
# Shape: (T, 17, 3) where T = number of frames in track
```

**Temporal Window Processing:**
- Tracks processed in **non-overlapping windows** of 243 frames
- Windows extracted: `window_2d = poses_2d_array[start_idx:end_idx]`
- No overlap means discontinuities at window boundaries

---

### ✅ 1.3 Is there a pose history array or sliding window?

**Answer:** NO - List-based storage, array conversion on-demand

**Current Implementation:**
- **Storage:** `List[Pose2D]` in `PoseTrack.poses`
- **Conversion:** `to_array()` method creates numpy array when needed
- **No sliding window:** Fixed non-overlapping 243-frame windows in Stage 3
- **No history buffer:** Each frame processed independently in Stage 2

**Array Conversion Method:**
```python
def to_array(self) -> np.ndarray:
    """
    Return all poses as numpy array of shape (T, 17, 3).
    T = number of frames
    17 = number of keypoints
    3 = [x, y, confidence]
    """
    if not self.poses:
        return np.empty((0, 17, 3))
    return np.array([pose.to_array() for pose in self.poses])
```

**Implications:**
- Full temporal access available (entire track in memory)
- No pre-computed sliding windows
- Window extraction happens in Stage 3 during processing

---

### ✅ 1.4 Are confidence scores preserved?

**Answer:** YES - Fully preserved

**Location:** `football_app/backend/models/pose_estimation_2d.py:33-51`

**Data Structure:**
```python
@dataclass
class Keypoint2D:
    x: float
    y: float
    confidence: float  # ← Preserved from RTMPose output
```

**Flow:**
1. RTMPose outputs: `keypoints, scores = pose_estimator(crop)` (Line 399, 515)
2. Confidence stored: `confidence=float(scrs[i])` (Line 417, 533)
3. Array format: `(T, 17, 3)` where `3 = [x, y, confidence]`
4. Stage 3 uses: `confidences = window_2d[:, :, 2]` (Line 368)

**Confidence Usage in Stage 3:**
- Masking: `valid_mask = confidences >= self.confidence_threshold` (Line 369)
- Low-confidence joints set to zero before normalization (Lines 379-383)
- Window validation: Requires ≥50% valid joints (Line 373)

---

### ✅ 1.5 How does 2D data flow into MotionFormer?

**Answer:** Direct array extraction → windowing → normalization → tensor conversion

**Location:** `football_app/backend/models/pose_lifting_3d.py:348-394`

**Step-by-Step Flow:**

#### Step 1: Array Extraction
```python
poses_2d_array = pose_track.to_array()  # (T, 17, 3)
```
- Converts `List[Pose2D]` to numpy array
- Shape: `(T, 17, 3)` where T = track length

#### Step 2: Window Extraction
```python
num_windows = num_poses // self.temporal_window  # 243 frames per window
for window_idx in range(num_windows):
    start_idx = window_idx * self.temporal_window
    end_idx = start_idx + self.temporal_window
    window_2d = poses_2d_array[start_idx:end_idx]  # (243, 17, 3)
```
- **Non-overlapping** windows
- Fixed size: 243 frames (matches checkpoint training)

#### Step 3: Confidence Masking
```python
confidences = window_2d[:, :, 2]  # Extract confidence channel
valid_mask = confidences >= self.confidence_threshold  # Default: 0.2
# Set low-confidence joints to zero
for t in range(self.temporal_window):
    for j in range(17):
        if not valid_mask[t, j]:
            input_2d[t, j] = 0.0
```

#### Step 4: Spatial Normalization
```python
input_2d_normalized = self._normalize_2d_poses(input_2d)
```
**Normalization Method** (`_normalize_2d_poses`, Lines 433-456):
- Centers each frame around root joint (hip midpoint)
- Root = `(left_hip + right_hip) / 2.0`
- **Per-frame normalization** (no temporal smoothing)
- Formula: `poses_2d_normalized[t] = poses_2d[t] - root[t]`

#### Step 5: Tensor Conversion
```python
input_tensor = torch.from_numpy(input_2d_normalized).float().unsqueeze(0)
# Shape: (1, 243, 17, 3)
input_tensor = input_tensor.to(self.device)
```

#### Step 6: Model Inference
```python
with torch.no_grad():
    output_3d = model(input_tensor)  # (1, 243, 17, 3)
```

**MotionFormer Input Format:**
- Shape: `(B, T, J, C)` = `(1, 243, 17, 3)`
- B = batch size (1)
- T = temporal window (243 frames)
- J = joints (17 COCO keypoints)
- C = channels (3: x, y, confidence)

---

## 🔍 Critical Observations

### ❌ Missing Temporal Consistency

**Current State:**
- No temporal smoothing between Stage 2 and Stage 3
- Raw per-frame estimates flow directly to MotionFormer
- Normalization is **spatial only** (root joint centering per frame)
- No temporal filtering or jitter reduction

**Impact:**
- Temporal jitter in 2D poses propagates to 3D output
- MotionFormer must handle noise internally
- Window boundaries may have discontinuities (non-overlapping windows)

### ✅ Available Temporal Access

**Capabilities:**
- Full track available before Stage 3 processing
- Array conversion provides `(T, 17, 3)` access
- Frame IDs preserved for alignment
- Confidence scores available for quality assessment

**Opportunity:**
- Can implement temporal smoothing **before** window extraction
- Can compute temporal metrics (jitter, velocity) for diagnostics
- Can apply filters without modifying Stage 2 or Stage 3 architecture

### ⚠️ Window Boundary Discontinuities

**Issue:**
- Non-overlapping 243-frame windows
- Last frames of each window may not align smoothly with next window
- MotionFormer processes each window independently

**Evidence:**
```python
# Non-overlapping extraction
start_idx = window_idx * self.temporal_window
end_idx = start_idx + self.temporal_window
window_2d = poses_2d_array[start_idx:end_idx]
```

**Potential Impact:**
- Discontinuities in 3D output at window boundaries
- Temporal smoothing could help bridge windows

---

## 📁 File Locations Summary

### Stage 2 Implementation
- **Main File:** `football_app/backend/models/pose_estimation_2d.py`
- **Data Structures:**
  - `Keypoint2D` (Lines 33-51)
  - `Pose2D` (Lines 54-95)
  - `PoseTrack` (Lines 98-156)
  - `PoseTrackSet` (Lines 159-202)
  - `PoseEstimator2D` (Lines 205-552)

### Stage 3 Implementation
- **Main File:** `football_app/backend/models/pose_lifting_3d.py`
- **Data Structures:**
  - `Joint3D` (Lines 33-51)
  - `Pose3D` (Lines 54-85)
  - `Pose3DTrack` (Lines 88-142)
  - `Pose3DTrackSet` (Lines 145-193)
  - `PoseLifter3D` (Lines 196-456)

### Integration Point
- **Entry:** `PoseLifter3D.process_pose_track_set()` (Line 290)
- **Processing:** `PoseLifter3D._process_single_pose_track()` (Line 322)
- **Array Conversion:** `pose_track.to_array()` (Line 350)
- **Window Extraction:** Lines 360-365
- **Normalization:** `_normalize_2d_poses()` (Line 386, implementation Lines 433-456)

---

## 🎯 Task 2: Minimal Visibility Layer Assessment

### Current Visibility

**Available:**
- ✅ Full temporal sequence in `PoseTrack.poses`
- ✅ Array conversion via `to_array()` → `(T, 17, 3)`
- ✅ Confidence scores per joint per frame
- ✅ Frame IDs for temporal alignment

**Missing:**
- ❌ Temporal jitter metrics
- ❌ Per-joint velocity calculations
- ❌ Confidence trend analysis
- ❌ Temporal consistency scores

### Proposed Minimal Visibility Layer

**Location:** New diagnostic module (non-invasive)

**Purpose:** Enable temporal analysis without modifying Stage 2 or Stage 3

**Design:**
```python
# Diagnostic-only module
class Pose2DTemporalAnalyzer:
    """
    Computes temporal metrics on PoseTrack objects.
    Diagnostic-only, does not modify data.
    """
    
    def compute_jitter_metrics(self, pose_track: PoseTrack) -> Dict:
        """Compute per-joint temporal jitter"""
        # Extract array: (T, 17, 3)
        # Compute frame-to-frame differences
        # Return jitter statistics per joint
        
    def compute_velocity_spikes(self, pose_track: PoseTrack) -> Dict:
        """Detect velocity spikes (explosive motion vs noise)"""
        # Compute velocities
        # Identify spikes above threshold
        # Return spike locations and magnitudes
        
    def analyze_confidence_trends(self, pose_track: PoseTrack) -> Dict:
        """Analyze confidence score trends over time"""
        # Extract confidence channel
        # Compute trends per joint
        # Return trend statistics
```

**Constraints Met:**
- ✅ No architecture changes
- ✅ No new dependencies (uses numpy only)
- ✅ No schema modification
- ✅ No runtime impact (diagnostic-only, called separately)
- ✅ No modification of Stage 2 or Stage 3

**Implementation Location:**
- New file: `football_app/backend/models/pose_2d_temporal_analyzer.py`
- Or: Add to existing test/validation utilities

---

## 🗺️ Task 3: Stage 2 Optimization Roadmap

### 3.1 Stabilization Layer Assessment

**Question:** Is stabilization layer needed?

**Answer:** **YES** - Based on audit findings:

1. **No temporal smoothing exists** - Raw estimates go directly to MotionFormer
2. **Jitter propagation risk** - 2D jitter → 3D instability
3. **Window boundary issues** - Non-overlapping windows may have discontinuities
4. **MotionFormer handles noise** - But could benefit from cleaner input

**Recommendation:** Implement lightweight temporal stabilization **between Stage 2 and Stage 3**

---

### 3.2 Stabilization Placement

**Question:** Where should stabilization sit?

**Answer:** **Between Stage 2 output and Stage 3 input**

**Optimal Location:**
```
Stage 2: PoseTrackSet (raw 2D poses)
    ↓
[STABILIZATION LAYER] ← Insert here
    ↓
Stabilized PoseTrackSet
    ↓
Stage 3: PoseLifter3D (processes stabilized poses)
```

**Implementation Approach:**
1. **Option A:** New method in `PoseLifter3D` (before window extraction)
   - Location: `_process_single_pose_track()` method
   - Apply stabilization to `poses_2d_array` before windowing
   - Pros: Minimal code changes
   - Cons: Couples stabilization to Stage 3

2. **Option B:** Separate stabilization module (recommended)
   - New class: `Pose2DStabilizer`
   - Process `PoseTrackSet` → stabilized `PoseTrackSet`
   - Pros: Clean separation, testable independently
   - Cons: Additional module

3. **Option C:** Method in `PoseTrack` class
   - Add `stabilize()` method to `PoseTrack`
   - Returns new stabilized `PoseTrack`
   - Pros: Object-oriented, reusable
   - Cons: Modifies core data structure

**Recommendation:** **Option B** - Separate stabilization module

**Rationale:**
- Maintains architectural separation
- Enables A/B testing (stabilized vs raw)
- No modification to Stage 2 or Stage 3 core logic
- Can be disabled/enabled via configuration

---

### 3.3 Required Metrics Before Fine-Tuning

**Question:** What metrics must be computed before fine-tuning?

**Answer:** Baseline temporal quality metrics

#### 3.3.1 Per-Joint Temporal Jitter

**Metric:** Frame-to-frame position variance

**Formula:**
```python
# For each joint j, compute:
jitter[j] = std(diff(positions[j, :]))  # Standard deviation of frame differences
```

**Purpose:**
- Identify joints with high temporal noise
- Baseline for improvement measurement
- Target joints: ankles (15, 16), knees (13, 14)

#### 3.3.2 Velocity Spike Detection

**Metric:** Frame-to-frame velocity magnitude

**Formula:**
```python
# For each joint j:
velocity[t, j] = ||position[t, j] - position[t-1, j]||
spikes = velocity > threshold  # Identify explosive motion vs noise
```

**Purpose:**
- Distinguish real explosive motion from noise
- Preserve legitimate velocity spikes
- Avoid over-smoothing dynamic movements

#### 3.3.3 Confidence Trend Analysis

**Metric:** Temporal confidence score patterns

**Formula:**
```python
# For each joint j:
confidence_trend[j] = trend(confidence[t, j])  # Linear/quadratic fit
confidence_variance[j] = var(confidence[t, j])
```

**Purpose:**
- Identify joints with unstable confidence
- Weight stabilization by confidence
- Low-confidence joints need more smoothing

#### 3.3.4 Window Boundary Discontinuity

**Metric:** Position difference at window boundaries

**Formula:**
```python
# For non-overlapping windows:
discontinuity = ||pose[window_end] - pose[next_window_start]||
```

**Purpose:**
- Measure impact of non-overlapping windows
- Validate stabilization effectiveness
- Ensure smooth transitions

**Implementation Priority:**
1. **High:** Per-joint jitter (ankles, knees)
2. **High:** Velocity spike detection
3. **Medium:** Confidence trends
4. **Low:** Window boundary discontinuity (post-stabilization validation)

---

### 3.4 Downstream Validation in Stage 3

**Question:** What must be validated downstream in Stage 3?

**Answer:** 3D stability metrics and MotionFormer compatibility

#### 3.4.1 3D Output Stability

**Metrics:**
- **3D jitter:** Frame-to-frame 3D joint position variance
- **3D velocity consistency:** Smoothness of 3D motion trajectories
- **Root joint stability:** Hip midpoint stability (should be near zero)

**Validation:**
```python
# After Stage 3 processing:
pose_3d_track = pose_lifter.process_pose_track_set(stabilized_pose_track_set)

# Compute 3D metrics:
for pose_3d_track in pose_3d_track_set.pose_3d_tracks:
    poses_3d_array = pose_3d_track.to_array()  # (T, 17, 3)
    
    # 3D jitter per joint
    for j in range(17):
        jitter_3d[j] = std(diff(poses_3d_array[:, j, :]))
    
    # Root joint should be near (0, 0, 0) - verify stability
    root_joints = poses_3d_array[:, [11, 12], :]  # Hips
    root_stability = std(root_joints.mean(axis=1))
```

#### 3.4.2 MotionFormer Input Compatibility

**Validation:**
- Input shape: `(1, 243, 17, 3)` maintained
- Normalization: Root joint centering still valid
- Confidence masking: Low-confidence joints still masked correctly
- Window extraction: Non-overlapping windows still work

**Test:**
```python
# Verify stabilized poses can be processed by Stage 3
stabilized_array = stabilized_pose_track.to_array()  # (T, 17, 3)

# Extract window
window = stabilized_array[start_idx:end_idx]  # (243, 17, 3)

# Verify normalization still works
normalized = _normalize_2d_poses(window)

# Verify tensor conversion
tensor = torch.from_numpy(normalized).float().unsqueeze(0)
assert tensor.shape == (1, 243, 17, 3)
```

#### 3.4.3 Explosive Motion Preservation

**Validation:**
- Legitimate velocity spikes preserved
- Dynamic movements not over-smoothed
- Temporal characteristics maintained

**Test Cases:**
- Fast kicks: Should preserve rapid motion
- Quick direction changes: Should maintain sharp transitions
- Static periods: Should be smooth (low jitter)

#### 3.4.4 Stage 4 Metric Compatibility

**Validation:**
- Stage 4 (Motion Metrics) still produces valid outputs
- Deterministic metrics unchanged
- Stability index improvements measurable

**Test:**
```python
# Process through Stage 4
metric_track_set = metrics_engine.process_pose_3d_track_set(pose_3d_track_set)

# Verify metrics are valid
for metric_track in metric_track_set.metric_tracks:
    assert metric_track.status == MetricStatus.COMPLETE
    assert metric_track.summary_metrics.Stability > 0
    assert metric_track.summary_metrics.Stability <= 1
```

---

## 🚫 Constraints Compliance

### ✅ Architecture Lock Compliance

- **Stage 1-4 architecture locked:** ✅ No changes to core stages
- **Stage 3 (MotionFormer) not redesigned:** ✅ Stabilization external to Stage 3
- **Stage 4 metrics deterministic:** ✅ No changes to Stage 4
- **No schema changes:** ✅ Stabilization preserves data structures

### ✅ Single-Camera Mode Compliance

- **Multi-view removed:** ✅ No multi-view logic referenced
- **No triangulation:** ✅ No geometric fusion
- **Monocular only:** ✅ Stabilization operates on single-camera data

### ✅ Production Constraints

- **No deployment optimization:** ✅ Focus on quality, not performance
- **No UI changes:** ✅ Backend-only work
- **No ML scoring model:** ✅ Deterministic stabilization only

---

## 📋 Recommended Next Steps

### Phase 1: Diagnostic Visibility (Task 2)

1. **Implement `Pose2DTemporalAnalyzer`**
   - Compute jitter metrics
   - Detect velocity spikes
   - Analyze confidence trends
   - **Output:** Diagnostic report per track

2. **Run Baseline Analysis**
   - Process sample tracks through analyzer
   - Identify high-jitter joints (ankles, knees)
   - Measure current temporal quality
   - **Output:** Baseline metrics report

### Phase 2: Stabilization Design (Task 3)

1. **Design Stabilization Algorithm**
   - Temporal filter selection (moving average, median, Kalman)
   - Joint-specific smoothing (more for ankles/knees)
   - Confidence-weighted smoothing
   - Explosive motion preservation logic

2. **Implement `Pose2DStabilizer` Module**
   - Input: `PoseTrackSet`
   - Output: Stabilized `PoseTrackSet`
   - Configurable parameters (window size, strength)
   - **No modification to Stage 2 or Stage 3**

### Phase 3: Validation (Task 3.4)

1. **A/B Testing**
   - Process tracks: raw vs stabilized
   - Compare Stage 3 outputs
   - Measure 3D stability improvements
   - Validate MotionFormer compatibility

2. **Stage 4 Validation**
   - Verify Stage 4 metrics still valid
   - Measure stability index improvements
   - Ensure deterministic behavior maintained

---

## 📊 Summary Table

| Aspect | Status | Details |
|--------|--------|---------|
| **2D Storage** | ✅ Per-frame | `List[Pose2D]` in `PoseTrack.poses` |
| **Temporal Buffering** | ✅ Full track | Entire sequence available before Stage 3 |
| **Pose History** | ❌ List-based | Array conversion on-demand, no sliding window |
| **Confidence Scores** | ✅ Preserved | Stored in `Keypoint2D.confidence` |
| **Temporal Smoothing** | ❌ None | Raw estimates flow directly to MotionFormer |
| **Window Processing** | ⚠️ Non-overlapping | 243-frame windows, potential discontinuities |
| **Normalization** | ⚠️ Spatial only | Root joint centering per frame, no temporal |
| **Stabilization Needed** | ✅ YES | High jitter risk, no smoothing exists |
| **Placement** | 📍 Between Stage 2/3 | Separate module recommended |
| **Metrics Required** | 📊 Jitter, velocity, confidence | Baseline needed before optimization |

---

## 🎯 Conclusion

**Audit Status:** ✅ **COMPLETE**

The pipeline provides full temporal access to 2D poses but lacks temporal consistency enforcement. A stabilization layer between Stage 2 and Stage 3 is **feasible and recommended** without modifying core architecture.

**Key Insight:** Temporal smoothing can be implemented as a **separate, optional module** that processes `PoseTrackSet` objects before Stage 3, maintaining architectural integrity while improving 3D stability.

**Next Action:** Implement diagnostic visibility layer (Task 2) to establish baseline metrics before designing stabilization algorithm (Task 3).

---

**Report Generated:** February 17, 2026  
**Architecture Mode:** Single-Camera Production  
**Multi-View Components:** Archived (not referenced)
