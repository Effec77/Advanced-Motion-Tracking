# Football Hackathon Mode – System Redesign & Implementation Documentation

## Purpose of This Document

This document defines a **temporary but rigorous redesign** of the existing human-motion analysis system to align with a **football-focused GenAI hackathon (e.g., FC Barcelona-style analytics challenges)**.

The goal is to:
- Prioritize **football-specific insights** (ball control, stability, spatial dominance)
- Preserve the **core motion-intelligence architecture**
- Avoid technical debt or hacks that break the long-term multi-sport vision

This document should be read as a **Hackathon Mode Adapter** layered on top of the existing core system.

---

## 1. Design Philosophy for Hackathon Mode

### 1.1 Key Constraints

- Multi-player, multi-agent environment (football is not single-subject)
- Short clips (2–5 seconds)
- Broadcast-style or training-style camera views
- Judges value **clarity, explainability, and football relevance**

### 1.2 Non-Negotiables

- No random or heuristic-only scoring
- No single “best player” assumption
- No sport-agnostic abstractions removed
- No confidentiality violations (no raw video persistence)

---

## 2. Canonical Unit: ActionInstance (Unchanged, Reused)

### 2.1 Definition

An **ActionInstance** represents one real football event observed by one or more cameras.

Examples:
- A ball shot
- A pass reception
- A dribble sequence

Each ActionInstance may contain **multiple camera views**.

### 2.2 Folder-to-Instance Mapping

```
/event_001/
   cam_1.mp4
   cam_2.mp4
   cam_3.mp4
   cam_4.mp4
```

This folder maps to **ONE ActionInstance**, not multiple.

---

## 3. Revised High-Level Pipeline (Hackathon Mode)

```
ActionInstance
  ├─ View 1 (video)
  │    └─ Detection → Tracking → 2D Pose → 3D Pose → Metrics
  ├─ View 2 (video)
  │    └─ Detection → Tracking → 2D Pose → 3D Pose → Metrics
  └─ View N (video)
       └─ Detection → Tracking → 2D Pose → 3D Pose → Metrics
            ↓
      Field-Centric Projection (Top-Down Abstraction)
            ↓
      Ball Control Inference
            ↓
      Stability & Control Metrics
            ↓
      Football Insights (Hackathon Output)
```

---

## 4. Major Change 1: Field-Centric (Top-Down) Representation

### 4.1 Rationale

Football analysis is fundamentally spatial:
- Team shape
- Player spacing
- Ball-relative positioning

Camera-centric views obscure this.

### 4.2 Implementation Strategy

- Do NOT rotate video or skeletons
- Project player foot positions onto a **2D pitch plane**
- Normalize coordinates to a canonical field layout

### 4.3 Output

Each player at each timestep is represented as:

```
(X, Y)_field
```

This enables top-down analysis without requiring true overhead cameras.

---

## 5. Major Change 2: Remove CNN-Based Best Player Selection

### 5.1 Why the Old Approach Fails

- Football is multi-agent
- “Main player” changes continuously
- CNN-based visual selection ignores ball dynamics

### 5.2 New Approach: All-Player Processing

- Detect and track **all players simultaneously**
- Treat every player as a potential actor

The CNN-based selector is **disabled** in Hackathon Mode.

---

## 6. Major Change 3: Ball as a First-Class Entity

### 6.1 Ball Detection & Tracking

- Add ball as an explicit detection class
- Track ball trajectory across frames
- Apply temporal smoothing for robustness

### 6.2 Ball Control Inference

Ball control is inferred probabilistically using:
- Player–ball distance
- Velocity alignment (player motion vs ball motion)
- Temporal continuity of proximity

At each frame:

```
P(player_i controls ball | motion, proximity, history)
```

No hard labels are stored.

---

## 7. Major Change 4: Removal of Random / Heuristic Scoring

### 7.1 Deprecated Logic

All logic resembling:

```python
np.random.uniform(...)
```

is permanently removed.

### 7.2 Replacement Philosophy

Hackathon outputs must be:
- Measurable
- Explainable
- Grounded in motion physics

No single scalar “technique score” is produced.

---

## 8. Major Change 5: Enhanced Stability Metrics

### 8.1 Definition of Stability in Football

Stability is defined as **control of posture under dynamic conditions**, especially while interacting with the ball.

### 8.2 Stability Metrics Implemented

From 3D pose and field projection:

- Center of Mass (COM) position
- Base of support (foot polygon)
- COM-to-support distance
- Lateral sway during possession
- Recovery time after direction change

### 8.3 Stability Index

A composite stability index is computed using normalized physical measures, not ML predictions.

---

## 9. Output Schema (Hackathon Mode)

Each ActionInstance produces:

```
{
  "ball_controller": player_id,
  "controller_confidence": 0.87,
  "stability_metrics": {
    "com_support_distance": 0.12,
    "lateral_sway": 0.08,
    "recovery_time": 0.35
  },
  "field_positions": [...],
  "confidence": 0.91
}
```

No videos, images, or camera metadata are retained.

---

## 10. Relationship to Core Motion-Intelligence System

Hackathon Mode:
- Adds football-specific interpretation layers
- Does NOT modify pose extraction or motion representation

After the hackathon:
- Hackathon adapter can be removed
- Core system remains intact for multi-sport expansion

---

## 11. Summary of Required Code Changes

### Add
- Field-centric projection module
- Ball detection & tracking
- Ball control inference logic
- Stability metric computation

### Disable / Remove
- CNN-based best-player selection
- Random score generation
- Single-player assumptions

---

## 12. End State for Hackathon Submission

The final system:
- Analyzes football events holistically
- Handles all players simultaneously
- Explains decisions via physics and geometry
- Produces visually intuitive, top-down insights
- Remains legally safe and extensible

This design is suitable for both hackathon evaluation and future professional-grade development.

