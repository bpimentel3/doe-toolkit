# Dual-Mode Augmentation Wizard Refactor

## Overview

The augmentation wizard has been refactored from a single diagnostics-driven mode into a dual-mode system that supports both automatic problem-fixing and user-driven enhancement.

---

## Architecture

### Module Structure

```
src/core/augmentation/
├── goals.py              # Mode B: Goal definitions and mappings
├── recommendations.py    # Mode A: Diagnostics-driven recommendations
├── goal_driven.py        # Mode B: Goal-driven recommendation engine
├── interface.py          # Unified interface for both modes
├── plan.py               # Data structures (unchanged)
├── foldover.py          # Strategy execution (unchanged)
├── optimal.py           # Strategy execution (unchanged)
└── validation.py         # Validation logic (unchanged)
```

### Clean Separation of Concerns

**1. Diagnostics Logic** (`diagnostics/summary.py`)
- Detects statistical issues
- Computes quality metrics
- Identifies problematic effects
- **Does not** recommend augmentation (decoupled)

**2. Augmentation Strategy Selection**
- **Mode A** (`recommendations.py`): Issue → Strategy mapping
- **Mode B** (`goals.py` + `goal_driven.py`): Goal → Strategy mapping

**3. Augmentation Generation** (`foldover.py`, `optimal.py`)
- Pure execution logic
- No decision-making
- Works with any plan

**4. User Interface** (`ui/components/augmentation_wizard.py`)
- Mode selection
- Goal selection (Mode B)
- Plan display
- Parameter adjustment

---

## Mode A: Fix Issues (Diagnostics-Driven)

### Workflow

1. **Analyze diagnostics** → Detect issues
2. **Prioritize issues** → Critical first, then warnings
3. **Map issues to strategies** → Each issue gets targeted fix
4. **Generate plans** → Executable augmentation plans
5. **Rank by utility** → Priority × severity × cost

### Issue → Strategy Mapping

| Issue Category | Strategy | Rationale |
|---------------|----------|-----------|
| Aliasing | Foldover (full or partial) | Breaks alias chains |
| Lack of fit | Add quadratic terms or center points | Captures curvature |
| Precision | I-optimal augmentation | Reduces prediction variance |
| Estimability | D-optimal orthogonalization | Reduces collinearity |
| Pure error | Replicates | Enables LOF testing |

---

## Mode B: Enhance Design (Goal-Driven)

### Workflow

1. **User selects goal** → Engineering intention (not statistical jargon)
2. **Map goal to strategies** → Appropriate augmentation approaches
3. **Use diagnostics to inform** → Size, placement, warnings
4. **Generate plans** → Primary + alternatives
5. **Allow adjustment** → Run count, parameters
6. **Execute** → Diagnostics don't block (warnings only)

### Engineering Goals

**1. Increase Confidence in Current Conclusions**
- Add replicates and center points
- Enable lack-of-fit testing
- **When:** Initial results promising, want to be sure

**2. Improve Prediction Across Design Space**
- I-optimal or space-filling designs
- Reduce prediction variance
- **When:** Need predictions at many untested conditions

**3. Model Curvature or Search for an Optimum**
- CCD augmentation (axial points)
- Enable response surface modeling
- **When:** Screening complete, ready to optimize

**4. Reduce Aliasing or Clarify Effects**
- Full or partial foldover
- Increase resolution
- **When:** Multiple significant effects that could be confounded

**5. Expand or Shift Region of Interest**
- D-optimal in expanded region
- Explore new factor space
- **When:** Optimum appears outside current boundaries

**6. Add Robustness to Noise Factors**
- Outer array (Taguchi)
- Test control factors under variation
- **When:** Need robust settings for production

### Diagnostics Role in Mode B

**Diagnostics INFORM but do NOT BLOCK:**

✅ **Informative:**
- Adjust run count based on current variance
- Suggest specific factors for partial foldover
- Show warnings if goal conflicts with detected issues

❌ **Do NOT Block:**
- User wants curvature terms but no LOF detected → Allow with note
- User wants more precision but variance already low → Allow with warning

---

## Summary of Changes

### Files Added

1. **`src/core/augmentation/goals.py`**
   - `AugmentationGoal` enum (6 engineering goals)
   - `GoalDescription` dataclass
   - `GOAL_CATALOG` with user-friendly descriptions
   - `get_strategies_for_goal()` — maps goals to strategies
   - `get_available_goals()` — filters by design type

2. **`src/core/augmentation/recommendations.py`**
   - `recommend_from_diagnostics()` — Mode A entry point
   - Issue prioritization and mapping logic
   - Utility scoring for diagnostic-driven plans

3. **`src/core/augmentation/goal_driven.py`**
   - `GoalDrivenContext` dataclass
   - `recommend_from_goal()` — Mode B entry point
   - Goal-to-strategy translation
   - Diagnostic refinement (warnings, not blocking)

4. **`src/core/augmentation/interface.py`**
   - `AugmentationRequest` dataclass
   - `recommend_augmentation()` — unified entry point
   - Helper functions for mode availability and goal listing

5. **`docs/architecture/dual_mode_augmentation_refactor.md`**
   - Complete architecture documentation

### Files Modified

1. **`src/core/augmentation/__init__.py`**
   - Exports new interfaces and functions

2. **`src/ui/components/augmentation_wizard.py`**
   - Completely rewritten for dual-mode workflow
   - `display_mode_selection()` — choose Mode A or B
   - `display_goal_selection()` — select enhancement goal
   - `display_augmentation_plans()` — now accepts mode parameter
   - Plan display shows diagnostic warnings in Mode B

3. **`src/ui/pages/7_augmentation.py`**
   - Updated to use new workflow
   - Mode selection → Goal selection (if Mode B) → Plan generation → Execution
   - Proper state management for multi-step workflow

---

## Commit Message

```
feat: implement dual-mode augmentation wizard

Architecture Changes:
- Decouple diagnostics from augmentation recommendations
- Implement Mode A (diagnostics-driven) recommendation engine
- Implement Mode B (goal-driven) recommendation engine
- Create unified interface supporting both modes

New Features:
- Mode A: Automatic issue detection and targeted fixes
- Mode B: 6 engineering goals with strategy mappings
- Goal catalog with user-friendly descriptions
- Diagnostic warnings in Mode B (inform, don't block)
- Plan comparison table across modes

UI Changes:
- Dual-mode selection screen
- Goal selection interface with examples
- Plan display shows mode-specific metadata
- Parameter adjustment hooks (implementation TBD)

Files Added:
- src/core/augmentation/goals.py
- src/core/augmentation/recommendations.py
- src/core/augmentation/goal_driven.py
- src/core/augmentation/interface.py
- docs/architecture/dual_mode_augmentation_refactor.md

Files Modified:
- src/core/augmentation/__init__.py
- src/ui/components/augmentation_wizard.py
- src/ui/pages/7_augmentation.py

Breaking Changes:
- Old recommend_augmentation_plans() no longer exists
- Use recommend_augmentation() with AugmentationRequest instead

Future Work:
- I-optimal criterion implementation
- Space-filling designs
- Region expansion logic
- Parameter adjustment UI
- Reduced factor space handling
