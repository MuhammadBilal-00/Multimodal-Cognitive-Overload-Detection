// Channel order for the model's `states` output tensor.
//
// This order is NOT alphabetical and NOT a free choice — it is fixed by the
// training pipeline: ml/src/labels.py `LABEL_COLS = ["Boredom", "Engagement",
// "Confusion", "Frustration"]` -> binarize_states() builds its columns by
// iterating LABEL_COLS -> ml/src/dataset.py `states.loc[clip_id].to_numpy()`
// preserves that column order into y_states, which is what the head is
// trained against.
//
// Index N here MUST equal index N of `prediction.states`. Never reorder this
// array to change display order — sort at render time instead. Getting this
// wrong is silent: the bars still render, they just describe the wrong state.
// (A previous version listed these alphabetically, which swapped indices 1
// and 2, so the "Confused" bar was really showing P(engagement) ≈ 0.99.)
//
// Each entry is an INDEPENDENT binary probability (sigmoid, "is this state at
// DAiSEE level >= 2?"), not a share of a distribution — see labels.py
// STATE_THRESHOLD. They do not sum to 1.
export const STATE_CHANNELS = [
  { key: 'boredom', label: 'Bored', color: '#64748b' },
  { key: 'engagement', label: 'Engaged', color: '#16a34a' },
  { key: 'confusion', label: 'Confused', color: '#d97706' },
  { key: 'frustration', label: 'Frustrated', color: '#dc2626' },
] as const;

// Mirrors ml/src/labels.py LABEL_COLS exactly, for the regression test that
// guards the ordering above.
export const PYTHON_LABEL_COLS = [
  'Boredom', 'Engagement', 'Confusion', 'Frustration',
] as const;
