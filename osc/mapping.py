# OSC address mapping
# Modify these to match whatever address schema you set up in Unreal Engine.

# Format: field → OSC address
OSC_ADDRESSES = {
    "proximity":       "/person/proximity",       # string: close/medium/far/none
    "proximity_value": "/person/proximity/value", # float: 0.0–1.0 continuous
    "expression":      "/person/expression",
    "gesture":         "/person/gesture",
}

# Optional: if Unreal expects integer codes instead of string values,
# define a value map here. Set USE_STRING_VALUES = False and define
# VALUE_CODES to switch to int mode.
USE_STRING_VALUES = True

VALUE_CODES = {
    # proximity
    "close":    1,
    "medium":   2,
    "far":      3,
    "none":     0,
    # expression
    "smile":    11,
    "angry":    12,
    "sad":      13,
    "neutral":  14,
    # gesture
    "peekaboo": 21,
}
