# OSC address mapping
# Modify these to match whatever address schema you set up in Unreal Engine.
#
# Format: field → OSC address
OSC_ADDRESSES = {
    # Discrete states (sent on change)
    "proximity":       "/person/proximity",        # string: close/medium/far/none
    "expression":      "/person/expression",       # string: smile/angry/sad/neutral/none
    "gesture":         "/person/gesture",          # string: peekaboo/none

    # Continuous floats (sent every frame)
    "proximity_value": "/person/proximity/value",  # float 0.0–1.0
    "face_x":          "/person/face/x",           # float 0.0–1.0, -1.0 = no face
    "face_y":          "/person/face/y",           # float 0.0–1.0, -1.0 = no face
    "hand_right_x":    "/person/hand/right/x",     # float 0.0–1.0, -1.0 = not detected
    "hand_right_y":    "/person/hand/right/y",
    "hand_left_x":     "/person/hand/left/x",
    "hand_left_y":     "/person/hand/left/y",
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