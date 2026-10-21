"""CostModelParams schema shared by plugin.c and the Python cost model.

The C struct is initialised positionally, so this module is the single source
of truth for field order.  It can parse any `static const CostModelParams X =
{...};` initializer out of plugin.c and render the autotuned block back.
"""

import copy
import re

import numpy as np

from .constants import ALGO_NAMES, PROTO_NAMES

# (name, shape, type) in C struct declaration order.
FIELDS = [
    ("gpusPerNode", (), int),
    ("nChannels", (), int),
    ("bwIntra", (), float),
    ("bwInter", (), float),
    ("nvlsEfficiency", (), float),
    ("graphLatencyInter", (), float),
    ("baseLatencies", (7, 3), float),
    ("hwLatencies", (3, 7, 3), float),
    ("llMaxBw", (3,), float),
    ("perChMaxRingLL128Bw", (3,), float),
    ("perChMaxTreeBw", (3,), float),
    ("perChMaxTreeLL128Bw", (3,), float),
    ("perChMaxNVLSTreeBw", (3,), float),
    ("ringBwCapInter", (), float),
    ("ringBwCapIntra", (), float),
    ("patBwEff", (), float),
    ("patLatPerRank", (), float),
    ("regBwScale", (7, 3), float),
    ("disableUnmodeledCells", (), int),
]

PROFILE_VARS = {
    "mnnvl": "gb200fParamsDefault",
    "rail_nop2p": "railNoP2pParamsDefault",
}
PROFILE_VALIDATED_VARS = {
    "rail_nop2p": "railNoP2pParamsValidated",
}

BEGIN_MARKER = "// BEGIN AUTOTUNE PARAMS {profile}"
END_MARKER = "// END AUTOTUNE PARAMS {profile}"

_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_HW_NAMES = ["NVLINK", "PCI", "NET"]


def _size(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def num_values():
    return sum(_size(s) for _, s, _ in FIELDS)


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _initializer_body(text, var):
    m = re.search(r"static\s+const\s+CostModelParams\s+" + re.escape(var) + r"\s*=\s*\{", text)
    if not m:
        raise KeyError(f"CostModelParams initializer '{var}' not found")
    depth, i = 1, m.end()
    while depth and i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth:
        raise ValueError(f"unbalanced braces in '{var}'")
    return text[m.end():i - 1]


def params_from_values(values):
    """Build a params dict from a flat value list; missing trailing values are 0
    (same as C positional initialisation)."""
    total = num_values()
    if len(values) > total:
        raise ValueError(f"too many initializer values ({len(values)} > {total})")
    vals = list(values) + [0.0] * (total - len(values))
    out, k = {}, 0
    for name, shape, typ in FIELDS:
        n = _size(shape)
        chunk = vals[k:k + n]
        k += n
        if shape == ():
            out[name] = typ(chunk[0])
        else:
            out[name] = np.array(chunk, dtype=float).reshape(shape)
    return out


def params_to_values(params):
    vals = []
    for name, shape, _ in FIELDS:
        v = params[name]
        if shape == ():
            vals.append(v)
        else:
            vals.extend(np.asarray(v, dtype=float).reshape(-1).tolist())
    return vals


def parse_params(text, profile):
    """Parse the CostModelParams initializer of `profile` from plugin.c text."""
    body = _strip_comments(_initializer_body(text, PROFILE_VARS[profile]))
    return params_from_values([float(x) for x in _NUM_RE.findall(body)])


def parse_validated(text, profile):
    var = PROFILE_VALIDATED_VARS.get(profile)
    if var is None:
        return True
    m = re.search(r"static\s+const\s+int\s+" + re.escape(var) + r"\s*=\s*(\d+)\s*;", text)
    if not m:
        raise KeyError(f"'{var}' not found")
    return bool(int(m.group(1)))


def load_params(plugin_path, profile):
    with open(plugin_path) as f:
        return parse_params(f.read(), profile)


def copy_params(params):
    return copy.deepcopy(params)


# ---------------------------------------------------------------------------
# Addressing individual parameters with string keys, e.g. "hwLatencies[2][1][0]".
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^(\w+)((?:\[\d+\])*)$")


def split_key(key):
    m = _KEY_RE.match(key)
    if not m:
        raise ValueError(f"bad parameter key '{key}'")
    idx = tuple(int(x) for x in re.findall(r"\[(\d+)\]", m.group(2)))
    return m.group(1), idx


def get_param(params, key):
    name, idx = split_key(key)
    v = params[name]
    return float(v[idx]) if idx else float(v)


def set_param(params, key, value):
    name, idx = split_key(key)
    if idx:
        params[name][idx] = value
    else:
        params[name] = type(params[name])(value)


def key_label(key):
    """Human-readable label, e.g. hwLatencies[2][1][0] -> hwLatencies[NET][ring][ll]."""
    name, idx = split_key(key)
    if name in ("baseLatencies", "regBwScale") and len(idx) == 2:
        return f"{name}[{ALGO_NAMES[idx[0]]}][{PROTO_NAMES[idx[1]]}]"
    if name == "hwLatencies" and len(idx) == 3:
        return f"{name}[{_HW_NAMES[idx[0]]}][{ALGO_NAMES[idx[1]]}][{PROTO_NAMES[idx[2]]}]"
    return key


# ---------------------------------------------------------------------------
# Rendering the autotuned C block.
# ---------------------------------------------------------------------------
def _fmt(x):
    x = float(x)
    if x == 0.0:
        return "0.0"
    s = f"{x:.6g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def _row(vals, width=9):
    return "{ " + ", ".join(_fmt(v).rjust(width) for v in vals) + " }"


def render_initializer(params, profile):
    p = params
    lines = [f"static const CostModelParams {PROFILE_VARS[profile]} = {{"]
    lines.append(f"  /* gpusPerNode        */ {int(p['gpusPerNode'])},")
    lines.append(f"  /* nChannels          */ {int(p['nChannels'])},")
    for name in ("bwIntra", "bwInter", "nvlsEfficiency", "graphLatencyInter"):
        lines.append(f"  /* {name:<18} */ {_fmt(p[name])},")
    lines.append("  /* baseLatencies [algo][LL, LL128, SIMPLE] */")
    lines.append("  {")
    for a in range(7):
        lines.append(f"    /* {ALGO_NAMES[a]:<14} */ {_row(p['baseLatencies'][a])},")
    lines.append("  },")
    lines.append("  /* hwLatencies [hw][algo][LL, LL128, SIMPLE] */")
    lines.append("  {")
    for h in range(3):
        lines.append(f"    /* {_HW_NAMES[h]} */")
        lines.append("    {")
        for a in range(7):
            lines.append(f"      /* {ALGO_NAMES[a]:<14} */ {_row(p['hwLatencies'][h][a])},")
        lines.append("    },")
    lines.append("  },")
    for name in ("llMaxBw", "perChMaxRingLL128Bw", "perChMaxTreeBw",
                 "perChMaxTreeLL128Bw", "perChMaxNVLSTreeBw"):
        lines.append(f"  /* {name:<19} N1/N2/N4+ */ {_row(p[name])},")
    for name in ("ringBwCapInter", "ringBwCapIntra", "patBwEff", "patLatPerRank"):
        lines.append(f"  /* {name:<18} */ {_fmt(p[name])},")
    lines.append("  /* regBwScale [algo][LL, LL128, SIMPLE] (regBuff=1 busBw multiplier) */")
    lines.append("  {")
    for a in range(7):
        lines.append(f"    /* {ALGO_NAMES[a]:<14} */ {_row(p['regBwScale'][a])},")
    lines.append("  },")
    lines.append(f"  /* disableUnmodeledCells */ {int(p['disableUnmodeledCells'])},")
    lines.append("};")
    return "\n".join(lines)


def render_block(params, profile, validated, provenance_lines):
    begin = BEGIN_MARKER.format(profile=profile)
    end = END_MARKER.format(profile=profile)
    out = [begin,
           "// Generated by tuner/autotune (autotune codegen). Do not edit between the",
           "// markers by hand; re-run the fit + codegen instead."]
    out += ["// " + line for line in provenance_lines]
    vvar = PROFILE_VALIDATED_VARS[profile]
    out.append("// Set to 1 by codegen only when the fit passed the acceptance gate.  While 0,")
    out.append("// the profile is used only when selected explicitly via")
    out.append("// NCCL_TUNER_COST_MODEL_PROFILE.")
    out.append(f"static const int {vvar} = {1 if validated else 0};")
    out.append(render_initializer(params, profile))
    out.append(end)
    return "\n".join(out)


def replace_block(text, profile, block):
    begin = BEGIN_MARKER.format(profile=profile)
    end = END_MARKER.format(profile=profile)
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        raise KeyError(f"markers for profile '{profile}' not found in plugin source")
    return text[:i] + block + text[j + len(end):]
