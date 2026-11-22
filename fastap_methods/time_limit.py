"""Default QMIX target at a scenario's time limit for new training runs."""


def default_bootstrap_at_time_limit(scene_key):
    return scene_key not in ("lbf", "fastap-lbf-6x6")
