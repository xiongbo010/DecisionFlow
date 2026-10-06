from decisionflow_experiments.trex_full import should_plan


class _Obstacle:
    x = 180


class _Snapshot:
    obstacles = (_Obstacle(),)
    trex_x = 50
    trex = (93, 0, False, False, False, False, "run")


def test_full_course_trigger_uses_visible_distance_and_posture():
    assert should_plan(_Snapshot(), "run", 100)
    assert not should_plan(_Snapshot(), "run", 80)
    assert should_plan(_Snapshot(), "duck", 80)
