from reachaq.core.interfaces import CoverServoStatus


def test_or_value():
    ok = CoverServoStatus.OK
    err1 = CoverServoStatus.COVER_POSITION_ERROR
    err2 = CoverServoStatus.RELEASE_POSITION_ERROR
    combined_err = CoverServoStatus(ok | err1 | err2)
    assert combined_err is CoverServoStatus.COVER_AND_RELEASE_POS_ERROR
