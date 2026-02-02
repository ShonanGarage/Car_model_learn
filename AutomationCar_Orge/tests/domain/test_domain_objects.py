import unittest
from internal.domain.value_object.throttle import Throttle
from internal.domain.value_object.steer import Steer
from internal.domain.value_object.drive_state import DriveState
from internal.domain.entity.vehicle_motion import VehicleMotion, boot_vehicle_motion_ready
from internal.domain.value_object.safety_rules import SafetyRules
from internal.domain.value_object.sonar_frame import SonarFrame

class TestDomainObjects(unittest.TestCase):
    def test_throttle_creation(self):
        t = Throttle(1600)
        self.assertEqual(t.value, 1600)
        self.assertTrue(t.is_forward())
        self.assertFalse(t.is_backward())
        self.assertFalse(t.is_stop())

    def test_throttle_stop(self):
        t = Throttle.stop()
        self.assertEqual(t.value, 0)
        self.assertTrue(t.is_stop())

    def test_steer_creation(self):
        s = Steer(1500)
        self.assertEqual(s.value, 1500)
        self.assertTrue(s.is_center())

    def test_drive_state_validation(self):
        rules = SafetyRules(emergency_stop_threshold_m=0.1, blocked_threshold_m=0.2)
        # Safe distances
        frame = SonarFrame(front=[1.0, 0.5, 0.8], rear=[1.0, 1.0])
        self.assertEqual(
            DriveState.from_sonar_frame(frame, DriveState.ready(), rules),
            DriveState.ready(),
        )
        
        # Dangerous distances
        frame = SonarFrame(front=[1.0, 0.1, 0.8], rear=[1.0, 1.0])
        self.assertEqual(
            DriveState.from_sonar_frame(frame, DriveState.ready(), rules),
            DriveState.BLOCKED_FRONT,
        )

    def test_vehicle_motion_state_transition(self):
        rules = SafetyRules(emergency_stop_threshold_m=0.1, blocked_threshold_m=0.2)
        ctrl = boot_vehicle_motion_ready(steer=Steer(1500))
        self.assertEqual(ctrl.state, DriveState.ready())

        # Transition to MOVING when safe
        ctrl = ctrl.apply(SonarFrame(front=[1.0, 1.0, 1.0], rear=[1.0, 1.0]), 1600, ctrl.steer.value, rules)
        self.assertEqual(ctrl.state, DriveState.moving())
        self.assertEqual(ctrl.throttle.value, 1600)

        # Blocked while moving
        ctrl = ctrl.apply(SonarFrame(front=[0.1, 1.0, 1.0], rear=[1.0, 1.0]), ctrl.throttle.value, ctrl.steer.value, rules)
        self.assertEqual(ctrl.state, DriveState.BLOCKED_FRONT)
        self.assertEqual(ctrl.throttle.value, 0)

        # Safe again
        ctrl = ctrl.apply(SonarFrame(front=[1.0, 1.0, 1.0], rear=[1.0, 1.0]), ctrl.throttle.value, ctrl.steer.value, rules)
        self.assertEqual(ctrl.state, DriveState.ready())

    def test_vehicle_motion_backward_in_blocked(self):
        rules = SafetyRules(emergency_stop_threshold_m=0.1, blocked_threshold_m=0.2)
        ctrl = boot_vehicle_motion_ready(steer=Steer(1500))
        # Front obstacle -> BLOCKED
        ctrl = ctrl.apply(SonarFrame(front=[0.1, 1.0, 1.0], rear=[1.0, 1.0]), ctrl.throttle.value, ctrl.steer.value, rules)
        self.assertEqual(ctrl.state, DriveState.BLOCKED_FRONT)
        self.assertEqual(ctrl.throttle.value, 0)

        # Try forward (fixed_us = 1600) -> Should stay BLOCKED / stop
        ctrl = ctrl.apply(SonarFrame(front=[0.1, 1.0, 1.0], rear=[1.0, 1.0]), 1600, ctrl.steer.value, rules)
        self.assertEqual(ctrl.throttle.value, 0)
        self.assertEqual(ctrl.state, DriveState.BLOCKED_FRONT)

        # Try backward (back_us = 1400) -> Should allow MOVING
        ctrl = ctrl.apply(SonarFrame(front=[0.1, 1.0, 1.0], rear=[1.0, 1.0]), 1400, ctrl.steer.value, rules)
        self.assertEqual(ctrl.throttle.value, 1400)
        self.assertEqual(ctrl.state, DriveState.moving())

if __name__ == '__main__':
    unittest.main()
