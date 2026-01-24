import unittest
from internal.domain.value_object.throttle import Throttle
from internal.domain.value_object.steer import Steer
from internal.domain.value_object.drive_state import DriveState
from internal.domain.entity.drive_control import DriveControl

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
        # Safe distances
        distances = [1.0, 0.5, 0.8]
        self.assertTrue(DriveState.READY.can_transition_to_moving(distances))
        
        # Dangerous distances
        distances = [1.0, 0.1, 0.8]
        self.assertFalse(DriveState.READY.can_transition_to_moving(distances))
        self.assertEqual(DriveState.from_distances(distances, DriveState.READY), DriveState.BLOCKED)

    def test_drive_control_state_transition(self):
        ctrl = DriveControl.create_default()
        self.assertEqual(ctrl.state, DriveState.READY)
        
        # Transition to MOVING when safe
        ctrl.update_throttle(1600, [1.0, 1.0])
        self.assertEqual(ctrl.state, DriveState.MOVING)
        self.assertEqual(ctrl.throttle.value, 1600)
        
        # Blocked while moving
        ctrl.update_state([0.1, 1.0])
        self.assertEqual(ctrl.state, DriveState.BLOCKED)
        self.assertEqual(ctrl.throttle.value, 0)
        
        # Safe again
        ctrl.update_state([1.0, 1.0])
        self.assertEqual(ctrl.state, DriveState.READY)

    def test_drive_control_backward_in_blocked(self):
        ctrl = DriveControl.create_default()
        # Front obstacle -> BLOCKED
        ctrl.update_state([0.1, 1.0])
        self.assertEqual(ctrl.state, DriveState.BLOCKED)
        self.assertEqual(ctrl.throttle.value, 0)
        
        # Try forward (fixed_us = 1600) -> Should stay BLOCKED / stop
        ctrl.update_throttle(1600, [0.1, 1.0])
        self.assertEqual(ctrl.throttle.value, 0)
        self.assertEqual(ctrl.state, DriveState.BLOCKED)
        
        # Try backward (back_us = 1400) -> Should allow MOVING
        ctrl.update_throttle(1400, [0.1, 1.0])
        self.assertEqual(ctrl.throttle.value, 1400)
        self.assertEqual(ctrl.state, DriveState.MOVING)

if __name__ == '__main__':
    unittest.main()
