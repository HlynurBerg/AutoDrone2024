#!/usr/bin/env python3.6
import rospy
import time
import Jetson.GPIO as GPIO
import smbus
from main import DroneControl
from datetime import datetime, timedelta

# Constants and Globals
ROS_DEBUG = True
DEVICE_BUS1 = 1
DEVICE_BUS0 = 0
DEVICE_ADDRESS1 = 0x18
DEVICE_ADDRESS2 = 0x19
DEVICE_ADDRESS3 = 0x18
DEVICE_ADDRESS4 = 0x19
EMERGENCY_STOP_PIN = 15  # GPIO pin for the emergency stop

# Initialize GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(EMERGENCY_STOP_PIN, GPIO.IN)

# Initialize I2C buses
bus1 = smbus.SMBus(DEVICE_BUS1)
bus0 = smbus.SMBus(DEVICE_BUS0)


def handle_relay(bus, device_address, state):
    """Turn relay on or off based on the state."""
    command = 0x01 if state else 0x00
    bus.write_byte(device_address, command)


def update_relay_states(communication, killswitch, bus1, bus0):
    """Update the state of the relays based on mode and killswitch."""
    if communication.mode in ['AUTO', 'GUIDED'] and not killswitch:
        handle_relay(bus1, DEVICE_ADDRESS1, True)
        handle_relay(bus1, DEVICE_ADDRESS2, False)
        handle_relay(bus0, DEVICE_ADDRESS3, False)
        handle_relay(bus0, DEVICE_ADDRESS4, True)
    elif communication.mode == 'MANUAL' and not killswitch:
        handle_relay(bus1, DEVICE_ADDRESS1, True)
        handle_relay(bus1, DEVICE_ADDRESS2, False)
        handle_relay(bus0, DEVICE_ADDRESS3, True)
        handle_relay(bus0, DEVICE_ADDRESS4, False)
    else:
        handle_relay(bus1, DEVICE_ADDRESS1, False)
        handle_relay(bus1, DEVICE_ADDRESS2, True)
        handle_relay(bus0, DEVICE_ADDRESS3, False)
        handle_relay(bus0, DEVICE_ADDRESS4, False)


def main():
    rospy.init_node("ColliosonAvoidanceControl", log_level=rospy.DEBUG if ROS_DEBUG else rospy.INFO)
    rate = rospy.Rate(1)  # 1 Hz
    drone_control = DroneControl()

    try:
        while not rospy.is_shutdown():
            killswitch = not GPIO.input(EMERGENCY_STOP_PIN)
            print(drone_control.communication.mode)
            update_relay_states(drone_control.communication, killswitch, bus1, bus0)
            if not killswitch:
                if drone_control.communication.mode in ['AUTO', 'GUIDED']:
                    drone_control.ship_mission()
                else:
                    rospy.loginfo("System is idle.")
            else:
                print("killswitch on!")
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node shutdown.")
    finally:
        GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset


if __name__ == "__main__":
    main()
