import autopy
import random
import time
import numpy as np
import sys

from table_detection import TableDetector
from controller import GameController
from display import Display
from player import Player

current_goal = sys.argv[1]

while True:
    # Get preliminary screenshot.
    prelim_screenshot = GameController.get_screen_image()
    prelim_td = TableDetector()
    prelim_td.load_image(prelim_screenshot)
    prelim_td.detect_all()

    # Determine where to place the cue for real screenshot.
    prelim_controller = GameController(
        prelim_td.tableSize, prelim_td.tableCropTopLeft, prelim_td.balls, prelim_td.ballRadius)
    stick_target = prelim_controller.find_stick_position()
    prelim_controller.move_mouse(stick_target)

    # Get real screenshot.
    screenshot = GameController.get_screen_image()
    td = TableDetector()
    td.load_image(screenshot)
    td.detect_all()
    td.remove_nondup_balls(prelim_td.balls)

    player = Player(td.tableSize, td.pockets, td.balls, td.ballRadius, current_goal)
    target, force = player.get_shot(), 1

    controller = GameController(td.tableSize, td.tableCropTopLeft, td.balls, td.ballRadius)
    controller.make_shot(target, force)
    time.sleep(10)