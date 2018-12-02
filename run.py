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
    # Take a screenshot
    screenshot_name = GameController.get_screen_image()

    # Detect objects in the image
    td = TableDetector()
    td.load_image(screenshot_name)
    td.detect_all()

    # Create player to determine where to shoot
    player = Player(td.tableSize, td.pockets, td.balls, td.ballRadius, current_goal)
    target, force = player.get_shot(), 1

    # Create game controller
    controller = GameController(td.tableSize, td.tableCropTopLeft, td.balls["white"])
    controller.make_shot(target, force)
    time.sleep(10)