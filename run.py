import autopy
import random
import time
import numpy as np

from table_detection import TableDetector
from controller import GameController
from display import Display
from player import Player

time.sleep(2)

while True:
    # Take a screenshot
    screenshot_name = GameController.get_screen_image()

    # Detect objects in the image
    td = TableDetector()
    td.load_image(screenshot_name)
    td.detect_all()

    if td.turnIsOver:
        break

    # Create player to determine where to shoot
    player = Player(td.tableSize, td.pockets, td.balls)
    target, force = player.get_move()

    # Create game controller
    controller = GameController(td.table_size, td.tableCropTopLeft, td.balls)
    controller.make_shot((target, force)
    time.sleep(10)

print("Turn over!")