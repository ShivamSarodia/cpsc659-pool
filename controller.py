import autopy
import numpy as np
import time

class GameController:
    def __init__(self, table_size, crop_offset):
        # Size of the table image in pixels as (width, height)
        self.table_size = table_size

        # Coordinates of the top left of the table image, relative to the entire screen image.
        self.crop_offset = crop_offset

        # Coordinates of the cue ball, relative to the table image.
        self.cue_coords = None, None
    
    def set_cue_coords(self, coords):
        self.cue_coords = coords

    def drag_and_drop(self, start, end):
        start_x, start_y = start
        end_x, end_y = end

        adj_start = ((start_x + self.crop_offset[0]) / 2,
                     (start_y + self.crop_offset[1]) / 2)
        adj_end = ((end_x + self.crop_offset[0]) / 2,
                   (end_y + self.crop_offset[1]) / 2)

        autopy.mouse.move(*adj_start)
        time.sleep(0.5)
        autopy.mouse.toggle(None, True)
        autopy.mouse.smooth_move(*adj_end)
        autopy.mouse.toggle(None, False)

    def _get_edge_intersections(self, target):
        slope = (target[1] - self.cue_coords[1]) / (target[0] - self.cue_coords[0])
        x_to_y = lambda x: self.cue_coords[1] + slope * (x - self.cue_coords[0])
        y_to_x = lambda y: self.cue_coords[0] + (y - self.cue_coords[1]) / slope

        # Start the drag from an edge of the table image.
        intersections = []

        y = x_to_y(0)
        if y >= 0 and y <= self.table_size[1]:
            intersections.append((0, y))

        y = x_to_y(self.table_size[0])
        if y >= 0 and y <= self.table_size[1]:
            intersections.append((self.table_size[0], y))

        x = y_to_x(0)
        if x >= 0 and x <= self.table_size[0]:
            intersections.append((x, 0))

        x = y_to_x(self.table_size[1])
        if x >= 0 and x <= self.table_size[0]:
            intersections.append((x, self.table_size[1]))

        return intersections
    
    def make_shot(self, target, force):
        """Make a shot which launches the cue ball to make its center pass through the target point.
        
        Force is a value roughly between 0 and 1 that controls the force of the hit.
        """
        intersections = self._get_edge_intersections(target)
        print(intersections)

        if target[0] > self.cue_coords[0]:
            start = max(intersections, key=lambda i: i[0])
        elif target[0] < self.cue_coords[0]:
            start = min(intersections, key=lambda i: i[0])
        elif target[1] > self.cue_coords[1]:
            start = max(intersections, key=lambda i: i[1])
        elif target[1] < self.cue_coords[1]:
            start = min(intersections, key=lambda i: i[1])
        else:
            raise Exception("Could not find suitable start for shot.")

        drag_length = force * 500
        x_dist = target[0] - self.cue_coords[0]
        y_dist = target[1] - self.cue_coords[1]
        tot_dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        
        x_diff = (x_dist / tot_dist) * drag_length
        y_diff = (y_dist / tot_dist) * drag_length
        end = start[0] - x_diff, start[1] - y_diff

        self.drag_and_drop(start, end)
