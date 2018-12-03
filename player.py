import numpy as np

class Player:
    def __init__(self, table_size, pockets, balls, ball_radius, goal_color):
        self.table_size = table_size
        self.raw_pockets = pockets
        self.balls = balls
        self.ball_radius = ball_radius
        self.goal_color = goal_color

        assert self.balls["white"] is not None
        self.all_balls = self.balls["stripes"] + self.balls["solids"]
        self.all_balls.append(self.balls["white"])
        if "black" in self.balls:
            self.all_balls.append(self.balls["black"])

        # For each pocket, define a "pocket target" point to aim at in order to enter the pocket.
        self.pocket_targets = {}
        self.pocket_targets['mr'] = self.raw_pockets['mr']
        self.pocket_targets['ml'] = self.raw_pockets['ml']

        pocket_offset = self.ball_radius * 1.35
        for l in ["tl", "tr", "bl", "br"]:
            x, y = self.raw_pockets[l]
            if x == 0:
                x += pocket_offset
            else:
                x -= pocket_offset

            if y == 0:
                y += pocket_offset
            else:
                y -= pocket_offset
            self.pocket_targets[l] = (x,y)

        self.MIN_COS_ANGLE = -0.2
        self.MAX_MID_POCKET_RATIO = 0.6
        self.COLLISION_RANGE = 2.1

    def _is_same_ball(self, b1, b2):
        return np.abs(b1[0] - b2[0]) + np.abs(b1[1] - b2[1]) < 1

    def _get_rotation(self, p1, p2):
        """Produce a rotation matrix such that the first coordinates of p1' and p2' are equal."""
        delta = p2 - p1
        l = np.linalg.norm(delta)
        return np.array([[delta[1] / l, -delta[0] / l], [delta[0] / l, delta[1] / l]])

    def _clear_distance(self, start, end, excepts=[]):
        """
        If a point travels from `start` to `end`, what is the minimum distance
        between the point and the center of another ball during its trajectory?

        Ignores the centers of balls listed in `excepts`.
        """
        start = np.array(start)
        end = np.array(end)

        rotation = self._get_rotation(start, end)
        rot_start = rotation.dot(start)
        rot_end = rotation.dot(end)

        min_dist = 2 * (self.table_size[0] + self.table_size[1])
        for ball in self.all_balls:
            # Skip balls that are very close to a ball in the excepts list.
            to_skip = False
            for b2 in excepts:
                if self._is_same_ball(ball, b2):
                    to_skip = True
            if to_skip:
                continue

            rot_ball = rotation.dot(np.array(ball))
            if min(rot_start[1], rot_end[1]) <= rot_ball[1] <= max(rot_start[1], rot_end[1]):
                dist = np.abs(rot_ball[0] - rot_start[0])
                min_dist = min(dist, min_dist)

        return min_dist

    def _is_clear(self, start, end, excepts=[]):
        """
        Can a ball travel from `start` to `end` without striking another ball?

        Ignores the balls listed in `excepts`.
        """
        return self._clear_distance(start, end, excepts) > self.COLLISION_RANGE * self.ball_radius

    def _get_shots(self):
        shots = []
        if self.goal_color == "black":
            target_balls = [self.balls["black"]]
        elif self.goal_color is None:
            target_balls = self.balls["stripes"] + self.balls["solids"]
        else:
            target_balls = self.balls[self.goal_color]

        for ball in target_balls:
            for l in self.pocket_targets:
                new_shots = self._create_shots(ball, l)
                if new_shots:
                    shots.extend(new_shots)
        return shots

    def _get_reflection_point(self, src, target, orientation, pos):
        if (target[0] - src[0]) == 0:
            return (target[0], pos)
        if (target[1] - src[1]) == 0:
            return (pos, target[1])
        slope = (target[1] - src[1]) / (target[0] - src[0])
        x_to_y = lambda x: src[1] + slope * (x - src[0])
        y_to_x = lambda y: src[0] + (y - src[1]) / slope

        if orientation == 'h':
            x = y_to_x(pos)
            if x >= 0 and x <= self.table_size[0]:
                return(x, pos)
        else:
            y = x_to_y(pos)
            if y >= 0 and y <= self.table_size[1]:
                return(pos, y)

        return None

    def _rebound_shots(self, cue, target, target_ball):
        """
        Compute if any rebound shots exist
        """
        shifted_edges = [(0, self.ball_radius), (0, self.table_size[1]-self.ball_radius), (self.ball_radius, 0), (self.table_size[0]-self.ball_radius, 0)]

        shots = []
        for edge in shifted_edges:
            incidence_point = None
            if edge[0] == 0:
                # horizontal table edge
                reflection = (target[0], 2*edge[1]-target[1])
                incidence_point = self._get_reflection_point(cue, reflection, 'h', edge[1])
            else:
                # vertical edge
                reflection = (2*edge[0]-target[0], target[1])
                incidence_point = self._get_reflection_point(cue, reflection, 'v', edge[0])

            if self._is_clear(cue, incidence_point, [cue, target_ball]) and self._is_clear(incidence_point, target, [cue, target_ball]):
                shots.append(incidence_point)

        return shots


    def _create_shots(self, ball, pocket_label):
        """Create a shot to launch given ball into given pocket.

        If no such shot exists, return None.
        """
        pocket = np.array(self.pocket_targets[pocket_label])
        ball = np.array(ball)
        cue = np.array(self.balls["white"])

        # If this is a middle pocket, is the ball in range of the pocket?
        if pocket_label in {'ml', 'mr'}:
            angle_ratio = np.abs(ball[1] - pocket[1]) / np.abs(ball[0] - pocket[0])
            if angle_ratio > self.MAX_MID_POCKET_RATIO:
                return None

        # Is path from ball to pocket clear?
        if not self._is_clear(ball, pocket, [cue, ball]):
            return None

        unit_towards_target = (ball - pocket[0]) / np.linalg.norm(ball - pocket[0])
        target = ball + 2 * self.ball_radius * unit_towards_target

        # get reflection shots (if any)
        reflection_targets = self._rebound_shots(cue, target, ball)
        possible_targets = [(pocket, 'direct')]
        for rebound_target in reflection_targets:
            possible_targets.append((rebound_target, 'rebound'))

        shots = []
        for final_target in possible_targets:
            # Compute target position for cue ball
            if final_target[1] == 'direct':
                target = final_target[0]
                white_dist = np.linalg.norm(cue - target)

                if not self._is_clear(cue, target, [cue, ball]):
                    return None
            else:
                target = final_target[0]
                white_dist = 100 * (np.linalg.norm(cue - target) + np.linalg.norm(target - ball))

            # Compute quality of angle from cue ball to target
            v1 = cue - target
            v2 = pocket - target
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            if final_target[1] == 'direct' and cos_angle > self.MIN_COS_ANGLE:
                continue

            shots.append(Shot(target, pocket, -cos_angle, white_dist, np.linalg.norm(pocket - ball)))
        return shots

    def get_shot(self):
        if self._is_break():
            print("Breaking...")
            return (self.table_size[0] / 2, 0), 1

        shots = self._get_shots()
        if shots:
            best_shot = max(shots, key=lambda shot: shot.quality())
            total_dist = best_shot.white_dist + best_shot.target_dist
            return best_shot.target, max(min(total_dist / (900 * best_shot.cos_angle), 1), 0.5)
        else:
            print("No shots available! Doing a Hail Mary.")
            return self.hail_mary(), 0.7

    def hail_mary(self):
        """Aim directly at some ball of the correct color."""
        for ball in self.balls[self.goal_color]:
            if self._is_clear(self.balls["white"], ball, excepts=[self.balls["white"], ball]):
                return ball
        return self.balls[self.goal_color][0]

    def _is_break(self):
        # Normally, there's 16 balls.
        if len(self.all_balls) <= 14:
            return False

        not_near = 0
        for b1 in self.all_balls:
            nearby = 0
            for b2 in self.all_balls:
                ball_dist = np.linalg.norm(np.array(b1) - np.array(b2))
                if ball_dist < 2.5 * self.ball_radius:
                    nearby += 1
            if nearby < 2:
                not_near += 1

        return not_near < 2

class Shot:
    def __init__(self, target, pocket, cos_angle, white_dist, target_dist):
        self.target = target[0], target[1]
        self.pocket = pocket[0], pocket[1]
        self.cos_angle = cos_angle  # larger is better
        self.white_dist = white_dist
        self.target_dist = target_dist

    def quality(self):
        return -(self.white_dist + self.target_dist)
