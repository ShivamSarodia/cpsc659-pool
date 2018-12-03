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

        self.MIN_DIRECT_COS_ANGLE = -0.2
        self.MIN_REBOUND_COS_ANGLE = -0.7
        self.MAX_MID_POCKET_RATIO = 0.7
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
            if min(rot_start[1], rot_end[1]) - self.ball_radius <= rot_ball[1] <= max(rot_start[1], rot_end[1]) + self.ball_radius:
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
        shifted_edges = [(0, 4+self.ball_radius), (0, self.table_size[1]-2-self.ball_radius), (self.ball_radius+3, 0), (self.table_size[0]-self.ball_radius-2, 0)]

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

            if incidence_point and self._is_clear(cue, incidence_point, [cue]) and self._is_clear(incidence_point, target, [cue, target_ball]):
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

        unit_towards_target = (ball - pocket) / np.linalg.norm(ball - pocket)
        target_near_ball = ball + 2 * self.ball_radius * unit_towards_target

        # get reflection shots (if any)
        reflection_targets = self._rebound_shots(cue, target_near_ball, ball)
        possible_targets = [(target_near_ball, 'direct')]
        for rebound_target in reflection_targets:
            possible_targets.append((rebound_target, 'rebound'))

        shots = []
        for final_target in possible_targets:
            # Compute target position for cue ball
            target = np.array(final_target[0])

            if final_target[1] == 'direct':
                if not self._is_clear(cue, target, [cue, ball]):
                    return None

                white_dist = np.linalg.norm(cue - target)
                v1 = cue - target
                v2 = pocket - target
            else:
                white_dist = (np.linalg.norm(cue - target) + np.linalg.norm(target - ball))
                v1 = target - target_near_ball
                v2 = pocket - target_near_ball

            # Compute quality of angle from cue ball to target
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if cos_angle > self.MIN_DIRECT_COS_ANGLE and final_target[1] == 'direct':
                continue
            if cos_angle > self.MIN_REBOUND_COS_ANGLE and final_target[1] == 'rebound':
                continue

            shot_params = {
                "target": target,
                "pocket": pocket,
                "second_target": target_near_ball,
                "white_travel_dist": white_dist,
                "target_travel_dist": np.linalg.norm(pocket - ball),
                "cos_angle": -cos_angle,
            }
            if final_target[1] == "direct":
                shot_params["second_target"] = None
            shots.append(Shot(**shot_params))
        return shots

    def get_shot(self, flex=False):
        if self._is_break():
            shot_params = {
                "target": (self.table_size[0] / 2, 0),
                "is_break": True
            }
            return Shot(**shot_params)

        shots = self._get_shots()
        if shots:
            return max(shots, key=lambda shot: shot.quality(flex=flex))
        else:
            return self.hail_mary()

    def hail_mary(self):
        """Aim directly at some ball of the correct color."""
        if self.goal_color == "black":
            target_balls = [self.balls["black"]]
        elif self.goal_color is None:
            target_balls = self.balls["stripes"] + self.balls["solids"]
        else:
            target_balls = self.balls[self.goal_color]

        for ball in target_balls:
            if self._is_clear(self.balls["white"], ball, excepts=[self.balls["white"], ball]):
                break

        shot_params = {
            "target": ball,
            "is_hail_mary": True
        }
        return Shot(**shot_params)

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
    def __init__(self, target, pocket=None, second_target=None, white_travel_dist=None, target_travel_dist=None, cos_angle=None, is_hail_mary=False, is_break=False):
        # The point to aim the white ball at.
        if target is not None:
            self.target = target[0], target[1]
        else:
            self.target = None

        # The pocket at which the ball is being launched.
        if pocket is not None:
            self.pocket = pocket[0], pocket[1]
        else:
            self.pocket = None

        # If this is a rebound shot, the point near the target ball
        # the white ball should bounce to. Otherwise, set to None.
        self.second_target = second_target

        # Distance white ball travels before reaching target ball.
        self.white_travel_dist = white_travel_dist

        # Distance target ball travels before reaching pocket.
        self.target_travel_dist = target_travel_dist

        # Negative cos(theta) of angle between white ball and target path. Larger is better.
        self.cos_angle = cos_angle 

        # True if this is a hail mary shot.
        self.is_hail_mary = is_hail_mary

        # True if this is a break.
        self.is_break = is_break

    def force(self):
        if self.is_hail_mary:
            return 0.7
        if self.is_break:
            return 1
            
        if self.second_target is None:
            dist_factor = (self.white_travel_dist + self.target_travel_dist) / 900
        else:
            dist_factor = (self.white_travel_dist / 2 + self.target_travel_dist) / 900

        force = dist_factor / self.cos_angle
        if force > 1:
            return 1
        elif force < 0.5:
            return 0.5
        else:
            return force

    def quality(self, flex=False):
        print("---------------------------------")

        if self.is_hail_mary:
            return -10000

        if self.is_break:
            return 0

        qualities = []
        if self.second_target is not None:
            # If we're flexing, improve weight on bounce shots.
            if flex:
                qualities.append(3000)
            else:
                qualities.append(-1500)
            print(f"Rebound shot quality: {qualities[-1]}")

        if self.second_target is not None:
            # Reduce white travel cost for rebound shots.
            qualities.append(-self.white_travel_dist / 2)
        else:
            qualities.append(-self.white_travel_dist)
        print(f"White travel quality: {qualities[-1]}")

        qualities.append(-4 * self.target_travel_dist)
        print(f"Target travel quality: {qualities[-1]}")

        qualities.append(self.cos_angle * 1800)
        print(f"Cos angle quality: {qualities[-1]}")

        print(f"Total quality: {sum(qualities)}")
        return sum(qualities)