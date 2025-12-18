import numpy as np
import math

tile_size = 0.585  # meters
lane_width = tile_size / 2  # meters

x_min = 0
x_mid_1 = tile_size
x_mid_2 = 2 * tile_size
x_max = 3 * tile_size

y_min = 0
y_mid_1 = tile_size
y_mid_2 = 2 * tile_size
y_max = 3 * tile_size

# TODO: rewrite using tile pose information (x, y and orientation + relative to map), tile size and tile type
def is_out_of_lane(x, y):
    tile_num = tile_number(x, y)
    if tile_num == -1:
        return True
    if tile_num in [1, 3, 5, 7]:  # vertical and horizontal lanes -> easiest cases
        if tile_num == 1:
            lane_min_x = x_min
            lane_max_x = x_min + lane_width
            if not (lane_min_x <= x <= lane_max_x):
                return True
        elif tile_num == 3:
            lane_min_y = y_min
            lane_max_y = y_min + lane_width
            if not (lane_min_y <= y <= lane_max_y):
                return True
        elif tile_num == 5:
            lane_min_y = y_mid_2 + lane_width
            lane_max_y = y_max
            if not (lane_min_y <= y <= lane_max_y):
                return True
        elif tile_num == 7:
            lane_min_x = x_mid_2 + lane_width
            lane_max_x = x_max
            if not (lane_min_x <= x <= lane_max_x):
                return True
    else:  # corner tiles -> assuming tiles are quarter circles
        center_x = center_y = None
        if tile_num == 0:
            center_x = x_mid_1
            center_y = y_mid_1
        elif tile_num == 2:
            center_x = x_mid_1
            center_y = y_mid_2
        elif tile_num == 6:
            center_x = x_mid_2
            center_y = y_mid_1
        elif tile_num == 8:
            center_x = x_mid_2
            center_y = y_mid_2
        #print('############################################################')
        #print(f"(x, y) = ({round(x,2)}, {round(y, 2)})")
        #print(f"center_x =", center_x)
        #print(f"center_y =", center_y)
        #print(f"x**2 + y**2 =", (x - center_x) ** 2 + (y - center_y) ** 2)
        #print("tile_size ** 2 =", tile_size ** 2)
        #print("lane_width ** 2 =", lane_width ** 2)
        #print((tile_size ** 2) < ((x - center_x) ** 2 + (y - center_y) ** 2) < (lane_width) ** 2)

        r_square = (x - center_x) ** 2 + (y - center_y) ** 2
        if (tile_size ** 2) < r_square or r_square < (lane_width) ** 2:
                return True    
    return False

# TODO: compute distance from lane center (d) and orientation error (theta)
def compute_d(x, y):
    if is_out_of_lane(x, y):
        return -1
    
    # assuming the duckiebot is in line
    tile_num = tile_number(x, y)
    if tile_num in [1, 3, 5, 7]:  # vertical and horizontal lanes -> easiest cases
        if tile_num == 1: # vertical tile, distance in x
            center_road = x_min + lane_width / 2
            return abs(x - center_road)
        elif tile_num == 3:
            center_road = y_min + lane_width / 2
            return abs(y - center_road)
        elif tile_num == 5:
            center_road = y_mid_2 + 3 * lane_width / 2
            return abs(y - center_road)
        elif tile_num == 7:
            center_road = x_mid_2 + 3 * lane_width / 2
            return abs(x - center_road)
    else:
        center_x = center_y = None
        if tile_num == 0:
            center_x = x_mid_1
            center_y = y_mid_1
        elif tile_num == 2:
            center_x = x_mid_1
            center_y = y_mid_2
        elif tile_num == 6:
            center_x = x_mid_2
            center_y = y_mid_1
        elif tile_num == 8:
            center_x = x_mid_2
            center_y = y_mid_2
        
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return abs(r - 3 * lane_width / 2)

def compute_d_signed(x, y):
    """
    Compute signed distance from lane center.
    Positive = right side of center (toward yellow line), Negative = left side of center (toward white line).
    Returns -1 if out of lane.
    """
    if is_out_of_lane(x, y):
        return -1
    
    tile_num = tile_number(x, y)
    if tile_num in [1, 3, 5, 7]:  # vertical and horizontal lanes
        if tile_num == 1:  # vertical tile, distance in x
            center_road = x_min + lane_width / 2
            return x - center_road  # Positive if x > center (toward yellow line)
        elif tile_num == 3:
            center_road = y_min + lane_width / 2
            return y - center_road  # Positive if y > center
        elif tile_num == 5:
            center_road = y_mid_2 + 3 * lane_width / 2
            return y - center_road  # Positive if y > center
        elif tile_num == 7:
            center_road = x_mid_2 + 3 * lane_width / 2
            return x - center_road  # Positive if x > center
    else:
        # For corner tiles, compute signed distance from center radius
        center_x = center_y = None
        if tile_num == 0:
            center_x = x_mid_1
            center_y = y_mid_1
        elif tile_num == 2:
            center_x = x_mid_1
            center_y = y_mid_2
        elif tile_num == 6:
            center_x = x_mid_2
            center_y = y_mid_1
        elif tile_num == 8:
            center_x = x_mid_2
            center_y = y_mid_2
        else:
            return -1
        
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        target_radius = 3 * lane_width / 2
        return r - target_radius  # Positive if outside, negative if inside
    
    return -1
    
def compute_theta(x, y, yaw):
    def wrap_to_pi(angle):
        """Wrap any angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # TODO: implement termination if theta is greater or less than pi / 2
    if is_out_of_lane(x, y):
        return 1000

    tile_num = tile_number(x, y)

    # Straight tiles: fixed desired heading.
    if tile_num in [1, 3, 5, 7]:
        if tile_num == 1:
            desired_heading = -math.pi / 2
        elif tile_num == 3:
            desired_heading = 0
        elif tile_num == 5:
            desired_heading = math.pi
        elif tile_num == 7:
            desired_heading = math.pi / 2
        return wrap_to_pi(desired_heading - yaw)

    # Corner tiles: heading follows the tangent of the quarter-circle centerline (CCW loop).
    if tile_num == 0:
        center_x = x_mid_1
        center_y = y_mid_1
    elif tile_num == 2:
        center_x = x_mid_1
        center_y = y_mid_2
    elif tile_num == 6:
        center_x = x_mid_2
        center_y = y_mid_1
    elif tile_num == 8:
        center_x = x_mid_2
        center_y = y_mid_2
    else:
        # Should be unreachable if tile_number() is consistent.
        return 1000

    phi = math.atan2(y - center_y, x - center_x)
    # For tile 0 the atan2 range is [-pi, -pi/2]; unwrap so phi increases along the CCW path.
    if tile_num == 0 and phi < 0:
        phi += 2 * math.pi

    desired_heading = phi + math.pi / 2  # tangent for CCW traversal
    return wrap_to_pi(desired_heading - yaw)

def is_out_of_map(x, y):
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return True
    if (y_mid_1 < y < y_mid_2) and (x_mid_1 < x < x_mid_2): # in the hole
        return True
    return False

def tile_number(x, y):
    def in_tile_0_0(x, y):
        return (x_min <= x <= x_mid_1) and (y_min <= y <= y_mid_1)

    def in_tile_0_1(x, y):
        return (x_min <= x <= x_mid_1) and (y_mid_1 <= y <= y_mid_2)

    def in_tile_0_2(x, y):
        return (x_min <= x <= x_mid_1) and (y_mid_2 <= y <= y_max)

    def in_tile_1_0(x, y):
        return (x_mid_1 <= x <= x_mid_2) and (y_min <= y <= y_mid_1)

    def in_tile_1_2(x, y):
        return (x_mid_1 <= x <= x_mid_2) and (y_mid_2 <= y <= y_max)

    def in_tile_2_0(x, y):
        return (x_mid_2 <= x <= x_max) and (y_min <= y <= y_mid_1)

    def in_tile_2_1(x, y):
        return (x_mid_2 <= x <= x_max) and (y_mid_1 <= y <= y_mid_2)

    def in_tile_2_2(x, y):
        return (x_mid_2 <= x <= x_max) and (y_mid_2 <= y <= y_max)
    
    if is_out_of_map(x, y):
        return -1
    if in_tile_0_0(x, y):
        return 0
    if in_tile_0_1(x, y):
        return 1
    if in_tile_0_2(x, y):
        return 2
    if in_tile_1_0(x, y):
        return 3
    if in_tile_1_2(x, y):        
        return 5
    if in_tile_2_0(x, y):
        return 6
    if in_tile_2_1(x, y):
        return 7
    if in_tile_2_2(x, y):
        return 8
    raise RuntimeError("Unreachable")

def get_closest_tile(x, y):
    """
    Get the closest tile number to a given position, even if out of bounds.
    Returns the tile number (0-8) that is closest to the position.
    """
    # First try to get the tile directly
    tile = tile_number(x, y)
    if tile != -1:
        return tile
    
    # If out of map, find closest tile by distance to tile centers
    tile_centers = {
        0: (x_mid_1 / 2, y_mid_1 / 2),
        1: (x_mid_1 / 2, (y_mid_1 + y_mid_2) / 2),
        2: (x_mid_1 / 2, (y_mid_2 + y_max) / 2),
        3: ((x_mid_1 + x_mid_2) / 2, y_mid_1 / 2),
        5: ((x_mid_1 + x_mid_2) / 2, (y_mid_2 + y_max) / 2),
        6: ((x_mid_2 + x_max) / 2, y_mid_1 / 2),
        7: ((x_mid_2 + x_max) / 2, (y_mid_1 + y_mid_2) / 2),
        8: ((x_mid_2 + x_max) / 2, (y_mid_2 + y_max) / 2),
    }
    
    # Only consider drivable tiles (straight and curved)
    drivable_tiles = [0, 1, 2, 3, 5, 6, 7, 8]
    
    best_tile = 1  # Default
    best_distance = float('inf')
    
    for tile_num in drivable_tiles:
        if tile_num in tile_centers:
            center_x, center_y = tile_centers[tile_num]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance < best_distance:
                best_distance = distance
                best_tile = tile_num
    
    return best_tile

def random_initial_position(p):
    """
    Sample a random (x, y, yaw) within the right-hand lane.

    Args:
        p: Probability of choosing a curved tile; otherwise a straight tile.

    Returns:
        Tuple (x, y, yaw)
    """
    straight_tiles = [1, 3, 5, 7]
    curved_tiles = [0, 2, 6, 8]

    choose_curved = np.random.rand() < p
    tile = int(np.random.choice(curved_tiles if choose_curved else straight_tiles))

    lane_jitter = lane_width / 6.0  # stay close to lane centerline

    def wrap_to_pi(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    if tile in straight_tiles:
        if tile == 1:
            x_center = x_min + lane_width / 2
            y = np.random.uniform(y_mid_1, y_mid_2)
            x = np.random.uniform(x_center - lane_jitter, x_center + lane_jitter)
            desired_heading = -math.pi / 2
        elif tile == 3:
            y_center = y_min + lane_width / 2
            x = np.random.uniform(x_mid_1, x_mid_2)
            y = np.random.uniform(y_center - lane_jitter, y_center + lane_jitter)
            desired_heading = 0
        elif tile == 5:
            y_center = y_mid_2 + 3 * lane_width / 2  # center of top lane band
            x = np.random.uniform(x_mid_1, x_mid_2)
            y = np.random.uniform(y_center - lane_jitter, y_center + lane_jitter)
            desired_heading = math.pi
        elif tile == 7:
            x_center = x_mid_2 + 3 * lane_width / 2  # center of right lane band
            y = np.random.uniform(y_mid_1, y_mid_2)
            x = np.random.uniform(x_center - lane_jitter, x_center + lane_jitter)
            desired_heading = math.pi / 2
    else:
        if tile == 0:
            center_x, center_y = x_mid_1, y_mid_1
            phi_range = (math.pi, 1.5 * math.pi)
        elif tile == 2:
            center_x, center_y = x_mid_1, y_mid_2
            phi_range = (math.pi / 2, math.pi)
        elif tile == 6:
            center_x, center_y = x_mid_2, y_mid_1
            phi_range = (-math.pi / 2, 0)
        elif tile == 8:
            center_x, center_y = x_mid_2, y_mid_2
            phi_range = (0, math.pi / 2)

        phi = np.random.uniform(*phi_range)
        radius = 1.5 * lane_width + np.random.uniform(-lane_jitter, lane_jitter)
        x = center_x + radius * math.cos(phi)
        y = center_y + radius * math.sin(phi)
        desired_heading = phi + math.pi / 2

    yaw = wrap_to_pi(desired_heading + np.random.uniform(-math.pi / 4, math.pi / 4))
    return x, y, yaw

def perfect_initial_position(curve_prob: float = 0.5, tile: int | None = None, position_along_tile: float = 0.5):
    """
    Get a perfect (x, y, yaw) position in the exact center of the right-hand lane,
    with heading perfectly aligned to the lane direction.

    Args:
        curve_prob: Probability of choosing a curved tile; otherwise a straight tile.
                   Only used if tile is None.
        tile: Specific tile number to use (0-8). If None, randomly chooses based on curve_prob.
        position_along_tile: For straight tiles, position along the tile (0.0 to 1.0).
                            For curved tiles, position along the curve arc (0.0 to 1.0).

    Returns:
        Tuple (x, y, yaw) with perfect alignment
    """
    straight_tiles = [1, 3, 5, 7]
    curved_tiles = [0, 2, 6, 8]

    if tile is None:
        choose_curved = np.random.rand() < curve_prob
        tile = int(np.random.choice(curved_tiles if choose_curved else straight_tiles))
    else:
        tile = int(tile)

    def wrap_to_pi(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    if tile in straight_tiles:
        # Perfect center of lane, perfect heading
        if tile == 1:
            x_center = x_min + lane_width / 2
            y = y_mid_1 + position_along_tile * (y_mid_2 - y_mid_1)
            x = x_center  # Exact center, no jitter
            desired_heading = -math.pi / 2
        elif tile == 3:
            y_center = y_min + lane_width / 2
            x = x_mid_1 + position_along_tile * (x_mid_2 - x_mid_1)
            y = y_center  # Exact center, no jitter
            desired_heading = 0
        elif tile == 5:
            y_center = y_mid_2 + 3 * lane_width / 2  # center of top lane band
            x = x_mid_1 + position_along_tile * (x_mid_2 - x_mid_1)
            y = y_center  # Exact center, no jitter
            desired_heading = math.pi
        elif tile == 7:
            x_center = x_mid_2 + 3 * lane_width / 2  # center of right lane band
            y = y_mid_1 + position_along_tile * (y_mid_2 - y_mid_1)
            x = x_center  # Exact center, no jitter
            desired_heading = math.pi / 2
    else:
        # Curved tiles: perfect radius, perfect heading
        if tile == 0:
            center_x, center_y = x_mid_1, y_mid_1
            phi_range = (math.pi, 1.5 * math.pi)
        elif tile == 2:
            center_x, center_y = x_mid_1, y_mid_2
            phi_range = (math.pi / 2, math.pi)
        elif tile == 6:
            center_x, center_y = x_mid_2, y_mid_1
            phi_range = (-math.pi / 2, 0)
        elif tile == 8:
            center_x, center_y = x_mid_2, y_mid_2
            phi_range = (0, math.pi / 2)
        else:
            raise ValueError(f"Invalid curved tile number: {tile}")

        # Interpolate phi along the curve based on position_along_tile
        phi_min, phi_max = phi_range
        phi = phi_min + position_along_tile * (phi_max - phi_min)
        
        # Exact center radius (1.5 * lane_width), no jitter
        radius = 1.5 * lane_width
        x = center_x + radius * math.cos(phi)
        y = center_y + radius * math.sin(phi)
        desired_heading = phi + math.pi / 2

    # Perfect heading alignment, no angle noise
    yaw = wrap_to_pi(desired_heading)
    return x, y, yaw
