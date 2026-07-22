"""Deterministic connected-component inventory for 2D ARC color grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARC_COLOR_NAMES: dict[int, str] = {
    0: "white",
    1: "off_white",
    2: "light_gray",
    3: "gray",
    4: "dark_gray",
    5: "black",
    6: "magenta",
    7: "light_magenta",
    8: "red",
    9: "blue",
    10: "light_blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


Pixel = tuple[int, int]  # (row, column)


@dataclass(frozen=True)
class _Component:
    label: int
    color_id: int
    pixels: frozenset[Pixel]
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1; inclusive
    centroid: tuple[float, float]  # x, y
    width: int
    height: int
    area: int
    perimeter: int
    edge_contacts: tuple[str, ...]
    normalized_pixels: tuple[Pixel, ...]
    shape_signature: str
    rotation_signature: str
    dihedral_signature: str
    primitive_signature: str
    primitive_rotation_signature: str
    primitive_dihedral_signature: str
    scale_factor: int
    shape_kind: str
    orientation: str
    fill_ratio: float
    symmetry_horizontal: bool
    symmetry_vertical: bool
    symmetry_180: bool
    holes: int
    enclosed_pixels: frozenset[Pixel]
    role_hint: str


@dataclass(frozen=True)
class _Track:
    stable_id: str
    component: _Component
    action_number: int
    velocity: tuple[float, float]
    missed_analyses: int = 0


def _shape_encoding(pixels: Iterable[Pixel]) -> tuple[str, tuple[Pixel, ...], int, int]:
    points = tuple(sorted(pixels))
    if not points:
        return "0x0:", (), 0, 0
    min_y = min(point[0] for point in points)
    min_x = min(point[1] for point in points)
    normalized = tuple(sorted((y - min_y, x - min_x) for y, x in points))
    height = max(y for y, _ in normalized) + 1
    width = max(x for _, x in normalized) + 1
    payload = f"{height}x{width};" + ";".join(f"{y},{x}" for y, x in normalized)
    digest = hashlib.blake2s(payload.encode("ascii"), digest_size=6).hexdigest()
    return f"{height}x{width}:{digest}", normalized, height, width


def _rotate_shape(pixels: Sequence[Pixel], degrees: int) -> tuple[Pixel, ...]:
    if not pixels:
        return ()
    current = tuple(pixels)
    turns = (degrees // 90) % 4
    for _ in range(turns):
        height = max(y for y, _ in current) + 1
        current = tuple(sorted((x, height - 1 - y) for y, x in current))
    min_y = min(y for y, _ in current)
    min_x = min(x for _, x in current)
    return tuple(sorted((y - min_y, x - min_x) for y, x in current))


def _rotation_signature(pixels: Sequence[Pixel]) -> str:
    encodings: list[str] = []
    for degrees in (0, 90, 180, 270):
        rotated = _rotate_shape(pixels, degrees)
        height = max((y for y, _ in rotated), default=-1) + 1
        width = max((x for _, x in rotated), default=-1) + 1
        encodings.append(
            f"{height}x{width};" + ";".join(f"{y},{x}" for y, x in rotated)
        )
    canonical = min(encodings)
    digest = hashlib.blake2s(canonical.encode("ascii"), digest_size=6).hexdigest()
    return f"rot:{digest}"


def _reflected_shape(pixels: Sequence[Pixel]) -> tuple[Pixel, ...]:
    if not pixels:
        return ()
    width = max(x for _, x in pixels) + 1
    return tuple(sorted((y, width - 1 - x) for y, x in pixels))


def _dihedral_signature(pixels: Sequence[Pixel]) -> str:
    """Return a signature invariant to quarter-turns and reflection."""
    encodings: list[str] = []
    for base in (tuple(pixels), _reflected_shape(pixels)):
        for degrees in (0, 90, 180, 270):
            rotated = _rotate_shape(base, degrees)
            height = max((y for y, _ in rotated), default=-1) + 1
            width = max((x for _, x in rotated), default=-1) + 1
            encodings.append(
                f"{height}x{width};" + ";".join(f"{y},{x}" for y, x in rotated)
            )
    canonical = min(encodings)
    digest = hashlib.blake2s(canonical.encode("ascii"), digest_size=6).hexdigest()
    return f"dih:{digest}"


def _primitive_shape(pixels: Sequence[Pixel]) -> tuple[tuple[Pixel, ...], int]:
    """Remove the largest exact integer block scale from a binary shape."""
    if not pixels:
        return (), 1
    height = max(y for y, _ in pixels) + 1
    width = max(x for _, x in pixels) + 1
    mask = np.zeros((height, width), dtype=bool)
    for y, x in pixels:
        mask[y, x] = True
    common = math.gcd(height, width)
    for factor in range(common, 1, -1):
        if height % factor or width % factor:
            continue
        blocks = mask.reshape(height // factor, factor, width // factor, factor)
        all_on = blocks.all(axis=(1, 3))
        any_on = blocks.any(axis=(1, 3))
        if np.array_equal(all_on, any_on):
            reduced = tuple(
                (int(y), int(x)) for y, x in zip(*np.nonzero(all_on), strict=True)
            )
            return reduced, factor
    return tuple(pixels), 1


def _rotation_between(previous: _Component, current: _Component) -> int | None:
    for degrees in (0, 90, 180, 270):
        if _rotate_shape(previous.normalized_pixels, degrees) == current.normalized_pixels:
            return degrees
    return None


def _geometry_features(
    pixels: Sequence[Pixel], height: int, width: int, holes: int
) -> tuple[str, str, float, bool, bool, bool]:
    """Classify only geometries that are unambiguous on the normalized mask."""
    mask = np.zeros((height, width), dtype=bool)
    for y, x in pixels:
        mask[y, x] = True
    area = len(pixels)
    fill_ratio = area / (height * width)
    symmetry_horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    symmetry_vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    symmetry_180 = bool(np.array_equal(mask, np.rot90(mask, 2)))

    if area == 1:
        return (
            "point",
            "none",
            fill_ratio,
            symmetry_horizontal,
            symmetry_vertical,
            symmetry_180,
        )
    if height == 1 or width == 1:
        orientation = "horizontal" if height == 1 else "vertical"
        return (
            "line",
            orientation,
            fill_ratio,
            symmetry_horizontal,
            symmetry_vertical,
            symmetry_180,
        )
    if area == height * width:
        shape_kind = "filled square" if height == width else "filled rectangle"
        orientation = "square" if height == width else "horizontal" if width > height else "vertical"
        return (
            shape_kind,
            orientation,
            fill_ratio,
            symmetry_horizontal,
            symmetry_vertical,
            symmetry_180,
        )
    if holes:
        orientation = "square" if height == width else "horizontal" if width > height else "vertical"
        return (
            "hollow/frame/container",
            orientation,
            fill_ratio,
            symmetry_horizontal,
            symmetry_vertical,
            symmetry_180,
        )

    occupied = set(pixels)
    if area == height + width - 1:
        full_rows = [y for y in range(height) if bool(mask[y, :].all())]
        full_columns = [x for x in range(width) if bool(mask[:, x].all())]
        for row in full_rows:
            for column in full_columns:
                union = {(row, x) for x in range(width)} | {(y, column) for y in range(height)}
                if occupied != union:
                    continue
                row_edge = row in (0, height - 1)
                column_edge = column in (0, width - 1)
                if row_edge and column_edge:
                    vertical = "top" if row == 0 else "bottom"
                    horizontal = "left" if column == 0 else "right"
                    shape_kind = "L-like"
                    orientation = f"corner_{vertical}_{horizontal}"
                elif row_edge:
                    shape_kind = "T-like"
                    orientation = "stem_down" if row == 0 else "stem_up"
                elif column_edge:
                    shape_kind = "T-like"
                    orientation = "stem_right" if column == 0 else "stem_left"
                else:
                    shape_kind = "cross-like"
                    orientation = "orthogonal"
                return (
                    shape_kind,
                    orientation,
                    fill_ratio,
                    symmetry_horizontal,
                    symmetry_vertical,
                    symmetry_180,
                )

    orientation = "balanced"
    if width >= 1.25 * height:
        orientation = "horizontal"
    elif height >= 1.25 * width:
        orientation = "vertical"
    return (
        "angular/irregular",
        orientation,
        fill_ratio,
        symmetry_horizontal,
        symmetry_vertical,
        symmetry_180,
    )


def _hole_data(
    pixels: frozenset[Pixel], bbox: tuple[int, int, int, int]
) -> tuple[int, frozenset[Pixel]]:
    x0, y0, x1, y1 = bbox
    height = y1 - y0 + 1
    width = x1 - x0 + 1
    occupied = np.zeros((height, width), dtype=bool)
    for y, x in pixels:
        occupied[y - y0, x - x0] = True

    exterior = np.zeros_like(occupied)
    queue: deque[Pixel] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if not occupied[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not occupied[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and not occupied[ny, nx]
                and not exterior[ny, nx]
            ):
                exterior[ny, nx] = True
                queue.append((ny, nx))

    enclosed_mask = ~occupied & ~exterior
    enclosed_local = set(zip(*np.nonzero(enclosed_mask), strict=True))
    if not enclosed_local:
        return 0, frozenset()

    remaining = set(enclosed_local)
    holes = 0
    while remaining:
        holes += 1
        start = remaining.pop()
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (y + dy, x + dx)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)

    enclosed_global = frozenset((y + y0, x + x0) for y, x in enclosed_local)
    return holes, enclosed_global


def _role_hint(
    area: int,
    width: int,
    height: int,
    edge_contacts: Sequence[str],
    frame_height: int,
    frame_width: int,
    is_largest: bool,
) -> str:
    frame_area = frame_height * frame_width
    if is_largest or area >= max(1, round(frame_area * 0.20)) or len(edge_contacts) >= 3:
        return "terrain/background"

    thin_limit = max(2, min(frame_height, frame_width) // 16)
    elongated = max(width, height) >= 3 * max(1, min(width, height))
    thin = min(width, height) <= thin_limit
    if edge_contacts and (elongated or thin):
        return "edge_region_candidate"
    return "object_candidate"


def _extract_components(frame: np.ndarray) -> tuple[list[_Component], np.ndarray]:
    height, width = frame.shape
    visited = np.zeros((height, width), dtype=bool)
    labels = np.full((height, width), -1, dtype=np.int32)
    partial: list[dict[str, Any]] = []

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            label = len(partial)
            color_id = int(frame[start_y, start_x])
            queue: deque[Pixel] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            pixels: set[Pixel] = set()
            perimeter = 0
            while queue:
                y, x = queue.popleft()
                pixels.add((y, x))
                labels[y, x] = label
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        perimeter += 1
                    elif int(frame[ny, nx]) != color_id:
                        perimeter += 1
                    elif not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            frozen_pixels = frozenset(pixels)
            min_y = min(y for y, _ in pixels)
            max_y = max(y for y, _ in pixels)
            min_x = min(x for _, x in pixels)
            max_x = max(x for _, x in pixels)
            bbox = (min_x, min_y, max_x, max_y)
            shape_signature, normalized, component_height, component_width = _shape_encoding(pixels)
            holes, enclosed = _hole_data(frozen_pixels, bbox)
            primitive, scale_factor = _primitive_shape(normalized)
            primitive_signature, _, _, _ = _shape_encoding(primitive)
            (
                shape_kind,
                orientation,
                fill_ratio,
                symmetry_horizontal,
                symmetry_vertical,
                symmetry_180,
            ) = _geometry_features(normalized, component_height, component_width, holes)
            edges: list[str] = []
            if min_y == 0:
                edges.append("top")
            if max_x == width - 1:
                edges.append("right")
            if max_y == height - 1:
                edges.append("bottom")
            if min_x == 0:
                edges.append("left")
            partial.append(
                {
                    "label": label,
                    "color_id": color_id,
                    "pixels": frozen_pixels,
                    "bbox": bbox,
                    "centroid": (
                        sum(x for _, x in pixels) / len(pixels),
                        sum(y for y, _ in pixels) / len(pixels),
                    ),
                    "width": component_width,
                    "height": component_height,
                    "area": len(pixels),
                    "perimeter": perimeter,
                    "edge_contacts": tuple(edges),
                    "normalized_pixels": normalized,
                    "shape_signature": shape_signature,
                    "rotation_signature": _rotation_signature(normalized),
                    "dihedral_signature": _dihedral_signature(normalized),
                    "primitive_signature": primitive_signature,
                    "primitive_rotation_signature": _rotation_signature(primitive),
                    "primitive_dihedral_signature": _dihedral_signature(primitive),
                    "scale_factor": scale_factor,
                    "shape_kind": shape_kind,
                    "orientation": orientation,
                    "fill_ratio": fill_ratio,
                    "symmetry_horizontal": symmetry_horizontal,
                    "symmetry_vertical": symmetry_vertical,
                    "symmetry_180": symmetry_180,
                    "holes": holes,
                    "enclosed_pixels": enclosed,
                }
            )

    largest_label = max(partial, key=lambda item: (item["area"], -item["label"]))["label"]
    components = [
        _Component(
            **item,
            role_hint=_role_hint(
                item["area"],
                item["width"],
                item["height"],
                item["edge_contacts"],
                height,
                width,
                item["label"] == largest_label,
            ),
        )
        for item in partial
    ]
    return components, labels


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    intersection = intersection_width * intersection_height
    if not intersection:
        return 0.0
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return intersection / (area_a + area_b - intersection)


def _pair_cost(
    track: _Track,
    current: _Component,
    action_number: int,
    frame_diagonal: float,
) -> float:
    previous = track.component
    elapsed = max(1, action_number - track.action_number)
    direct_distance = math.dist(previous.centroid, current.centroid)
    expected = (
        previous.centroid[0] + track.velocity[0] * elapsed,
        previous.centroid[1] + track.velocity[1] * elapsed,
    )
    predicted_distance = math.dist(expected, current.centroid)
    distance = min(direct_distance, predicted_distance * 0.85)
    same_rotation_class = previous.rotation_signature == current.rotation_signature
    same_shape = previous.shape_signature == current.shape_signature
    overlap = _bbox_iou(previous.bbox, current.bbox)
    large = previous.role_hint == "terrain/background" or current.role_hint == "terrain/background"
    allowed_shift = max(3.0 * elapsed, frame_diagonal * (0.50 if large else 0.35 if same_rotation_class else 0.18))

    if distance > allowed_shift and overlap == 0.0:
        return math.inf
    if previous.color_id != current.color_id and direct_distance > max(2.0 * elapsed, 3.0) and overlap == 0.0:
        return math.inf
    area_ratio = max(previous.area, current.area) / max(1, min(previous.area, current.area))
    if area_ratio > 8.0 and not large:
        return math.inf

    color_cost = 0.0 if previous.color_id == current.color_id else 0.28
    if same_shape:
        shape_cost = 0.0
    elif same_rotation_class:
        shape_cost = 0.08
    else:
        shape_cost = 0.35
    area_cost = min(0.18, abs(math.log(area_ratio)) * 0.08)
    distance_cost = 0.46 * min(1.0, distance / max(allowed_shift, 1.0))
    overlap_cost = 0.04 * (1.0 - overlap)
    role_cost = 0.0 if previous.role_hint == current.role_hint else 0.03
    return color_cost + shape_cost + area_cost + distance_cost + overlap_cost + role_cost


def _hungarian(costs: Sequence[Sequence[float]]) -> list[int]:
    """Return the minimum-cost column for each row (columns must cover rows)."""
    row_count = len(costs)
    if not row_count:
        return []
    column_count = len(costs[0])
    if column_count < row_count:
        raise ValueError("Hungarian assignment requires at least as many columns as rows")
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        p[0] = row
        min_values = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                value = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if value < min_values[column]:
                    min_values[column] = value
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            if not math.isfinite(delta):
                raise ValueError("No finite assignment exists")
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def _assign_tracks(
    tracks: Sequence[_Track],
    components: Sequence[_Component],
    action_number: int,
    frame_diagonal: float,
) -> tuple[dict[int, _Track], set[str]]:
    if not tracks or not components:
        return {}, {track.stable_id for track in tracks}

    unmatched_cost = 0.92
    component_count = len(components)
    matches: dict[int, _Track] = {}

    # Exact global assignment is useful for duplicate-looking objects. Fall back to
    # sparse deterministic matching for pathological checkerboards with thousands
    # of one-pixel components.
    if max(len(tracks), component_count) <= 256:
        costs: list[list[float]] = []
        for row, track in enumerate(tracks):
            real_costs = [
                _pair_cost(track, component, action_number, frame_diagonal)
                + column * 1e-10
                for column, component in enumerate(components)
            ]
            dummy_costs = [unmatched_cost + abs(row - dummy) * 1e-12 for dummy in range(len(tracks))]
            costs.append(real_costs + dummy_costs)
        assignment = _hungarian(costs)
        for row, column in enumerate(assignment):
            if column < component_count and costs[row][column] < unmatched_cost:
                matches[column] = tracks[row]
    else:
        # Large component sets (for example a checkerboard of isolated cells)
        # must not fall back to an all-pairs O(n^2) comparison. Resolve exact
        # identity first, then inspect only nearby spatial buckets.
        available = set(range(component_count))
        used_tracks: set[str] = set()

        exact: dict[tuple[int, str, tuple[int, int, int, int]], list[int]] = {}
        for index, component in enumerate(components):
            key = (component.color_id, component.shape_signature, component.bbox)
            exact.setdefault(key, []).append(index)
        for track in sorted(tracks, key=lambda item: item.stable_id):
            previous = track.component
            key = (previous.color_id, previous.shape_signature, previous.bbox)
            indices = [index for index in exact.get(key, []) if index in available]
            if not indices:
                continue
            index = min(indices)
            matches[index] = track
            available.remove(index)
            used_tracks.add(track.stable_id)

        by_bbox: dict[tuple[int, int, int, int], list[int]] = {}
        for index in available:
            by_bbox.setdefault(components[index].bbox, []).append(index)
        for track in sorted(tracks, key=lambda item: item.stable_id):
            if track.stable_id in used_tracks:
                continue
            indices = [
                index
                for index in by_bbox.get(track.component.bbox, [])
                if index in available
            ]
            if len(indices) != 1:
                continue
            index = indices[0]
            matches[index] = track
            available.remove(index)
            used_tracks.add(track.stable_id)

        bucket_size = 4
        buckets: dict[tuple[int, int], list[int]] = {}
        for index in available:
            x, y = components[index].centroid
            buckets.setdefault((int(x) // bucket_size, int(y) // bucket_size), []).append(index)

        candidates: list[tuple[float, str, int, _Track]] = []
        for track in tracks:
            if track.stable_id in used_tracks:
                continue
            elapsed = max(1, action_number - track.action_number)
            previous = track.component
            expected = (
                previous.centroid[0] + track.velocity[0] * elapsed,
                previous.centroid[1] + track.velocity[1] * elapsed,
            )
            radius = min(16, max(4, 4 * elapsed))
            bucket_radius = math.ceil(radius / bucket_size)
            nearby: set[int] = set()
            for center_x, center_y in (previous.centroid, expected):
                base_x = int(center_x) // bucket_size
                base_y = int(center_y) // bucket_size
                for offset_y in range(-bucket_radius, bucket_radius + 1):
                    for offset_x in range(-bucket_radius, bucket_radius + 1):
                        nearby.update(buckets.get((base_x + offset_x, base_y + offset_y), ()))
            for index in nearby:
                cost = _pair_cost(track, components[index], action_number, frame_diagonal)
                if cost < unmatched_cost:
                    candidates.append((cost, track.stable_id, index, track))
        for _, stable_id, index, track in sorted(candidates, key=lambda item: item[:3]):
            if stable_id not in used_tracks and index in available:
                used_tracks.add(stable_id)
                available.remove(index)
                matches[index] = track

    matched_ids = {track.stable_id for track in matches.values()}
    return matches, {track.stable_id for track in tracks if track.stable_id not in matched_ids}


def _relations(
    components: Sequence[_Component], labels: np.ndarray, ids_by_label: Mapping[int, str]
) -> dict[str, dict[str, list[str]]]:
    result = {
        stable_id: {"contains": [], "touches": [], "near": []}
        for stable_id in ids_by_label.values()
    }
    touching_labels: set[tuple[int, int]] = set()
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            source = int(labels[y, x])
            if x + 1 < width:
                target = int(labels[y, x + 1])
                if source != target:
                    touching_labels.add(tuple(sorted((source, target))))
            if y + 1 < height:
                target = int(labels[y + 1, x])
                if source != target:
                    touching_labels.add(tuple(sorted((source, target))))

    for left, right in touching_labels:
        left_id, right_id = ids_by_label[left], ids_by_label[right]
        result[left_id]["touches"].append(right_id)
        result[right_id]["touches"].append(left_id)

    contained_pairs: set[tuple[int, int]] = set()
    for outer in components:
        if not outer.enclosed_pixels:
            continue
        candidate_labels = {
            int(labels[y, x]) for y, x in outer.enclosed_pixels if int(labels[y, x]) != outer.label
        }
        for inner_label in candidate_labels:
            inner = components[inner_label]
            if inner.pixels.issubset(outer.enclosed_pixels):
                contained_pairs.add((outer.label, inner_label))
                result[ids_by_label[outer.label]]["contains"].append(ids_by_label[inner_label])

    near_pairs: set[tuple[int, int]] = set()
    near_radius = 3
    for component in components:
        for y, x in component.pixels:
            for dy in range(-near_radius, near_radius + 1):
                remaining = near_radius - abs(dy)
                for dx in range(-remaining, remaining + 1):
                    distance = abs(dy) + abs(dx)
                    if distance <= 1:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        other = int(labels[ny, nx])
                        if other != component.label:
                            near_pairs.add(tuple(sorted((component.label, other))))

    for left, right in near_pairs:
        if (left, right) in touching_labels or (left, right) in contained_pairs or (right, left) in contained_pairs:
            continue
        left_id, right_id = ids_by_label[left], ids_by_label[right]
        result[left_id]["near"].append(right_id)
        result[right_id]["near"].append(left_id)

    for relation_set in result.values():
        for values in relation_set.values():
            values.sort()
    return result


def _change_record(previous: _Component | None, current: _Component) -> dict[str, Any]:
    if previous is None:
        return {
            "state": "new",
            "events": ["new"],
            "previous_bbox": None,
            "delta": None,
            "rotation_degrees": None,
            "area_delta": None,
            "geometry_changed": None,
            "color_changed": None,
        }

    dx = current.centroid[0] - previous.centroid[0]
    dy = current.centroid[1] - previous.centroid[1]
    moved = not math.isclose(dx, 0.0, abs_tol=1e-9) or not math.isclose(dy, 0.0, abs_tol=1e-9)
    rotation = _rotation_between(previous, current)
    rotated = rotation not in (None, 0)
    geometry_changed = previous.rotation_signature != current.rotation_signature
    color_changed = previous.color_id != current.color_id
    area_delta = current.area - previous.area
    events: list[str] = []
    if moved:
        events.append("moved")
    if rotated:
        events.append("rotated")
    if color_changed:
        events.append("color_changed")
    if geometry_changed:
        events.append("geometry_changed")
    if area_delta:
        events.append("resized")
    if not events:
        events.append("unchanged")
    return {
        "state": "+".join(events),
        "events": events,
        "previous_bbox": list(previous.bbox),
        "delta": [round(dx, 3), round(dy, 3)],
        "rotation_degrees": rotation if rotated else 0 if rotation == 0 else None,
        "area_delta": area_delta,
        "geometry_changed": geometry_changed,
        "color_changed": color_changed,
    }


def build_perceptual_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact semantic view while leaving the raw component census intact."""
    components = list(snapshot.get("components", []))
    frame = snapshot.get("frame", {})
    frame_area = max(1, int(frame.get("width", 0)) * int(frame.get("height", 0)))
    by_id = {str(component["id"]): component for component in components}
    background_roles = {"terrain/background", "edge_region_candidate"}
    background = [
        component
        for component in components
        if component.get("role_hint") in background_roles
    ]
    foreground = [
        component
        for component in components
        if component.get("role_hint") not in background_roles
    ]
    foreground_ids = {str(component["id"]) for component in foreground}

    background_colors: dict[int, dict[str, Any]] = {}
    for component in background:
        color_id = int(component["color_id"])
        item = background_colors.setdefault(
            color_id,
            {
                "color_id": color_id,
                "color_name": str(component["color_name"]),
                "pixels": 0,
            },
        )
        item["pixels"] += int(component["area"])
    background_area = sum(int(component["area"]) for component in background)
    sorted_background_colors = sorted(
        background_colors.values(), key=lambda item: (-item["pixels"], item["color_id"])
    )
    dominant_name = sorted_background_colors[0]["color_name"] if sorted_background_colors else "none"
    background_summary = {
        "description": (
            f"{len(background)} background/terrain region(s), mainly {dominant_name}, "
            f"covering {background_area / frame_area:.1%}"
        ),
        "component_ids": sorted(str(component["id"]) for component in background),
        "component_count": len(background),
        "area": background_area,
        "coverage": round(background_area / frame_area, 4),
        "colors": sorted_background_colors,
        "edge_contacts": sorted(
            {
                str(edge)
                for component in background
                for edge in component.get("edge_contacts", [])
            }
        ),
    }

    parent = {stable_id: stable_id for stable_id in foreground_ids}

    def find(stable_id: str) -> str:
        while parent[stable_id] != stable_id:
            parent[stable_id] = parent[parent[stable_id]]
            stable_id = parent[stable_id]
        return stable_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for component in foreground:
        source = str(component["id"])
        relations = component.get("relations", {})
        for relation_name in ("touches", "contains"):
            for target_value in relations.get(relation_name, []):
                target = str(target_value)
                if target in foreground_ids:
                    union(source, target)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for component in foreground:
        grouped.setdefault(find(str(component["id"])), []).append(component)
    groups = sorted(
        grouped.values(),
        key=lambda group: (
            min(int(component["bbox"][1]) for component in group),
            min(int(component["bbox"][0]) for component in group),
            min(str(component["id"]) for component in group),
        ),
    )

    candidates: list[dict[str, Any]] = []
    candidate_by_component: dict[str, str] = {}
    for sequence, group in enumerate(groups, start=1):
        candidate_id = f"O{sequence:03d}"
        member_ids = sorted(str(component["id"]) for component in group)
        for stable_id in member_ids:
            candidate_by_component[stable_id] = candidate_id
        x0 = min(int(component["bbox"][0]) for component in group)
        y0 = min(int(component["bbox"][1]) for component in group)
        x1 = max(int(component["bbox"][2]) for component in group)
        y1 = max(int(component["bbox"][3]) for component in group)
        area = sum(int(component["area"]) for component in group)
        colors: dict[int, dict[str, Any]] = {}
        for component in group:
            color_id = int(component["color_id"])
            color = colors.setdefault(
                color_id,
                {
                    "color_id": color_id,
                    "color_name": str(component["color_name"]),
                    "pixels": 0,
                },
            )
            color["pixels"] += int(component["area"])
        color_values = sorted(colors.values(), key=lambda item: item["color_id"])
        active_events = sorted(
            {
                str(event)
                for component in group
                for event in component.get("change", {}).get("events", [])
                if event not in {"new", "unchanged"}
            }
        )
        roles = sorted({str(component.get("role_hint", "object_candidate")) for component in group})
        shape_kinds = sorted({str(component.get("shape_kind", "unknown")) for component in group})
        contained_parts = sorted(
            {
                (str(component["id"]), str(inner_id))
                for component in group
                for inner_id in component.get("relations", {}).get("contains", [])
                if str(inner_id) in member_ids
            }
        )
        intrinsic_symmetries = [
            {
                "component_id": str(component["id"]),
                "horizontal": bool(component.get("symmetry", {}).get("horizontal")),
                "vertical": bool(component.get("symmetry", {}).get("vertical")),
                "rotation_180": bool(component.get("symmetry", {}).get("rotation_180")),
            }
            for component in group
            if any(bool(value) for value in component.get("symmetry", {}).values())
        ]
        if "edge_ui" in roles:
            description = "edge UI group"
        elif len(color_values) > 1:
            description = "multi-color composite"
        elif len(group) > 1:
            description = "compound same-color object"
        else:
            description = shape_kinds[0]
        salience = 3.0
        reasons: list[str] = []
        if len(color_values) > 1:
            salience += 2.0
            reasons.append("multi_color")
        if active_events:
            salience += 2.0
            reasons.append("changed")
        if "edge_ui" in roles:
            salience += 1.0
            reasons.append("edge_ui")
        if any(int(component.get("holes", 0)) for component in group):
            salience += 1.0
            reasons.append("contains_space")
        if area / frame_area <= 0.02:
            salience += 1.0
            reasons.append("small_distinct_region")
        candidates.append(
            {
                "id": candidate_id,
                "description": description,
                "component_ids": member_ids,
                "bbox": [x0, y0, x1, y1],
                "centroid": [
                    round(sum(float(component["centroid"][0]) * int(component["area"]) for component in group) / area, 3),
                    round(sum(float(component["centroid"][1]) * int(component["area"]) for component in group) / area, 3),
                ],
                "area": area,
                "colors": color_values,
                "shape_kinds": shape_kinds,
                "contained_parts": [list(pair) for pair in contained_parts],
                "intrinsic_symmetries": intrinsic_symmetries,
                "roles": roles,
                "change_events": active_events,
                "salience_score": round(min(10.0, salience), 1),
                "salience_reasons": reasons,
            }
        )

    def geometry_group(
        relationship: str, members: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        member_ids = sorted(str(component["id"]) for component in members)
        return {
            "relationship": relationship,
            "component_ids": member_ids,
            "candidate_ids": sorted(
                {candidate_by_component[stable_id] for stable_id in member_ids}
            ),
            "shape_kinds": sorted({str(component.get("shape_kind", "unknown")) for component in members}),
            "orientations": sorted({str(component.get("orientation", "unknown")) for component in members}),
            "colors": sorted({str(component.get("color_name", "unknown")) for component in members}),
        }

    geometry_matches: list[dict[str, Any]] = []

    def grouped_by(field: str) -> list[list[Mapping[str, Any]]]:
        families: dict[str, list[Mapping[str, Any]]] = {}
        for component in foreground:
            if component.get("role_hint") != "object_candidate" or int(component.get("area", 0)) < 2:
                continue
            key = component.get(field)
            if key is not None:
                families.setdefault(str(key), []).append(component)
        return [members for _, members in sorted(families.items()) if len(members) >= 2]

    for members in grouped_by("shape_signature"):
        geometry_matches.append(geometry_group("repeated", members))
    for members in grouped_by("rotation_signature"):
        if len({str(component.get("shape_signature")) for component in members}) >= 2:
            geometry_matches.append(geometry_group("rotated", members))
    for members in grouped_by("dihedral_signature"):
        if len({str(component.get("rotation_signature")) for component in members}) >= 2:
            geometry_matches.append(geometry_group("reflected", members))
    for members in grouped_by("primitive_dihedral_signature"):
        scales = {int(component.get("scale_factor", 1)) for component in members}
        if len(scales) >= 2:
            match = geometry_group("scaled", members)
            match["scale_factors"] = sorted(scales)
            geometry_matches.append(match)

    exact_candidate_pairs = {
        tuple(sorted((left, right)))
        for match in geometry_matches
        for index, left in enumerate(match["candidate_ids"])
        for right in match["candidate_ids"][index + 1 :]
    }

    def coarse_shape_families(candidate: Mapping[str, Any]) -> set[str]:
        families: set[str] = set()
        for shape in candidate["shape_kinds"]:
            text = str(shape).lower()
            if "container" in text or "hollow" in text or "frame" in text:
                families.add("container")
            elif "l-like" in text or "t-like" in text or "cross" in text or "angular" in text:
                families.add("angular")
            elif "point" in text or "square" in text:
                families.add("inner_mark")
            elif "rectangle" in text or "line" in text:
                families.add("bar")
            else:
                families.add(text)
        return families

    semantic_candidates = [
        candidate for candidate in candidates if "edge_ui" not in candidate["roles"]
    ]
    for left_index, left in enumerate(semantic_candidates):
        for right in semantic_candidates[left_index + 1 :]:
            pair = tuple(sorted((str(left["id"]), str(right["id"]))))
            if pair in exact_candidate_pairs:
                continue
            left_colors = {item["color_name"] for item in left["colors"]}
            right_colors = {item["color_name"] for item in right["colors"]}
            shared_colors = sorted(left_colors & right_colors)
            left_families = coarse_shape_families(left)
            right_families = coarse_shape_families(right)
            shared_families = sorted(left_families & right_families)
            if len(shared_colors) < 2 or not {"container", "angular"}.issubset(
                set(shared_families)
            ):
                continue
            transforms = ["rotation", "reflection"]
            larger_area = max(int(left["area"]), int(right["area"]))
            smaller_area = max(1, min(int(left["area"]), int(right["area"])))
            if larger_area / smaller_area >= 1.5:
                transforms.append("scale")
            geometry_matches.append(
                {
                    "relationship": "composite_analogy",
                    "candidate_ids": list(pair),
                    "component_ids": sorted(
                        set(left["component_ids"]) | set(right["component_ids"])
                    ),
                    "shared_colors": shared_colors,
                    "shared_shape_families": shared_families,
                    "possible_transforms": transforms,
                }
            )

    relationship_order = {
        "repeated": 0,
        "rotated": 1,
        "reflected": 2,
        "scaled": 3,
        "composite_analogy": 4,
    }
    geometry_matches.sort(
        key=lambda item: (
            relationship_order[item["relationship"]],
            item["component_ids"],
        )
    )
    candidate_lookup = {candidate["id"]: candidate for candidate in candidates}
    for match in geometry_matches:
        relationship = str(match["relationship"])
        for candidate_id in match["candidate_ids"]:
            candidate = candidate_lookup[candidate_id]
            reason = f"geometry_{relationship}"
            if reason not in candidate["salience_reasons"]:
                candidate["salience_reasons"].append(reason)
                candidate["salience_score"] = round(
                    min(10.0, float(candidate["salience_score"]) + 2.0), 1
                )
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["salience_score"]),
            int(candidate["bbox"][1]),
            int(candidate["bbox"][0]),
        )
    )

    color_families: dict[int, list[Mapping[str, Any]]] = {}
    for component in foreground:
        color_families.setdefault(int(component["color_id"]), []).append(component)
    background_color_ids = set(background_colors)
    salient_colors: list[dict[str, Any]] = []
    for color_id, members in sorted(color_families.items()):
        pixels = sum(int(component["area"]) for component in members)
        candidate_ids = sorted(
            {candidate_by_component[str(component["id"])] for component in members}
        )
        active_events = sorted(
            {
                str(event)
                for component in members
                for event in component.get("change", {}).get("events", [])
                if event not in {"new", "unchanged"}
            }
        )
        reasons: list[str] = []
        score = 3.0
        if color_id not in background_color_ids:
            score += 2.0
            reasons.append("distinct_from_background")
        if pixels / frame_area <= 0.02:
            score += 1.5
            reasons.append("rare_color")
        if active_events:
            score += 2.0
            reasons.append("changed")
        if len(members) >= 2:
            score += 0.5
            reasons.append("repeated_regions")
        if any(component.get("role_hint") == "edge_ui" for component in members):
            score += 1.0
            reasons.append("edge_ui")
        salient_colors.append(
            {
                "color_id": color_id,
                "color_name": str(members[0]["color_name"]),
                "pixels": pixels,
                "coverage": round(pixels / frame_area, 4),
                "component_ids": sorted(str(component["id"]) for component in members),
                "candidate_ids": candidate_ids,
                "change_events": active_events,
                "salience_score": round(min(10.0, score), 1),
                "salience_reasons": reasons,
            }
        )
    salient_colors.sort(key=lambda item: (-item["salience_score"], item["color_id"]))

    return {
        "background": background_summary,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "geometry_matches": geometry_matches,
        "salient_color_groups": salient_colors,
        "raw_component_count": len(components),
    }


class SceneAnalyzer:
    """Inventory all 4-connected color components and track them across frames."""

    def __init__(self, color_names: Mapping[int, str] | None = None) -> None:
        self.color_names = dict(ARC_COLOR_NAMES)
        if color_names:
            self.color_names.update({int(key): str(value) for key, value in color_names.items()})
        self.reset_level()

    def reset_level(self) -> None:
        """Clear temporal identity state before the first frame of a new level."""
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_action_number: int | None = None
        self._last_color_counts: dict[int, int] = {}

    def analyze(self, frame: np.ndarray, action_number: int) -> dict[str, Any]:
        """Return a compact, JSON-serializable inventory for one 2D color grid."""
        array = np.asarray(frame)
        if array.ndim != 2:
            raise ValueError(f"frame must be a 2D color grid, got shape {array.shape}")
        if not array.size:
            raise ValueError("frame must not be empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("frame values must be numeric ARC color ids")
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("frame values must be finite integer ARC color ids")
        action_number = int(action_number)
        if self._last_action_number is not None and action_number < self._last_action_number:
            raise ValueError("action_number decreased; call reset_level() before a new level")
        integer_frame = array.astype(np.int64, copy=False)
        components, labels = _extract_components(integer_frame)
        height, width = integer_frame.shape
        frame_diagonal = math.hypot(width, height)
        matches, disappeared_ids = _assign_tracks(
            self._tracks, components, action_number, frame_diagonal
        )

        previous_by_id = {track.stable_id: track for track in self._tracks}
        ids_by_label: dict[int, str] = {}
        for index, component in enumerate(components):
            matched = matches.get(index)
            if matched is not None:
                stable_id = matched.stable_id
            else:
                stable_id = f"C{self._next_id:03d}"
                self._next_id += 1
            ids_by_label[component.label] = stable_id
        if len(set(ids_by_label.values())) != len(components):
            raise AssertionError("component identities must be one-to-one within a frame")

        relations = _relations(components, labels, ids_by_label)
        records: list[dict[str, Any]] = []
        new_tracks: list[_Track] = []
        event_index: dict[str, list[str]] = {
            "new": [],
            "moved": [],
            "rotated": [],
            "color_changed": [],
            "geometry_changed": [],
            "resized": [],
            "unchanged": [],
        }
        for index, component in enumerate(components):
            stable_id = ids_by_label[component.label]
            prior_track = matches.get(index)
            previous = prior_track.component if prior_track else None
            change = _change_record(previous, component)
            for event in change["events"]:
                event_index[event].append(stable_id)
            records.append(
                {
                    "id": stable_id,
                    "color_id": component.color_id,
                    "color_name": self.color_names.get(component.color_id, f"color_{component.color_id}"),
                    "bbox": list(component.bbox),
                    "centroid": [round(component.centroid[0], 3), round(component.centroid[1], 3)],
                    "width": component.width,
                    "height": component.height,
                    "area": component.area,
                    "perimeter": component.perimeter,
                    "edge_contacts": list(component.edge_contacts),
                    "shape_signature": component.shape_signature,
                    "rotation_signature": component.rotation_signature,
                    "dihedral_signature": component.dihedral_signature,
                    "primitive_signature": component.primitive_signature,
                    "primitive_rotation_signature": component.primitive_rotation_signature,
                    "primitive_dihedral_signature": component.primitive_dihedral_signature,
                    "scale_factor": component.scale_factor,
                    "shape_kind": component.shape_kind,
                    "orientation": component.orientation,
                    "fill_ratio": round(component.fill_ratio, 4),
                    "symmetry": {
                        "horizontal": component.symmetry_horizontal,
                        "vertical": component.symmetry_vertical,
                        "rotation_180": component.symmetry_180,
                    },
                    "holes": component.holes,
                    "role_hint": component.role_hint,
                    "relations": relations[stable_id],
                    "change": change,
                }
            )
            if prior_track:
                elapsed = max(1, action_number - prior_track.action_number)
                velocity = (
                    (component.centroid[0] - previous.centroid[0]) / elapsed,
                    (component.centroid[1] - previous.centroid[1]) / elapsed,
                )
            else:
                velocity = (0.0, 0.0)
            new_tracks.append(_Track(stable_id, component, action_number, velocity))

        disappeared: list[dict[str, Any]] = []
        for stable_id in sorted(disappeared_ids):
            prior_track = previous_by_id[stable_id]
            previous = prior_track.component
            if prior_track.missed_analyses == 0:
                disappeared.append(
                    {
                        "id": stable_id,
                        "color_id": previous.color_id,
                        "color_name": self.color_names.get(previous.color_id, f"color_{previous.color_id}"),
                        "last_bbox": list(previous.bbox),
                    }
                )
            missed_analyses = prior_track.missed_analyses + 1
            if missed_analyses <= 2:
                new_tracks.append(
                    _Track(
                        stable_id=stable_id,
                        component=previous,
                        action_number=prior_track.action_number,
                        velocity=prior_track.velocity,
                        missed_analyses=missed_analyses,
                    )
                )

        color_summary: list[dict[str, Any]] = []
        current_color_counts: dict[int, int] = {}
        for color_id in sorted(int(value) for value in np.unique(integer_frame)):
            pixels = int(np.count_nonzero(integer_frame == color_id))
            current_color_counts[color_id] = pixels
            color_summary.append(
                {
                    "color_id": color_id,
                    "color_name": self.color_names.get(color_id, f"color_{color_id}"),
                    "pixels": pixels,
                    "components": sum(component.color_id == color_id for component in components),
                }
            )
        color_deltas = []
        for color_id in sorted(set(self._last_color_counts) | set(current_color_counts)):
            before = self._last_color_counts.get(color_id, 0)
            after = current_color_counts.get(color_id, 0)
            if before != after:
                color_deltas.append(
                    {
                        "color_id": color_id,
                        "color_name": self.color_names.get(color_id, f"color_{color_id}"),
                        "previous_pixels": before,
                        "current_pixels": after,
                        "delta_pixels": after - before,
                    }
                )

        snapshot: dict[str, Any] = {
            "action_number": action_number,
            "frame": {"width": width, "height": height},
            "component_count": len(records),
            "colors_present": color_summary,
            "color_deltas": color_deltas,
            "components": records,
            "changes": {
                **event_index,
                "disappeared": disappeared,
            },
        }
        snapshot["perceptual_summary"] = build_perceptual_summary(snapshot)
        self._tracks = new_tracks
        self._last_action_number = action_number
        self._last_color_counts = current_color_counts
        return snapshot

    @staticmethod
    def to_markdown(snapshot: Mapping[str, Any]) -> str:
        """Render an analyzer snapshot as a compact Markdown table."""
        return snapshot_to_markdown(snapshot)

    @staticmethod
    def summarize(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Build or rebuild the compact perceptual layer from a raw snapshot."""
        return build_perceptual_summary(snapshot)


def snapshot_to_markdown(snapshot: Mapping[str, Any]) -> str:
    """Render every component in a JSON snapshot without filtering by importance."""
    frame = snapshot["frame"]
    lines = [
        f"# Scene inventory - action {snapshot['action_number']}",
        "",
        f"Frame: {frame['width']}x{frame['height']} | Components: {snapshot['component_count']}",
        "",
        "| ID | Color | BBox `(x0,y0,x1,y1)` | Area | Geometry | Orientation | Fill | Symmetry | Holes | Role | Change | Relations |",
        "|---|---|---|---:|---|---|---:|---|---:|---|---|---|",
    ]
    for component in snapshot["components"]:
        relation_parts = []
        for name in ("contains", "touches", "near"):
            values = component["relations"][name]
            if values:
                relation_parts.append(f"{name}: {','.join(values)}")
        relations = "; ".join(relation_parts) or "-"
        bbox = ",".join(str(value) for value in component["bbox"])
        symmetry = ",".join(
            label
            for key, label in (("horizontal", "H"), ("vertical", "V"), ("rotation_180", "R180"))
            if component["symmetry"][key]
        ) or "none"
        lines.append(
            f"| {component['id']} | {component['color_name']} ({component['color_id']}) | "
            f"({bbox}) | {component['area']} | {component['shape_kind']} "
            f"`{component['shape_signature']}` | {component['orientation']} | "
            f"{component['fill_ratio']:.3f} | {symmetry} | {component['holes']} | "
            f"{component['role_hint']} | "
            f"{component['change']['state']} | {relations} |"
        )
    disappeared = snapshot["changes"]["disappeared"]
    if disappeared:
        lines.extend(
            [
                "",
                "Disappeared: " + ", ".join(item["id"] for item in disappeared),
            ]
        )
    color_deltas = snapshot.get("color_deltas", [])
    if color_deltas:
        lines.extend(
            [
                "",
                "Color deltas: "
                + ", ".join(
                    f"{item['color_name']} {item['delta_pixels']:+d}"
                    for item in color_deltas
                ),
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ARC_COLOR_NAMES",
    "SceneAnalyzer",
    "build_perceptual_summary",
    "snapshot_to_markdown",
]
