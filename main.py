import importlib
import math
import time

import numpy as np
from sympy import im
from ETS2LA.Plugin import ETS2LAPlugin
from ETS2LA.Plugin import PluginDescription
from ETS2LA.Plugin import Author


from ETS2LA.Utils.Console import logging
from Plugins.Map.settings import settings
import Plugins.Map.utils.data_handler as data_handler
import Plugins.Map.utils.data_reader as data_reader
import Plugins.Map.classes as c
from ETS2LA.Utils.translator import _


from Plugins.Map import main

planning = importlib.import_module("Plugins.Map.route.planning")
driving = importlib.import_module("Plugins.Map.route.driving")

class Plugin(ETS2LAPlugin):

    description = PluginDescription(
        name="[Experimental] High-Speed Map Mode",
        version="1.0.0",
        description="This overrides the steering points calculation to allow for higher speeds.",
        modules=["Traffic", "TruckSimAPI", "SDKController"],
        listen=["*.py"],
        tags=["Base"],
        fps_cap=15
    )

    author = Author(
        name="Playzzero97",
        url="https://github.com/Playzzero97",
        icon="https://avatars.githubusercontent.com/u/219891638?v=4"
    )

    def init(self):
        global api, steering
        api = self.modules.TruckSimAPI
        api.TRAILER = True
        steering = self.modules.Steering
        steering.OFFSET = 0
        steering.SMOOTH_TIME = self.steering_smoothness
        steering.IGNORE_SMOOTH = False
        steering.SENSITIVITY = 1.2
        self.api = self.modules.TruckSimAPI


    def override_map_run(self):
        import types

        main.Plugin.run = types.MethodType(self.newrun, main.Plugin)

        plugins_dict = getattr(main, "plugins", None)
        if plugins_dict:
            for p in plugins_dict.values():
                if isinstance(p, main.Plugin):
                    p.run = types.MethodType(self.newrun, p)


    def newrun(self, *args, **kwargs):
        import Plugins.Map.data as data
        start = time.perf_counter()
        now = time.time()

        FRAME_BUDGET = 0.03
        PLANNING_INTERVAL = 0.25
        ROAD_INTERVAL = 0.20
        CITY_INTERVAL = 5.0
        EXTERNAL_MAP_INTERVAL = 0.25

        is_different_data = (
            settings.selected_data != ""
            and settings.selected_data != settings.downloaded_data
        )

        if not data.data_downloaded or is_different_data:
            data.data_downloaded = False

            if data_handler.IsDownloaded(data.data_path) and not is_different_data:
                self.state.text = _("Preparing to load data... (OVERRIDDEN)")
                data_reader.path = data.data_path
                data.map = data_reader.ReadData(state=self.state)
                data.data_downloaded = True
                data.data_needs_update = True
                self.state.reset()
                return

            if settings.selected_data:
                data_handler.UpdateData(settings.selected_data)
                return

            self.state.text = _("Waiting for game data selection in the Settings -> Map...")
            return

        try:
            api_data = self.api.run()
            data.UpdateData(api_data)
        except Exception:
            logging.exception("API update failed")
            return

        if data.calculate_steering:
            if (self.route_future is None or self.route_future.done()) and now - self.last_planning_update > PLANNING_INTERVAL:
                self.route_future = self.executor.submit(planning.UpdateRoutePlan)
                self.last_planning_update = now
                self._last_route_update_time = now
                self._last_valid_route_time = now

        if data.calculate_steering:
            steering_value = driving.GetSteering()
            if steering_value is not None:
                steering_value = max(-0.95, min(0.95, steering_value / 180))
                self.tags.steering = steering_value
                steering.run(
                    value=steering_value,
                    sendToGame=data.enabled,
                    drawLine=False,
                )



        if data.internal_map:
            self.MapWindowInitialization()
            im.DrawMap()

        self.UpdateNavigation()

        if data.external_data_changed and now - self.last_external_map_update > EXTERNAL_MAP_INTERVAL:
            self.tags.map = data.external_data
            self.tags.map_update_time = data.external_data_time
            data.external_data_changed = False
            self.last_external_map_update = now

        if data.route_points:
            self.tags.steering_points = [p.tuple() for p in data.route_points]
        else:
            self.tags.steering_points = []

        self._last_route_points = self.tags.steering_points

        if now - self.last_city_update > CITY_INTERVAL:
            self.last_city_update = now
            try:
                closest = None
                closest_distance = math.inf
                for city in data.map.cities:
                    d = (data.truck_x - city.x) ** 2 + (data.truck_z - city.y) ** 2
                    if d < closest_distance:
                        closest_distance = d
                        closest = city

                if closest:
                    self.tags.closest_city = closest.name
                    self.tags.closest_city_distance = math.sqrt(closest_distance)
                    self.tags.closest_country = closest.country_token.capitalize()
            except Exception:
                pass

        if now - self.last_road_update > ROAD_INTERVAL:
            self.last_road_update = now
            try:
                roads = data.current_sector_roads
                prefabs = data.current_sector_prefabs

                if roads and prefabs:
                    xy = c.Position(data.truck_x, data.truck_z, data.truck_y)
                    found = any(r.bounding_box.is_in(xy) for r in roads) or any(
                        p.bounding_box.is_in(xy) for p in prefabs
                    )

                    if found:
                        self.tags.closest_road_distance = 0
                        self.tags.closest_road_angle = 0
                    else:
                        truck = c.Position(data.truck_x, data.truck_y, data.truck_z)
                        road = min(roads, key=lambda r: r.distance_to(truck))
                        self.tags.closest_road_distance = road.distance_to(xy)

                        point = min(
                            road.points, key=lambda p: p.distance_to(truck)
                        ) - truck

                        forward = np.array(
                            [-math.sin(data.truck_rotation), -math.cos(data.truck_rotation)]
                        )
                        to_road = np.array(point.tuple(xz=True))

                        forward /= np.linalg.norm(forward)
                        to_road /= np.linalg.norm(to_road)

                        self.tags.closest_road_angle = np.degrees(
                            np.arccos(np.clip(np.dot(forward, to_road), -1.0, 1.0))
                        )
                else:
                    self.tags.closest_road_distance = 0
                    self.tags.closest_road_angle = 0
            except Exception:
                self.tags.closest_road_distance = 0
                self.tags.closest_road_angle = 0

        if time.perf_counter() - start > FRAME_BUDGET:
            return


    def run(self): 
        self.override_map_run()